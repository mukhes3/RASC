"""
build_dataset.py
================
Constructs train/validation/test splits for the value set code inclusion classifier.

Pipeline per target value set:
  1. Embed title with SAPBert
  2. Retrieve top-K most similar value sets (excluding itself)
  3. Union all codes from retrieved sets → candidate pool
  4. Label each candidate: 1 if in target's true expansion, 0 otherwise
  5. Record retrieval_recall@k = |true_codes ∩ candidates| / |true_codes|

Split strategy:
  - Unit of splitting: value set (never split a value set's codes across folds)
  - Stratify by publisher, then by inferred type within each publisher split
  - Hold-out publishers: 2 largest publishers go exclusively to test
    (tests generalisation across organisations)
  - Remaining value sets: 70/15/15 train/val/test by stratified sampling

──────────────────────────────────────────────────────────────────────────────
DECOMPOSED STORAGE FORMAT  (replaces the old monolithic X.npz approach)
──────────────────────────────────────────────────────────────────────────────
Instead of materialising the full (N × 1545) feature matrix, we store three
separate artefacts and reconstruct X on-the-fly during training/eval:

  title_embs.npz
      Arrays: 'vecs'  float32 (V_t × 768) – one row per unique title
              'keys'  str     (V_t,)       – parallel title strings
      One entry per unique value-set title across ALL splits.

  code_embs.npz
      Arrays: 'vecs'  float32 (V_c × 768) – one row per unique display string
              'keys'  str     (V_c,)       – parallel display strings
      One entry per unique code display string across ALL splits.

  {split}_meta.pkl   (train / val / test)
      List[dict], one dict per value set:
        oid           str   – value set OID (with date suffix)
        split         str   – "train" | "val" | "test"
        title         str   – value set title (lookup key into title_embs)
        n_pos         int
        n_neg         int
        true_codes    List[{code, system}]
        candidates    List[{code, system, display, score, label}]
                        display  → lookup key into code_embs
                        score    → float32 retrieval similarity
                        label    → int8  (1 = in true expansion, 0 = not)

  split_manifest.jsonl  – one JSON line per value set (lightweight summary)
  dataset_stats.json    – statistics for the paper
  sanity_check_results.json

Feature reconstruction (use VSDataset below, or replicate in your own loader):
  X[i] = concat(title_embs[title], code_embs[display_i], onehot(system_i), score_i)
        = 768 + 768 + 8 + 1 = 1545 dims  (identical to the old format)

Space comparison (full ~9.5 k value-set corpus, top-k=10):
  Old format : ~10 GB  (train.npz alone)
  New format : ~50 MB  (embeddings) + ~20 MB (meta pkls)  ≈ 200× smaller

──────────────────────────────────────────────────────────────────────────────
Usage:
    python create_dataset/build_dataset.py \
        --vsac-dir vsac_data \
        --index-dir vsac_index \
        --strategy title \
        --top-k 10 \
        --out-dir dataset \
        --holdout-publishers "Clinical Architecture" "CSTE Steward"

Requirements:
    pip install faiss-cpu sentence-transformers torch numpy tqdm scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Code system vocabulary
# ---------------------------------------------------------------------------

CODE_SYSTEM_VOCAB = [
    "SNOMED-CT",
    "ICD-10-CM",
    "RxNorm",
    "LOINC",
    "CPT",
    "ICD-10-PCS",
    "HCPCS",
    "OTHER",
]
CODE_SYSTEM_TO_IDX = {cs: i for i, cs in enumerate(CODE_SYSTEM_VOCAB)}


def normalise_code_system(raw: str) -> str:
    uri_map = {
        "http://snomed.info/sct":                      "SNOMED-CT",
        "http://hl7.org/fhir/sid/icd-10-cm":          "ICD-10-CM",
        "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
        "http://loinc.org":                            "LOINC",
        "http://www.ama-assn.org/go/cpt":              "CPT",
        "http://hl7.org/fhir/sid/icd-10-pcs":         "ICD-10-PCS",
        "http://www.cms.gov/Medicare/Coding/ICD10":    "ICD-10-PCS",
        "http://www.nlm.nih.gov/research/umls/hcpcs":  "HCPCS",
    }
    if raw in uri_map:
        return uri_map[raw]
    for vocab_name in CODE_SYSTEM_VOCAB[:-1]:
        if vocab_name.lower() in raw.lower():
            return vocab_name
    return "OTHER"


def code_system_onehot(cs_raw: str) -> np.ndarray:
    idx = CODE_SYSTEM_TO_IDX.get(
        normalise_code_system(cs_raw), CODE_SYSTEM_TO_IDX["OTHER"]
    )
    vec = np.zeros(len(CODE_SYSTEM_VOCAB), dtype=np.float32)
    vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Infer value set type from primary code system
# ---------------------------------------------------------------------------

def infer_type(code_systems: list) -> str:
    primary = normalise_code_system(code_systems[0]) if code_systems else "OTHER"
    return {
        "SNOMED-CT":  "Condition/Clinical",
        "ICD-10-CM":  "Condition/Diagnosis",
        "RxNorm":     "Medication",
        "LOINC":      "Lab/Observation",
        "CPT":        "Procedure",
        "ICD-10-PCS": "Procedure",
        "HCPCS":      "Administrative",
    }.get(primary, "Other/Unknown")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_index_records(vsac_dir: Path) -> list:
    path = vsac_dir / "index.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_exp_path(oid: str, exp_dir: Path):
    exact = exp_dir / f"{oid}.json"
    if exact.exists():
        return exact
    matches = sorted(exp_dir.glob(f"{oid}-*.json"))
    return matches[0] if matches else None


def load_true_codes(oid: str, exp_dir: Path) -> List[dict]:
    path = find_exp_path(oid, exp_dir)
    if path is None:
        return []
    with open(path) as f:
        resource = json.load(f)
    result = []
    for c in resource.get("expansion", {}).get("contains", []):
        if c.get("code"):
            result.append({
                "code":    c["code"],
                "display": c.get("display", ""),
                "system":  c.get("system", ""),
            })
    return result


def load_model_for_inference(model_name: str):
    """
    Load a SentenceTransformer with correct pooling.
    SAPBert requires CLS pooling; patch explicitly since HF has no ST config.
    """
    from sentence_transformers import SentenceTransformer, models

    sapbert_ids = {
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    }
    if model_name in sapbert_ids:
        log.info("Loading SAPBert with explicit CLS pooling...")
        word_embedding = models.Transformer(model_name, max_seq_length=512)
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False,
        )
        return SentenceTransformer(modules=[word_embedding, pooling])
    return SentenceTransformer(model_name)


def load_built_index(index_dir: Path, strategy: str):
    index = faiss.read_index(str(index_dir / f"index_{strategy}.faiss"))
    with open(index_dir / f"metadata_{strategy}.pkl", "rb") as f:
        meta = pickle.load(f)
    model = load_model_for_inference(meta["model"])
    return index, meta["records"], meta["texts"], model


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(index, records, model, n_samples=200, seed=42) -> dict:
    log.info("Running sanity check (n=%d)...", n_samples)
    rng = random.Random(seed)

    def random_case(s):
        return "".join(c.upper() if rng.random() > 0.5 else c.lower() for c in s)

    perturbations = {
        "lowercase":   str.lower,
        "uppercase":   str.upper,
        "title_case":  str.title,
        "random_case": random_case,
    }

    oid_to_idx = {rec["oid"]: i for i, rec in enumerate(records)}

    from collections import Counter as _Counter
    title_counts = _Counter(r.get("title", "").strip().lower() for r in records)
    unique_records = [
        r for r in records
        if title_counts[r.get("title", "").strip().lower()] == 1
    ]
    log.info("Sanity check: %d unique-title records (from %d total)",
             len(unique_records), len(records))
    sample  = rng.sample(unique_records, min(n_samples, len(unique_records)))
    results = {name: {"correct": 0, "total": 0} for name in perturbations}

    for rec in tqdm(sample, desc="Sanity check"):
        oid   = rec["oid"]
        title = rec.get("title", "").strip()
        if not title or oid not in oid_to_idx:
            continue
        for pert_name, fn in perturbations.items():
            emb = model.encode([fn(title)], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            _, idxs = index.search(emb, 1)
            results[pert_name]["total"] += 1
            if records[idxs[0][0]]["oid"] == oid:
                results[pert_name]["correct"] += 1

    summary      = {}
    overall_pass = True
    for name, counts in results.items():
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        summary[name] = {"accuracy": round(acc, 4), **counts}
        if acc < 0.95:
            log.warning("FAILED %s: top-1 acc = %.1f%%", name, acc * 100)
            overall_pass = False
        else:
            log.info("PASSED %s: %.1f%%", name, acc * 100)

    summary["overall_pass"] = overall_pass
    return summary


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(records, holdout_publishers, val_frac=0.15, test_frac=0.15, seed=42):
    rng         = np.random.default_rng(seed)
    assignment  = {}
    holdout_set = set(holdout_publishers)
    remaining   = []

    for rec in records:
        if rec.get("publisher", "") in holdout_set:
            assignment[rec["oid"]] = "test"
        else:
            remaining.append(rec)

    log.info("Holdout → test: %d  |  remaining: %d",
             len(records) - len(remaining), len(remaining))

    if not remaining:
        return assignment

    pub_counts = Counter(r.get("publisher", "") for r in remaining)
    top_pubs   = set(sorted(pub_counts, key=pub_counts.__getitem__, reverse=True)[:10])

    def strat_key(rec):
        t   = infer_type(rec.get("code_systems") or [])
        pub = rec.get("publisher", "")
        return f"{t}|{pub if pub in top_pubs else 'other_publisher'}"

    oids      = [r["oid"] for r in remaining]
    keys      = [strat_key(r) for r in remaining]
    key_cnts  = Counter(keys)
    keys_safe = [k if key_cnts[k] >= 4 else "rare" for k in keys]

    _rng_state = int(rng.integers(1 << 31))
    n_classes  = len(set(keys_safe))
    n_test     = max(1, int(len(oids) * test_frac))
    _stratify  = keys_safe if n_test >= n_classes else None
    if _stratify is None:
        log.warning("Falling back to non-stratified test split: n_test=%d < n_classes=%d",
                    n_test, n_classes)
    tv_oids, te_oids, tv_keys, _ = train_test_split(
        oids, keys_safe, test_size=test_frac, stratify=_stratify, random_state=_rng_state)
    for oid in te_oids:
        assignment[oid] = "test"

    tr_oids, va_oids = train_test_split(
        tv_oids, test_size=val_frac / (1.0 - test_frac), stratify=tv_keys,
        random_state=int(rng.integers(1 << 31)))
    for oid in tr_oids:
        assignment[oid] = "train"
    for oid in va_oids:
        assignment[oid] = "val"

    log.info("Split value-set counts: %s", dict(Counter(assignment.values())))
    return assignment


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(target_oid, target_title, index, records, model,
                        exp_dir, top_k) -> Tuple[list, list]:
    emb = model.encode([target_title], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    scores, indices = index.search(emb, top_k + 1)

    candidates       = []
    sim_scores       = []
    seen_source_oids = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        rec = records[idx]
        if rec["oid"] == target_oid:
            continue
        if len(seen_source_oids) >= top_k:
            break
        seen_source_oids.add(rec["oid"])
        for c in load_true_codes(rec["oid"], exp_dir):
            candidates.append({**c, "source_oid": rec["oid"]})
            sim_scores.append(float(score))

    return candidates, sim_scores


def retrieval_recall_at_k(true_codes: list, candidates: list) -> float:
    if not true_codes:
        return float("nan")
    true_set = {c["code"] for c in true_codes}
    cand_set = {c["code"] for c in candidates}
    return len(true_set & cand_set) / len(true_set)


def deduplicate_candidates(candidates: list, sim_scores: list) -> list:
    """Collapse (code, system) duplicates, retaining highest sim_score."""
    best: Dict[tuple, dict] = {}
    for c, score in zip(candidates, sim_scores):
        key = (c["code"], c["system"])
        if key not in best or score > best[key]["score"]:
            best[key] = {**c, "score": score}
    return list(best.values())


# ---------------------------------------------------------------------------
# VSDataset  — import this in training / eval notebooks
# ---------------------------------------------------------------------------

class VSDataset:
    """
    PyTorch-compatible dataset that reconstructs X on-the-fly from the
    decomposed storage format.  No large matrix ever lives in RAM.

    Parameters
    ----------
    meta_path        : path to {split}_meta.pkl
    title_embs_path  : path to title_embs.npz
    code_embs_path   : path to code_embs.npz

    Each __getitem__ returns (x, y) where x is (1545,) float32 and y is scalar.
    Use with a standard DataLoader; shuffle=True for train splits.

    For LLM / retrieval-only eval you do NOT need this class — just iterate
    over the meta list directly; each entry has true_codes and candidates with
    per-code labels so you can compute precision/recall without any embeddings.

    Example (classifier training)
    ──────────────────────────────
        ds   = VSDataset("dataset/train_meta.pkl",
                         "dataset/title_embs.npz",
                         "dataset/code_embs.npz")
        loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=4)
        for x, y in loader:
            ...

    Example (LLM / retrieval eval — no embeddings needed)
    ───────────────────────────────────────────────────────
        with open("dataset/test_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        for entry in meta:
            true_set  = {(c["code"], c["system"]) for c in entry["true_codes"]}
            predicted = llm_predict(entry["title"], entry["candidates"])
            # compute precision / recall / F1 against true_set
    """

    FEAT_DIM = 1545

    def __init__(self, meta_path, title_embs_path, code_embs_path):
        import torch
        self._torch = torch

        with open(meta_path, "rb") as f:
            self.meta: List[dict] = pickle.load(f)

        te = np.load(title_embs_path, allow_pickle=True)
        self._title_emb: Dict[str, np.ndarray] = dict(zip(te["keys"], te["vecs"]))

        ce = np.load(code_embs_path, allow_pickle=True)
        self._code_emb: Dict[str, np.ndarray] = dict(zip(ce["keys"], ce["vecs"]))

        # Flat index: (value_set_idx, candidate_idx)
        self._index: List[Tuple[int, int]] = [
            (vs_i, c_i)
            for vs_i, entry in enumerate(self.meta)
            for c_i in range(len(entry["candidates"]))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        vs_i, c_i = self._index[idx]
        entry = self.meta[vs_i]
        cand  = entry["candidates"][c_i]

        title_v = self._title_emb[entry["title"]]
        code_v  = self._code_emb[cand["display"]]
        onehot  = code_system_onehot(cand["system"])
        score   = np.array([cand["score"]], dtype=np.float32)

        x = np.concatenate([title_v, code_v, onehot, score])
        y = np.int8(cand["label"])

        return (
            self._torch.from_numpy(x),
            self._torch.tensor(y, dtype=self._torch.float32),
        )

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Materialise full (X, y) arrays.  Fine for val/test; for large train
        splits prefer the DataLoader path to avoid peak-RAM issues.
        """
        n = len(self._index)
        X = np.empty((n, self.FEAT_DIM), dtype=np.float32)
        y = np.empty(n, dtype=np.int8)
        for i, (vs_i, c_i) in enumerate(tqdm(self._index, desc="to_numpy")):
            entry = self.meta[vs_i]
            cand  = entry["candidates"][c_i]
            X[i]  = np.concatenate([
                self._title_emb[entry["title"]],
                self._code_emb[cand["display"]],
                code_system_onehot(cand["system"]),
                np.array([cand["score"]], dtype=np.float32),
            ])
            y[i] = cand["label"]
        return X, y

    def iter_value_sets(self):
        """Yield one meta dict per value set (for LLM / retrieval-only eval)."""
        yield from self.meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vsac-dir",   default=str(REPO_ROOT / "vsac_data"))
    p.add_argument("--index-dir",  default=str(REPO_ROOT / "vsac_index"))
    p.add_argument("--strategy",   default="title")
    p.add_argument("--top-k",      type=int, default=10)
    p.add_argument("--out-dir",    default=str(REPO_ROOT / "dataset"))
    p.add_argument("--holdout-publishers", nargs="+",
                   default=["Clinical Architecture", "CSTE Steward"])
    p.add_argument("--val-frac",       type=float, default=0.15)
    p.add_argument("--test-frac",      type=float, default=0.15)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--sanity-n",       type=int,   default=200)
    p.add_argument("--skip-sanity",    action="store_true")
    p.add_argument("--min-true-codes", type=int,   default=3)
    p.add_argument("--max-value-sets", type=int,   default=None,
                   help="Cap for dry runs")
    return p.parse_args()


def main():
    args     = parse_args()
    vsac_dir = Path(args.vsac_dir)
    out_dir  = Path(args.out_dir)
    exp_dir  = vsac_dir / "expansions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load FAISS index ──────────────────────────────────────────────────────
    log.info("Loading index (strategy=%s)...", args.strategy)
    index, records, texts, model = load_built_index(Path(args.index_dir), args.strategy)
    log.info("Index loaded: %d value sets", len(records))

    # ── Sanity check ──────────────────────────────────────────────────────────
    if not args.skip_sanity:
        sanity = run_sanity_check(index, records, model,
                                  n_samples=args.sanity_n, seed=args.seed)
        with open(out_dir / "sanity_check_results.json", "w") as f:
            json.dump(sanity, f, indent=2)
        if not sanity["overall_pass"]:
            log.error("Sanity check failed — consider switching to BioLORD.")
    else:
        log.info("Sanity check skipped.")

    # ── Assign splits ─────────────────────────────────────────────────────────
    split_assignment = assign_splits(
        records,
        holdout_publishers=args.holdout_publishers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Pass 1: retrieval  (FAISS + disk reads, no encoding)
    # Gather the universe of unique texts to embed, build candidate lists.
    # ─────────────────────────────────────────────────────────────────────────
    target_records  = records[: args.max_value_sets] if args.max_value_sets else records
    skipped         = Counter()
    pending         = []          # (rec, true_codes, deduped_candidates)
    needed_titles   = set()
    needed_displays = set()

    log.info("Pass 1/3: retrieval for %d value sets...", len(target_records))
    for rec in tqdm(target_records, desc="Retrieving", unit="vs",
                    dynamic_ncols=True, leave=True):
        oid   = rec["oid"]
        title = rec.get("title", "").strip()
        split = split_assignment.get(oid)
        if not split or not title:
            skipped["no_split_or_title"] += 1
            continue

        true_codes = load_true_codes(oid, exp_dir)
        if not true_codes:
            skipped["no_expansion"] += 1
            continue
        if len(true_codes) < args.min_true_codes:
            skipped["too_few_codes"] += 1
            continue

        candidates, sim_scores = retrieve_candidates(
            oid, title, index, records, model, exp_dir, top_k=args.top_k)
        if not candidates:
            skipped["no_candidates"] += 1
            continue

        deduped = deduplicate_candidates(candidates, sim_scores)
        pending.append((rec, true_codes, deduped))
        needed_titles.add(title)
        for c in deduped:
            needed_displays.add(c["display"] or c["code"])

    log.info("Pass 1 done: %d queued | %d titles | %d displays",
             len(pending), len(needed_titles), len(needed_displays))

    # ─────────────────────────────────────────────────────────────────────────
    # Pass 2: batch-encode all unique texts
    # Peak RAM: ~1 GB for the full corpus (vs ~16 GB in the old approach)
    # ─────────────────────────────────────────────────────────────────────────
    def batch_encode(texts):
        return model.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=256,
                            show_progress_bar=False).astype(np.float32)

    log.info("Pass 2a/3: encoding %d unique titles...", len(needed_titles))
    title_list = list(needed_titles)
    title_vecs = batch_encode(title_list)
    title_emb_cache: Dict[str, np.ndarray] = dict(zip(title_list, title_vecs))

    log.info("Pass 2b/3: encoding %d unique code displays...", len(needed_displays))
    disp_list  = list(needed_displays)
    chunk      = 4096
    disp_parts = []
    for i in tqdm(range(0, len(disp_list), chunk), desc="Encoding displays",
                  unit="chunk", dynamic_ncols=True, leave=True):
        disp_parts.append(batch_encode(disp_list[i:i+chunk]))
    disp_vecs = np.vstack(disp_parts)
    code_emb_cache: Dict[str, np.ndarray] = dict(zip(disp_list, disp_vecs))
    del disp_vecs, disp_parts

    # ── Save embedding lookup tables ──────────────────────────────────────────
    log.info("Saving title_embs.npz (%d vectors)...", len(title_list))
    np.savez_compressed(
        out_dir / "title_embs.npz",
        vecs=np.stack([title_emb_cache[t] for t in title_list]),
        keys=np.array(title_list, dtype=object),
    )

    log.info("Saving code_embs.npz (%d vectors)...", len(disp_list))
    np.savez_compressed(
        out_dir / "code_embs.npz",
        vecs=np.stack([code_emb_cache[d] for d in disp_list]),
        keys=np.array(disp_list, dtype=object),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Pass 3: build meta dicts — no feature matrix materialised
    # ─────────────────────────────────────────────────────────────────────────
    meta_by_split: Dict[str, list] = defaultdict(list)
    manifest_rows: list            = []

    log.info("Pass 3/3: building meta for %d value sets...", len(pending))
    for rec, true_codes, deduped in tqdm(pending, desc="Building meta",
                                          unit="vs", dynamic_ncols=True, leave=True):
        oid   = rec["oid"]
        split = split_assignment[oid]
        title = rec.get("title", "").strip()

        true_set = {c["code"] for c in true_codes}
        recall   = retrieval_recall_at_k(true_codes, deduped)

        # Compact candidate list — all the info needed for classifier AND LLM eval
        compact_candidates = [
            {
                "code":    c["code"],
                "system":  c["system"],
                # 'display' is the lookup key into code_embs; also human-readable
                "display": c["display"] or c["code"],
                # retrieval similarity score (feature for classifier)
                "score":   float(c["score"]),
                # ground-truth label (used by all eval paths)
                "label":   int(c["code"] in true_set),
            }
            for c in deduped
        ]

        if not compact_candidates:
            skipped["empty_candidates"] += 1
            continue

        n_pos = sum(c["label"] for c in compact_candidates)
        n_neg = len(compact_candidates) - n_pos

        meta_by_split[split].append({
            "oid":   oid,
            "split": split,
            # Lookup key into title_embs — also passed to LLM as context
            "title": title,
            "n_pos": n_pos,
            "n_neg": n_neg,
            # ── Classifier eval ──────────────────────────────────────────────
            # VSDataset.__getitem__ looks up title + display in embedding dicts,
            # concatenates with onehot(system) and score → 1545-dim X.
            "candidates": compact_candidates,
            # ── LLM / retrieval-only eval ────────────────────────────────────
            # true_codes is the ground-truth set; compare any predicted set
            # against this without needing embeddings.
            "true_codes": [
                {"code": c["code"], "system": normalise_code_system(c["system"])}
                for c in true_codes
            ],
        })

        manifest_rows.append({
            "oid":                   oid,
            "title":                 title,
            "publisher":             rec.get("publisher", ""),
            "type":                  infer_type(rec.get("code_systems") or []),
            "split":                 split,
            "n_true_codes":          len(true_codes),
            "n_candidates":          len(deduped),
            "n_pos":                 n_pos,
            "n_neg":                 n_neg,
            "retrieval_recall_at_k": round(recall, 4) if not np.isnan(recall) else None,
            "top_k":                 args.top_k,
        })

    log.info("Skipped breakdown: %s", dict(skipped))

    # ── Write per-split meta pkls ─────────────────────────────────────────────
    for sname in ("train", "val", "test"):
        entries = meta_by_split.get(sname, [])
        if not entries:
            log.warning("No entries for split '%s', skipping.", sname)
            continue
        with open(out_dir / f"{sname}_meta.pkl", "wb") as f:
            pickle.dump(entries, f)
        n_total = sum(e["n_pos"] + e["n_neg"] for e in entries)
        n_pos   = sum(e["n_pos"] for e in entries)
        log.info("Saved %s_meta.pkl: %d value sets | %d examples | pos=%.1f%%",
                 sname, len(entries), n_total,
                 100 * n_pos / n_total if n_total else 0)

    # ── Manifest ──────────────────────────────────────────────────────────────
    with open(out_dir / "split_manifest.jsonl", "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    log.info("Manifest: %d rows", len(manifest_rows))

    # ── Dataset statistics ────────────────────────────────────────────────────
    recalls_arr = np.array([
        r["retrieval_recall_at_k"] for r in manifest_rows
        if r["retrieval_recall_at_k"] is not None
    ])

    def recall_stats(arr):
        if not len(arr):
            return None
        return {
            "mean":               round(float(arr.mean()), 4),
            "median":             round(float(np.median(arr)), 4),
            "p25":                round(float(np.percentile(arr, 25)), 4),
            "p75":                round(float(np.percentile(arr, 75)), 4),
            "pct_perfect_recall": round(float((arr == 1.0).mean()), 4),
        }

    split_stats = {}
    for sname in ("train", "val", "test"):
        rows    = [r for r in manifest_rows if r["split"] == sname]
        n_total = sum(r["n_pos"] + r["n_neg"] for r in rows)
        n_pos   = sum(r["n_pos"] for r in rows)
        s_recalls = np.array([
            r["retrieval_recall_at_k"] for r in rows
            if r["retrieval_recall_at_k"] is not None
        ])
        split_stats[sname] = {
            "n_value_sets":          len(rows),
            "n_examples":            n_total,
            "positive_fraction":     round(n_pos / n_total, 4) if n_total else 0,
            "retrieval_recall_at_k": recall_stats(s_recalls),
        }

    stats = {
        "top_k":              args.top_k,
        "strategy":           args.strategy,
        "holdout_publishers": args.holdout_publishers,
        "feature_dim":        1545,
        "code_system_vocab":  CODE_SYSTEM_VOCAB,
        "n_value_sets_total": len(manifest_rows),
        "retrieval_recall_at_k_overall": recall_stats(recalls_arr),
        "splits":             split_stats,
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats written. Done.")


if __name__ == "__main__":
    main()
