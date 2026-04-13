#!/usr/bin/env python3
"""Build the semantic retrieval index used by dataset construction."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VSAC_DIR = REPO_ROOT / "vsac_data"
DEFAULT_INDEX_DIR = REPO_ROOT / "vsac_index"
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_BATCH_SIZE = 64
MAX_CODES_SAMPLED = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsac-dir", type=Path, default=DEFAULT_VSAC_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--strategy", default="title", choices=["title", "title_desc", "title_desc_codes"])
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def load_index_records(vsac_dir: Path) -> list[dict]:
    records = []
    with open(vsac_dir / "index.jsonl") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_exp_path(oid: str, exp_dir: Path) -> Path | None:
    exact = exp_dir / f"{oid}.json"
    if exact.exists():
        return exact
    matches = sorted(exp_dir.glob(f"{oid}-*.json"))
    return matches[0] if matches else None


def load_code_displays(oid: str, exp_dir: Path, max_codes: int = MAX_CODES_SAMPLED) -> list[str]:
    path = find_exp_path(oid, exp_dir)
    if path is None:
        return []
    with open(path) as handle:
        resource = json.load(handle)
    codes = resource.get("expansion", {}).get("contains", [])
    return [concept["display"] for concept in codes[:max_codes] if concept.get("display")]


def build_text(record: dict, strategy: str, exp_dir: Path) -> str:
    title = record.get("title", "").strip()
    description = (record.get("description", "") or "").strip()

    if strategy == "title":
        return title
    if strategy == "title_desc":
        return f"{title}. {description}".strip(" .") if description else title
    if strategy == "title_desc_codes":
        parts = [title]
        if description:
            parts.append(description)
        code_displays = load_code_displays(record["oid"], exp_dir=exp_dir)
        if code_displays:
            parts.append("Codes include: " + "; ".join(code_displays))
        return " | ".join(parts)
    raise ValueError(f"Unknown strategy: {strategy}")


def load_model_for_inference(model_name: str) -> SentenceTransformer:
    sapbert_ids = {
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    }
    if model_name in sapbert_ids:
        word_embedding = models.Transformer(model_name, max_seq_length=512)
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False,
        )
        return SentenceTransformer(modules=[word_embedding, pooling])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def main() -> None:
    args = parse_args()
    args.index_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = args.vsac_dir / "expansions"
    records = load_index_records(args.vsac_dir)
    texts = []
    valid_records = []

    print(f"Building retrieval index with strategy='{args.strategy}'")
    for record in tqdm(records, desc="Preparing texts"):
        text = build_text(record, args.strategy, exp_dir)
        if text.strip():
            texts.append(text)
            valid_records.append(record)

    model = load_model_for_inference(args.model_name)
    embeddings = embed_texts(texts, model, args.batch_size)
    index = build_faiss_index(embeddings)

    index_path = args.index_dir / f"index_{args.strategy}.faiss"
    metadata_path = args.index_dir / f"metadata_{args.strategy}.pkl"
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as handle:
        pickle.dump(
            {"records": valid_records, "texts": texts, "model": args.model_name},
            handle,
        )

    print(f"Saved {index_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
