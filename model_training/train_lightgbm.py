#!/usr/bin/env python3
"""Train a LightGBM value-set code inclusion model."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lightgbm"

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
CODE_SYSTEM_TO_IDX = {code_system: idx for idx, code_system in enumerate(CODE_SYSTEM_VOCAB)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.6)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--lambda-l1", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--tune-threshold", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalise_code_system(raw: str) -> str:
    uri_map = {
        "http://snomed.info/sct": "SNOMED-CT",
        "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10-CM",
        "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
        "http://loinc.org": "LOINC",
        "http://www.ama-assn.org/go/cpt": "CPT",
        "http://hl7.org/fhir/sid/icd-10-pcs": "ICD-10-PCS",
        "http://www.cms.gov/Medicare/Coding/ICD10": "ICD-10-PCS",
        "http://www.nlm.nih.gov/research/umls/hcpcs": "HCPCS",
    }
    if raw in uri_map:
        return uri_map[raw]
    for vocab_name in CODE_SYSTEM_VOCAB[:-1]:
        if vocab_name.lower() in raw.lower():
            return vocab_name
    return "OTHER"


def code_system_onehot(code_system_raw: str) -> np.ndarray:
    idx = CODE_SYSTEM_TO_IDX.get(
        normalise_code_system(code_system_raw),
        CODE_SYSTEM_TO_IDX["OTHER"],
    )
    vec = np.zeros(len(CODE_SYSTEM_VOCAB), dtype=np.float32)
    vec[idx] = 1.0
    return vec


class VSDataset:
    FEAT_DIM = 1545

    def __init__(self, meta_path: Path, title_embs_path: Path, code_embs_path: Path):
        with open(meta_path, "rb") as handle:
            self.meta: List[dict] = pickle.load(handle)

        title_embs = np.load(title_embs_path, allow_pickle=True)
        self._title_emb: Dict[str, np.ndarray] = dict(zip(title_embs["keys"], title_embs["vecs"]))

        code_embs = np.load(code_embs_path, allow_pickle=True)
        self._code_emb: Dict[str, np.ndarray] = dict(zip(code_embs["keys"], code_embs["vecs"]))

        self._index: List[Tuple[int, int]] = [
            (value_set_idx, candidate_idx)
            for value_set_idx, entry in enumerate(self.meta)
            for candidate_idx in range(len(entry["candidates"]))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(self._index)
        x = np.empty((n_rows, self.FEAT_DIM), dtype=np.float32)
        y = np.empty(n_rows, dtype=np.int8)
        for idx, (value_set_idx, candidate_idx) in enumerate(self._index):
            entry = self.meta[value_set_idx]
            candidate = entry["candidates"][candidate_idx]
            x[idx] = np.concatenate(
                [
                    self._title_emb[entry["title"]],
                    self._code_emb[candidate["display"]],
                    code_system_onehot(candidate["system"]),
                    np.array([candidate["score"]], dtype=np.float32),
                ]
            )
            y[idx] = candidate["label"]
        return x, y


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_arr = np.where(
        (precision_arr + recall_arr) > 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        0.0,
    )
    best_idx = int(f1_arr.argmax())
    threshold_idx = min(best_idx, len(thresholds) - 1)
    return float(thresholds[threshold_idx]), float(f1_arr[best_idx])


def compute_scale_pos_weight(dataset_dir: Path) -> float:
    manifest_rows = [
        json.loads(line)
        for line in (dataset_dir / "split_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    pos_frac_train = sum(row["n_pos"] for row in train_rows) / sum(
        row["n_pos"] + row["n_neg"] for row in train_rows
    )
    return (1 - pos_frac_train) / pos_frac_train


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    title_embs_path = dataset_dir / "title_embs.npz"
    code_embs_path = dataset_dir / "code_embs.npz"

    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")

    print("Loading datasets...")
    train_ds = VSDataset(dataset_dir / "train_meta.pkl", title_embs_path, code_embs_path)
    val_ds = VSDataset(dataset_dir / "val_meta.pkl", title_embs_path, code_embs_path)
    print(f"  train : {len(train_ds):>9,} examples")
    print(f"  val   : {len(val_ds):>9,} examples")

    print("Materializing train...")
    t0 = time.time()
    x_train, y_train = train_ds.to_numpy()
    print(f"  X_train : {x_train.shape} ({time.time() - t0:.1f}s)")

    print("Materializing val...")
    t0 = time.time()
    x_val, y_val = val_ds.to_numpy()
    print(f"  X_val   : {x_val.shape} ({time.time() - t0:.1f}s)")

    scale_pos_weight = compute_scale_pos_weight(dataset_dir)
    print(f"scale_pos_weight : {scale_pos_weight:.1f}")

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": -1,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "n_jobs": -1,
        "seed": args.random_seed,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, free_raw_data=True)

    print("Training LightGBM...")
    t0 = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=args.log_every),
        ],
    )
    print(f"Training finished in {time.time() - t0:.1f}s")

    artifacts = {
        "best_iteration": model.best_iteration,
        "params": params,
    }
    if args.tune_threshold:
        val_prob = model.predict(x_val)
        threshold, best_f1 = find_best_threshold(y_val, val_prob)
        val_pred = (val_prob >= threshold).astype(int)
        artifacts["threshold"] = threshold
        artifacts["val_f1"] = round(float(f1_score(y_val, val_pred, zero_division=0)), 4)
        artifacts["best_threshold_f1"] = round(best_f1, 4)
        print(f"Tuned threshold on val: {threshold:.4f} (F1={best_f1:.4f})")

    model_path = output_dir / "model.txt"
    artifacts_path = output_dir / "artifacts.pkl"
    model.save_model(str(model_path))
    joblib.dump(artifacts, artifacts_path)

    print(f"Saved {model_path}")
    print(f"Saved {artifacts_path}")


if __name__ == "__main__":
    main()
