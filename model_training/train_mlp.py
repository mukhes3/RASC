#!/usr/bin/env python3
"""Train an MLP value-set code inclusion model."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mlp"

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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tune-threshold", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class VSDataset(Dataset):
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

    def __getitem__(self, idx: int):
        value_set_idx, candidate_idx = self._index[idx]
        entry = self.meta[value_set_idx]
        candidate = entry["candidates"][candidate_idx]
        x = np.concatenate(
            [
                self._title_emb[entry["title"]],
                self._code_emb[candidate["display"]],
                code_system_onehot(candidate["system"]),
                np.array([candidate["score"]], dtype=np.float32),
            ]
        )
        y = np.float32(candidate["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

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


class ValueSetMLP(nn.Module):
    def __init__(self, input_dim: int = 1545, hidden: tuple[int, ...] = (512, 256, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(dataset: VSDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")

    title_embs_path = dataset_dir / "title_embs.npz"
    code_embs_path = dataset_dir / "code_embs.npz"

    print("Loading datasets...")
    train_ds = VSDataset(dataset_dir / "train_meta.pkl", title_embs_path, code_embs_path)
    val_ds = VSDataset(dataset_dir / "val_meta.pkl", title_embs_path, code_embs_path)
    print(f"  train : {len(train_ds):>9,} examples")
    print(f"  val   : {len(val_ds):>9,} examples")

    print("Materializing val labels...")
    t0 = time.time()
    _, y_val = val_ds.to_numpy()
    print(f"  y_val : {y_val.shape} ({time.time() - t0:.1f}s)")

    scale_pos_weight = compute_scale_pos_weight(dataset_dir)
    print(f"scale_pos_weight : {scale_pos_weight:.1f}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    model = ValueSetMLP().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scale_pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    print("Training MLP...")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
        train_loss = train_loss_total / max(train_batches, 1)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                val_loss_total += float(criterion(logits, labels).item())
        val_loss = val_loss_total / max(len(val_loader), 1)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.max_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    artifacts = {
        "best_val_loss": round(best_val, 6),
        "arch": {"input_dim": 1545, "hidden": [512, 256, 64], "dropout": 0.3},
    }
    if args.tune_threshold:
        val_probs = collect_probs(model, val_loader, device)
        threshold, best_f1 = find_best_threshold(y_val, val_probs)
        val_pred = (val_probs >= threshold).astype(int)
        artifacts["threshold"] = threshold
        artifacts["val_f1"] = round(float(f1_score(y_val, val_pred, zero_division=0)), 4)
        artifacts["best_threshold_f1"] = round(best_f1, 4)
        print(f"Tuned threshold on val: {threshold:.4f} (F1={best_f1:.4f})")

    model_path = output_dir / "model.pt"
    artifacts_path = output_dir / "artifacts.pkl"
    torch.save(
        {
            "model_state": best_state,
            **artifacts,
        },
        model_path,
    )
    with open(artifacts_path, "wb") as handle:
        pickle.dump(artifacts, handle)

    print(f"Saved {model_path}")
    print(f"Saved {artifacts_path}")


if __name__ == "__main__":
    main()
