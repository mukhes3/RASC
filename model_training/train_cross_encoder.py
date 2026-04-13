#!/usr/bin/env python3
"""Train a SAPBERT cross-encoder value-set code inclusion model."""

from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "cross_encoder"


class CrossEncoderDataset(Dataset):
    def __init__(self, meta_path: Path):
        with open(meta_path, "rb") as handle:
            meta = pickle.load(handle)
        self.examples: list[tuple[str, str, int]] = []
        for entry in meta:
            title = entry["title"]
            for candidate in entry["candidates"]:
                display = candidate["display"] or candidate["code"]
                label = int(candidate["label"])
                self.examples.append((title, display, label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        return self.examples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-name",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def get_probs(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    fp16: bool,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    for encoded, labels in tqdm(loader, desc="Validation", leave=False):
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(**encoded).logits.squeeze(-1)
            loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs), total_loss / max(n_batches, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = torch.cuda.is_available() and not args.disable_fp16

    print(f"Device      : {device}")
    print(f"FP16        : {fp16}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")

    train_ds = CrossEncoderDataset(dataset_dir / "train_meta.pkl")
    val_ds = CrossEncoderDataset(dataset_dir / "val_meta.pkl")
    print(f"Train examples : {len(train_ds):,}")
    print(f"Val examples   : {len(val_ds):,}")

    train_labels = [example[2] for example in train_ds.examples]
    pos_frac = sum(train_labels) / len(train_labels)
    scale_pos_weight = (1 - pos_frac) / pos_frac
    print(f"scale_pos_weight : {scale_pos_weight:.1f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def collate_fn(batch: list[tuple[str, str, int]]):
        titles = [example[0] for example in batch]
        displays = [example[1] for example in batch]
        labels = torch.tensor([example[2] for example in batch], dtype=torch.float32)
        encoded = tokenizer(
            titles,
            displays,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return encoded, labels

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 16,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ).to(device)
    model.gradient_checkpointing_enable()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scale_pos_weight], device=device)
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(exclusion in name for exclusion in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(exclusion in name for exclusion in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.lr)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.max_epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    print("Training cross-encoder...")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}", leave=False)
        for step, (encoded, labels) in enumerate(progress, start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(**encoded).logits.squeeze(-1)
                loss = criterion(logits, labels) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * args.grad_accum_steps
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        y_val, val_prob, val_loss = get_probs(val_loader, model, device, fp16, criterion)
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
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    if args.tune_threshold:
        y_val, val_prob, _ = get_probs(val_loader, model, device, fp16, criterion)
        threshold, best_f1 = find_best_threshold(y_val, val_prob)
        val_pred = (val_prob >= threshold).astype(int)
        artifacts["threshold"] = threshold
        artifacts["val_f1"] = round(float(f1_score(y_val, val_pred, zero_division=0)), 4)
        artifacts["best_threshold_f1"] = round(best_f1, 4)
        print(f"Tuned threshold on val: {threshold:.4f} (F1={best_f1:.4f})")

    model_dir = output_dir / "model"
    artifacts_path = output_dir / "artifacts.pkl"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(artifacts_path, "wb") as handle:
        pickle.dump(artifacts, handle)

    print(f"Saved {model_dir}")
    print(f"Saved {artifacts_path}")


if __name__ == "__main__":
    main()
