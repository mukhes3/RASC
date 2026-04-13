#!/usr/bin/env python3
"""Export and recover the lightweight split manifest released with this repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_CREATE_DATASET_DIR = REPO_ROOT / "create_dataset"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "vsac_data"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def oid_variants(oid: str) -> set[str]:
    variants = {oid}
    head, sep, tail = oid.rpartition("-")
    if sep and tail.isdigit() and len(tail) == 8:
        variants.add(head)
    return variants


def collect_local_oids(download_dir: Path) -> set[str]:
    local_oids = set()
    for folder_name in ("metadata", "expansions"):
        folder = download_dir / folder_name
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            local_oids.update(oid_variants(path.stem))

    index_path = download_dir / "index.jsonl"
    if index_path.exists():
        for row in iter_jsonl(index_path):
            local_oids.update(oid_variants(row["oid"]))
    return local_oids


def export_manifest(input_manifest: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(output_path, "w") as handle:
        for row in iter_jsonl(input_manifest):
            lightweight = {
                "oid": row["oid"],
                "split": row["split"],
                "retrieval_recall_at_k": row.get("retrieval_recall_at_k"),
                "type": row.get("type", ""),
                "publisher": row.get("publisher", ""),
                "top_k": row.get("top_k"),
            }
            handle.write(json.dumps(lightweight) + "\n")
            n_rows += 1
    print(f"Wrote {n_rows} rows to {output_path}")


def recover_splits(download_dir: Path, manifest_path: Path, output_path: Path) -> None:
    local_oids = collect_local_oids(download_dir)
    if not local_oids:
        raise FileNotFoundError(f"No VSAC download found under {download_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    missing = 0
    with open(output_path, "w") as handle:
        for row in iter_jsonl(manifest_path):
            if oid_variants(row["oid"]) & local_oids:
                handle.write(json.dumps(row) + "\n")
                matched += 1
            else:
                missing += 1

    print(f"Recovered {matched} rows to {output_path}")
    print(f"Manifest rows not present locally: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export the lightweight release manifest.")
    export_parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_DATASET_DIR / "split_manifest.jsonl",
    )
    export_parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_CREATE_DATASET_DIR / "split_manifest_release.jsonl",
    )

    recover_parser = subparsers.add_parser("recover", help="Recover split assignments for a local VSAC download.")
    recover_parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    recover_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_CREATE_DATASET_DIR / "split_manifest_release.jsonl",
    )
    recover_parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_CREATE_DATASET_DIR / "recovered_splits.jsonl",
    )

    args = parser.parse_args()
    if args.command == "export":
        export_manifest(args.input_manifest.resolve(), args.output_path.resolve())
    else:
        recover_splits(
            args.download_dir.resolve(),
            args.manifest_path.resolve(),
            args.output_path.resolve(),
        )


if __name__ == "__main__":
    main()
