# RASC

Companion codebase for the paper **"Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring"**.

Paper link:

This repository contains:

- `create_dataset/`: VSAC download, retrieval-index construction, dataset reconstruction, and the released lightweight split manifest
- `model_training/`: standalone training scripts for the MLP, LightGBM, and cross-encoder models

This repository does **not** include raw VSAC content. To reconstruct the dataset artifacts used for training, you must download the value set content locally with a valid UMLS API key.

Model weights will be released separately on Hugging Face.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install from `pyproject.toml`:

```bash
pip install -e .
```

## Repository Layout

```text
RASC/
  create_dataset/
    download_vsac.py
    build_index.py
    build_dataset.py
    release_manifest.py
    split_manifest_release.jsonl
  model_training/
    train_mlp.py
    train_lightgbm.py
    train_cross_encoder.py
```

## Data Reconstruction

### 1. Download VSAC content locally

```bash
export UMLS_API_KEY="YOUR_UMLS_API_KEY"
python create_dataset/download_vsac.py
```

This writes the local corpus to `vsac_data/`.

### 2. Build the semantic retrieval index

The notebook configuration used `title` retrieval with SAPBERT.

```bash
python create_dataset/build_index.py --strategy title
```

This writes the FAISS index to `vsac_index/`.

### 3. Reconstruct the manifest-compatible dataset artifacts

```bash
python create_dataset/build_dataset.py \
  --vsac-dir vsac_data \
  --index-dir vsac_index \
  --strategy title \
  --top-k 10 \
  --out-dir dataset \
  --holdout-publishers "Clinical Architecture" "CSTE Steward"
```

This produces:

- `dataset/train_meta.pkl`
- `dataset/val_meta.pkl`
- `dataset/test_meta.pkl`
- `dataset/title_embs.npz`
- `dataset/code_embs.npz`
- `dataset/split_manifest.jsonl`
- `dataset/dataset_stats.json`

### 4. Recover split assignments from the released lightweight manifest

The lightweight release manifest is:

- `create_dataset/split_manifest_release.jsonl`

To match your local download against it:

```bash
python create_dataset/release_manifest.py recover
```

## Train Models

All default hyperparameters are set to mimic the training notebooks used in this project.

### MLP

```bash
python model_training/train_mlp.py
```

Optional threshold tuning on validation:

```bash
python model_training/train_mlp.py --tune-threshold
```

### LightGBM

```bash
python model_training/train_lightgbm.py
```

Optional threshold tuning on validation:

```bash
python model_training/train_lightgbm.py --tune-threshold
```

### Cross-Encoder

```bash
python model_training/train_cross_encoder.py
```

Optional threshold tuning on validation:

```bash
python model_training/train_cross_encoder.py --tune-threshold
```

## Notes

- The released manifest is lightweight and contains no VSAC content.
- Raw value set content must be downloaded locally by each user with their own UMLS credentials.
- The training scripts expect reconstructed dataset artifacts under `dataset/`.
