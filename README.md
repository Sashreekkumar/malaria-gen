# Malaria Vector Taxon Classifier

A scalable, memory-efficient machine learning pipeline for species classification within the *Anopheles gambiae* complex using whole-genome SNP data from the [MalariaGEN Vector Observatory](https://www.malariagen.net/).

---

## Overview

This pipeline classifies mosquito species (*gambiae*, *coluzzii*, *arabiensis*, *bissau*, *gcx3*) from 48.5M SNP variants per sample. It is designed to run on consumer-grade hardware through streaming chunk-wise processing, sparse matrix representations, and disk-backed memory-mapped arrays.

**Preliminary results: 90% accuracy, MCC 0.85 on held-out test samples (83-sample training subset).**

---

## Pipeline Architecture
```
Raw Zarr Stores
      ↓
Metadata Preprocessing      malariagendata.ipynb
      ↓
Data Download               dataloader.py
      ↓
MAF Filtering               data_preprocessing.py → cache/maf_mask.npy
      ↓
012 Encoding + Save         data_preprocessing.py → cache/encoded_filtered/*.dat
      ↓
Sparse PCA (TruncatedSVD)   pca.py               → cache/svd_model.pkl, X_pca.npy
      ↓
CNN Classification          cnn.py               → cnn_model.pt
      ↓
Inference                   test_cnn.py
```

---

## Repository Structure
```
malaria-gen/
├── src/
│   ├── malariagendata.ipynb       # metadata preprocessing
│   ├── dataloader.py              # streaming download + extraction
│   ├── data_preprocessing.py      # MAF mask, 012 encoding, call rate filtering
│   ├── pca.py                     # sparse TruncatedSVD
│   ├── cnn.py                     # 1D CNN architecture + training
│   ├── xgboost_pipeline.py        # baseline XGBoost (exploratory)
│   ├── test_cnn.py                # CNN inference + evaluation
│   └── test_xgboost.py            # XGBoost inference + evaluation
├── cache/
│   ├── maf_mask.npy               # boolean variant mask (48.5M,)
│   ├── encoded_filtered/          # per-sample .dat memmaps
│   ├── svd_model.pkl              # fitted TruncatedSVD
│   ├── X_pca.npy                  # PCA matrix (n_samples, 50)
│   └── valid_ids.npy              # sample IDs that passed filtering
├── data/
│   └── data_csv/
│       ├── sampled_100.csv        # training subset
│       └── test.csv               # test samples
├── cnn_model.pt                   # trained CNN weights
├── cnn_label_encoder.pkl          # CNN label encoder
├── model.ubj                      # trained XGBoost model
└── label_encoder.pkl              # XGBoost label encoder
```

---

## Installation
```bash
git clone https://github.com/Sashreekkumar/malaria-gen.git
cd malaria-gen
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch
- XGBoost
- scikit-learn
- scipy
- zarr
- numpy
- pandas
- joblib

---

## Usage

### Step 1 — Metadata Preprocessing
Open and run `src/malariagendata.ipynb` to generate `data/data_csv/sampled_100.csv`.

### Step 2 — Download Data
```bash
python src/dataloader.py
```

### Step 3 — Train
```bash
python src/cnn.py
```
This runs the full pipeline: MAF mask → encoding → sparse PCA → CNN training. All intermediate artifacts are cached — rerunning skips already completed steps.

### Step 4 — Test
Place test zarr folders in `extracted_test/` and run:
```bash
python src/test_cnn.py
```

---

## Key Design Decisions

| Challenge | Solution |
|---|---|
| 48.5M variants per sample exceeds RAM | Chunk-wise zarr streaming (100k variants/chunk) |
| Full dataset too large for PCA | Sparse `csr_matrix` + `TruncatedSVD` (no densification) |
| Raw encoded arrays OOM during batching | `np.memmap` disk-backed arrays |
| 97% of variants uninformative | MAF filtering (≥5% threshold) |
| String labels for XGBoost/CNN | `LabelEncoder` → integer mapping |
| Pipeline interrupted mid-run | Resumable — checks for existing `.dat` and mask files |

---

## Results (Preliminary — 83 training samples, 10 test samples)

### CNN

| Metric | Value |
|---|---|
| Accuracy | 90.00% |
| Balanced Accuracy | 83.33% |
| Matthews CC | 0.8535 |
| Macro F1 | 0.85 |

### XGBoost (Baseline)

| Metric | Value |
|---|---|
| Accuracy | 80.00% |
| Macro F1 | 0.60 |

*Note: Three of the five target species (arabiensis, gcx3, and part of bissau) were absent from the training subset due to data availability constraints. Full dataset training expected to substantially improve minority class recall.*

---

## Architecture — SNP_CNN
```
Input: (batch, 50)            PCA components
  ↓ unsqueeze
(batch, 50, 1)
  ↓ Linear(1, 128) + ReLU     learned embedding per component
(batch, 50, 128)
  ↓ permute
(batch, 128, 50)
  ↓ Conv1d(128→64) + BN + ReLU
  ↓ Conv1d(64→128) + BN + ReLU
  ↓ Conv1d(128→256) + BN + ReLU
  ↓ AdaptiveAvgPool1d(1)
(batch, 256)
  ↓ Linear(256→128) + ReLU + Dropout(0.3)
  ↓ Linear(128→n_classes)
Output: (batch, n_classes)    raw logits
```

---
