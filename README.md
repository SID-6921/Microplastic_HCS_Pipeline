# Microplastic HCS Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MS2 companion repository** — Transfer-learning classification of microplastic-induced cell death morphology in A549 lung epithelial cells.  
> Siddhardha Nanda · Siri Tech Solutions, 2025–2026

---

## Repository Layout

```
Microplastic_HCS_Pipeline/
├── pipeline/                     Core library (importable as `pipeline.*`)
│   ├── __init__.py
│   ├── data_loader.py            Simulated BBBC-style dataset generator (SEED=42)
│   ├── preprocess.py             Image resize, denoise, normalise
│   ├── detect.py                 Nucleus detection & morphology measurement
│   ├── features.py               18-descriptor feature extraction
│   └── classify.py               5-model classification stack + validation
├── scripts/
│   ├── build_all_results.py      ← SINGLE ENTRY POINT (generates all tables + figures)
│   └── audit_nb.py               Cross-checks notebook filenames against disk
├── results/
│   ├── features.csv              Feature matrix (192 cells × 18 descriptors)
│   ├── tables/                   10 CSV tables (T1–T9 + CV summary)
│   └── figures/                  9 main + 7 supplementary figures (300 DPI PNG)
├── notebooks/
│   └── MS2_Manuscript.ipynb      Full manuscript with all outputs rendered inline
├── data/
│   └── README.md                 Dataset description & BBBC021 pointers
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Re-generate all tables and figures (~15 s, uses cached features.csv)
python scripts/build_all_results.py

# 3. Open the manuscript notebook (outputs already embedded — no re-run needed)
jupyter notebook notebooks/MS2_Manuscript.ipynb
```

---

## Models

| # | Model | Input | Hold-out Acc | AUC |
|---|-------|-------|-------------|-----|
| 1 | Logistic Regression | 18-descriptor tabular | 0.896 | 0.982 |
| 2 | Random Forest (300 trees) | 18-descriptor tabular | 0.968 | 0.993 |
| 3 | TinyCNN (scratch) | 512×512 DAPI | — | — |
| 4 | ResNet-18 (scratch) | 512×512 DAPI | — | — |
| 5 | ResNet-18 (ImageNet pretrained) | 512×512 DAPI | — | — |

> DL model results are simulated; feature-based models use fully deterministic sklearn pipelines.

---

## Results

| Artifact | Description |
|----------|-------------|
| `results/tables/table_1_model_performance.csv` | Accuracy, AUC, F1 per model |
| `results/tables/table_2_transfer_learning.csv` | Transfer learning comparison |
| `results/tables/table_3_calibration_ece.csv` | ECE calibration scores |
| `results/tables/table_4_feature_ablation.csv` | Ablation study (18→1 features) |
| `results/tables/table_5_biological_validation.csv` | Kruskal-Wallis + Spearman rho |
| `results/tables/table_6_computational_cost.csv` | Training time per model |
| `results/tables/table_7_class_distribution_by_mp.csv` | MP type × class distribution |
| `results/tables/table_8_dose_response.csv` | Dose–response Spearman |
| `results/tables/table_9_delong_tests.csv` | Pairwise DeLong AUC tests |
| `results/tables/table_cv_summary.csv` | 5-fold CV accuracy ± std |

---

## Statistical Validation

13 independent tests across 3 tiers:
- **Model performance**: Accuracy, AUC, F1 with 95% CI (bootstrap n=1000)
- **Calibration**: Expected Calibration Error (ECE), Platt scaling
- **Biology**: Kruskal–Wallis (feature differences), Spearman ρ (dose–response, n=2), DeLong pairwise AUC, PCA before/after normalisation, MP type/size class distribution, 5-fold CV

---

## Dataset

Fully in silico simulated A549 lung epithelial cell dataset (SEED=42).  
192 images × 4 classes: Viable (0), Early Apoptosis (1), Late Apoptosis (2), Necrosis (3).  
48 images per class. See [`data/README.md`](data/README.md).

---

## Citation

```
Nanda, S. (2026). Transfer-learning classification of microplastic-induced
cell death morphology in A549 lung epithelial cells via high-content screening.
https://github.com/SID-6921/Microplastic_HCS_Pipeline
```

## License

MIT
