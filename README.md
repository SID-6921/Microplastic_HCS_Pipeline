# Microplastic HCS Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> **MS2 companion repository** — Transfer-learning classification of microplastic-induced cell death morphology in A549 lung epithelial cells.
> Siddhardha Nanda · Siri Tech Solutions, 2025-2026

---

## Repository Layout

```
Microplastic_HCS_Pipeline/
├── pipeline/               Core library (importable as `pipeline.*`)
│   ├── data_loader.py      Simulated BBBC-style dataset generator
│   ├── preprocess.py       Image resize, denoise, normalise
│   ├── detect.py           Nucleus detection & morphology measurement
│   ├── features.py         18-descriptor feature extraction
│   └── classify.py         5-model classification stack + validation
├── scripts/
│   └── build_all_results.py   <- SINGLE ENTRY POINT
├── results/
│   ├── features.csv           Feature matrix (192 images x 18 descriptors)
│   ├── tables/                9 CSV tables (T1-T9)
│   └── figures/               9 main + 7 supplementary figures (300 DPI)
├── notebooks/
│   └── MS2_Manuscript.ipynb   Full manuscript draft
├── data/README.md             Dataset description & BBBC pointers
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/build_all_results.py
jupyter notebook notebooks/MS2_Manuscript.ipynb
```

## Models

| # | Model | Input |
|---|-------|-------|
| 1 | Logistic Regression | 18-descriptor tabular features |
| 2 | Random Forest (300 trees) | 18-descriptor tabular features |
| 3 | CNN from scratch (TinyCNN) | 512x512 DAPI images |
| 4 | ResNet-18 from scratch | 512x512 DAPI images |
| 5 | ResNet-18 ImageNet pretrained + staged fine-tuning | 512x512 DAPI images |

## Statistical Validation (13 tests)

Accuracy/AUC/ECE + 95% CI, DeLong test, 5-fold CV, Feature ablation,
Kruskal-Wallis, Spearman rho, PCA before/after normalisation,
MP type/size class distribution, Dose-response Spearman (2 new).

## Dataset

Fully in silico simulated A549 dataset. SEED = 42. See data/README.md.

## Citation

> Nanda, S. (2026). Transfer-learning classification of microplastic-induced
> cell death morphology in A549 lung epithelial cells via high-content screening.
> https://github.com/SID-6921/Microplastic_HCS_Pipeline

## License

MIT
