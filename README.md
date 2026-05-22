# Microplastic HCS Pipeline

[![CI](https://github.com/SID-6921/Microplastic_HCS_Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/SID-6921/Microplastic_HCS_Pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Code and reproducible pipeline for the manuscript:

> **A reproducible high-content screening pipeline for multi-class microplastic-associated
> cell death classification: simulation-based feasibility benchmarking of feature
> engineering and deep learning approaches.**
> *Submitted to Environment International.*

This repository contains all code, generated tables (9), and figures (17) needed to
fully reproduce the simulation benchmark described in the paper.

## Repository structure

| Directory / File | Contents |
|---|---|
| `pipeline/` | Image loading, preprocessing, nucleus segmentation, feature extraction, classification |
| `scripts/build_all_results.py` | End-to-end build: runs all 7 pipeline steps, writes 9 tables + 17 figures to `results/` |
| `notebooks/MS2_Manuscript.ipynb` | Methods narrative and declarations (simulation-only, no code cells) |
| `results/tables/` | 9 CSV tables (model performance, calibration, ablation, dose-response, etc.) |
| `results/figures/` | 9 main figures + 8 supplementary figures (PNG, 300 DPI) |
| `docs/` | Reviewer response matrix, figure captions, appendix templates |
| `tests/` | Smoke tests validating core pipeline behaviour |
| `requirements.txt` | Runtime dependencies (pinned) |
| `requirements-dev.txt` | Dev/test dependencies |

## Quick start

```powershell
python -m pip install -r requirements-dev.txt
pytest
```

If you want to regenerate the figures and tables:

```powershell
python scripts/build_all_results.py
```

If you need a faster validation run, set `TARGET_PER_CLASS` before launching the build:

```powershell
$env:TARGET_PER_CLASS = "20"
python scripts/build_all_results.py
```

## Notebook

Open `notebooks/MS2_Manuscript.ipynb` after the build completes to review the generated outputs. The notebook is written to load from the local `results/` directory.

## Reproducibility

- `requirements.txt` captures the runtime dependencies.
- `requirements-dev.txt` adds notebook execution and test tooling.
- `RELEASE_NOTES.md` summarizes the publishable release state.
- `LICENSE` is MIT for straightforward GitHub publication.

## Citing this work

If you use this code or pipeline, please cite the manuscript (citation details will be
updated upon acceptance). A `CITATION.cff` file is included in this repository.
Zenodo DOI metadata is prepared via `.zenodo.json` and should be activated on the
first tagged GitHub release.

## Scope and limitations

All results in this repository are derived from a **synthetic simulation benchmark**.
No in vitro or in vivo experiments were conducted. The pipeline is designed as a
validated computational substrate for future deployment on experimental HCS datasets.
See the manuscript Methods §2.1 for full scope statement.

## Key results at a glance

| Model | Macro AUC | Accuracy | ECE (post-cal) |
|---|---|---|---|
| Logistic Regression | **0.981** | 0.905 | 0.183 |
| Random Forest | 0.955 | 0.845 | 0.365 |
| ResNet-18 (pretrained) | 0.954 | 0.940 | **0.057** |
| ResNet-18 (scratch) | 0.910 | 0.860 | 0.172 |
| CNN (scratch) | 0.873 | 0.810 | 0.175 |

Feature ablation: AUC drops from **0.972 → 0.603** over 8 sequential descriptor removals.
Permutation test: pretrained ResNet-18 non-inferior to Random Forest (p = 0.948).
Dose-encoding significant in **8/9** strata after BH correction.
