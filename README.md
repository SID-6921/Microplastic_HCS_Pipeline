# Microplastic HCS Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## Scope and limitations

All results in this repository are derived from a **synthetic simulation benchmark**.
No in vitro or in vivo experiments were conducted. The pipeline is designed as a
validated computational substrate for future deployment on experimental HCS datasets.
See the manuscript Methods §2.1 for full scope statement.
