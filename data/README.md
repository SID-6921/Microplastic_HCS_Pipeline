# Data Directory

## Dataset Description

This repository uses a **fully in silico simulated dataset** (Option C from the execution plan),
clearly labelled as such throughout the manuscript. No new wet-lab experiments were conducted.

### Simulation parameters

| Parameter | Value |
|-----------|-------|
| Cell line | A549 (lung epithelial, simulated) |
| Channels | DAPI (nuclear) + PI (membrane permeability) |
| Images per class | 48 (harder simulation) |
| Total images | 192 |
| Image size | 512 × 512 px |
| Random seed | 42 |

### Class labels

| ID | Label | Morphological definition |
|----|-------|--------------------------|
| 0 | Viable | Control — normal nuclear morphology, PI-negative |
| 1 | Early Apoptosis | PI-negative, chromatin condensation, cell shrinkage |
| 2 | Late Apoptosis | PI-positive, nuclear fragmentation, blebbing |
| 3 | Necrosis | PI-high, cell swelling, loss of membrane integrity |

### Microplastic covariates simulated

- **Types**: Polystyrene (PS), Polyethylene (PE), PET
- **Sizes**: nano (100 nm), micro (1–10 μm), large (>10 μm)
- **Concentrations**: 0, 10, 50, 100, 200 μg/mL
- **Exposure times**: 24 h, 48 h, 72 h

### Reproducibility

All images are generated deterministically from `pipeline/data_loader.py` using `random_seed=42`.
Running `python scripts/build_all_results.py` from the repo root regenerates the complete dataset,
features, and all results from scratch.

### Public dataset note

Suitable public BBBC datasets for future replacement include:
- **BBBC017** — A549 cytoplasm-nucleus-ER multi-channel images
- **BBBC025** — DAPI/Alexa 488 cell painting

See https://bbbc.broadinstitute.org/ for download instructions.
