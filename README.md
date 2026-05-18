# Microplastic HCS Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> MS2 companion repository for an automated high-content screening workflow in A549 morphology data.
> Current release is framed as a pilot feasibility package.

## Important Scope Note

This repository contains a pilot-scale workflow and derived outputs.
Performance/statistical tables should be interpreted as exploratory, not confirmatory.
Current build expands to a balanced 1,000-sample simulation benchmark (250/class) for stability checks,
and uses permutation-based finite-sample inference in place of asymptotic small-n comparisons.
The notebook and tables are provided for transparent reproducibility of the current pipeline state.

## Repository Layout

```
Microplastic_HCS_Pipeline/
├── pipeline/                     Core library (importable as pipeline.*)
├── scripts/
│   ├── build_all_results.py      Single entry point for result generation
│   └── audit_nb.py               Checks notebook figure/table filename references
├── results/
│   ├── features.csv              Feature matrix export
│   ├── tables/                   Main result tables (table_*.csv)
│   └── figures/                  Main and supplementary PNG figures
├── notebooks/
│   └── MS2_Manuscript.ipynb      Reviewer-safe pilot manuscript notebook
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/build_all_results.py
jupyter notebook notebooks/MS2_Manuscript.ipynb
```

## What This Version Emphasizes

- End-to-end pipeline reproducibility
- Transparent table/figure generation
- Explicit limitations and conservative interpretation
- Separation of discrimination vs calibration interpretation

## Outputs Included

- `results/features.csv`
- `results/tables/table_1_model_performance.csv`
- `results/tables/table_2_transfer_learning.csv`
- `results/tables/table_3_calibration_ece.csv`
- `results/tables/table_4_feature_ablation.csv`
- `results/tables/table_5_biological_validation.csv`
- `results/tables/table_6_computational_cost.csv`
- `results/tables/table_7_class_distribution_by_mp.csv`
- `results/tables/table_8_dose_response.csv`
- `results/tables/table_9_delong_tests.csv`
- `results/tables/table_cv_summary.csv`
- `results/figures/fig_*.png`
- `results/figures/supp_*.png`

## Dataset Note

Current pipeline data is synthetic/simulated for workflow demonstration and stress-testing.
See `data/README.md` for dataset context and pointers.

## Citation

```
Nanda, S. (2026). Microplastic HCS Pipeline (MS2 Companion Repository).
https://github.com/SID-6921/Microplastic_HCS_Pipeline
```

## License

MIT
