# Contributing

This repository is maintained as a reproducible research package.

## Before you submit a change

- Run `pytest`.
- If your change affects tables, figures, or notebook content, run `python scripts/build_all_results.py`.
- Keep changes scoped to the pipeline, tests, or release assets unless you are intentionally changing the manuscript narrative.

## Recommended workflow

1. Install the development dependencies with `python -m pip install -r requirements-dev.txt`.
2. Make your change.
3. Run the smoke tests.
4. Rebuild the results if the pipeline changed.
5. Update `RELEASE_NOTES.md` if the change affects the published package.

## What to include in a good pull request

- A clear description of the analysis or manuscript impact.
- Tests that cover the changed code path.
- Any regenerated outputs if the repository artifacts changed.
