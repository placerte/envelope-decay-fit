# Validation Workflows

This folder contains **slow / human review** workflows. These are not part of
the default test suite.

## Quick start

Run the review runner (all steps):

```bash
uv run python validation/review_runner.py
```

Skip the full dataset sweep:

```bash
uv run python validation/review_runner.py --skip-all-datasets
```

## Outputs

All outputs are written under `out/`.

* `out/run_all_examples/<timestamp>/` for the review runner
* `out/verification/<timestamp>/...` for dataset sweeps

## Scripts

* `review_runner.py` — orchestrates the validation steps
* `review_all_datasets.py` — full dataset sweep (slow)
* `bench_fitters.py` — manual fitter validation on a single dataset
* `bench_windows.py` — window scan diagnostics
