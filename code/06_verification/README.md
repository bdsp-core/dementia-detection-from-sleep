# 06_verification — paper-headline reproduction check

Two scripts that re-run the canonical paper notebooks end-to-end and
print AUROC/AUPRC alongside the paper headlines. See
[`docs/reproduction_check.md`](../../docs/reproduction_check.md) for the
results from a 2026-04-25 run.

## Setup

These scripts read the **PHI source tables** (the raw cohort + feature
tables that the paper notebooks load from `../medical_data/` and
`../eeg_data/` relative to the working tree). Adjust the `ROOT` /
`COHORT` / `FEATURES` paths near the top of each script to point at your
local copy.

If you only have the deidentified data from S3
(`s3://bdsp-opendata-credentialed/sleep-dementia-detection/`), use
`features_full_deid.csv` and `study_groups_deid.csv` and replace the
`FolderName` join key with `HashID` in `build_task()` /
`build_full_dataset()`.

## Files

- `verify_paper.py` — runs all 9 task × model combinations
  (DEM/MCI/DEM-MCI vs CN, with LR/SVM/RF) using the notebook's
  balanced-evaluation recipe (CN undersampled to 1,000 inside both
  training and test splits). Reports balanced AUROC; AUPRC is inflated
  by ~3× because of the artificial 50:50 prevalence in the test fold.
  Total runtime ~6 minutes on an Apple M-series CPU.

- `verify_auprc_imbalanced.py` — single-task LR run for DEM vs CN with
  proper train-balanced / test-imbalanced evaluation. Reproduces the
  paper's headline AUROC = 0.78 and AUPRC = 0.22. Runtime ~30 seconds.

## Reproducibility expectations

- **AUROC**: matches paper headline within ±0.02 (seed and grid
  reduction account for the residual).
- **AUPRC**: matches within ±0.01 once the test fold preserves the
  natural ≈4 % dementia prevalence.
- **Cohort sizes**: 10,784 PSGs / 8,044 participants — exact match.
