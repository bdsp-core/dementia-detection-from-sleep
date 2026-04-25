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

- `regenerate_figures.py` — reproduces the paper's main tables and
  figures from saved fold-prediction pickles + the cohort PHI:
    - Table 2 (group characteristics)
    - Table S4 (per-classifier accuracy/precision/recall/F1/MCC)
    - Supplementary BH-corrected p-value summary
    - Figure 1B (age distribution by group)
    - Figure 2 (top discriminative features by OR, stage-grouped)
    - Figure 3 (occipital power spectra by stage, DEM/MCI/CN)
    - Figure 5 (ROC + PR curves, three binary tasks × LR/SVM/RF)
    - Confusion matrices per task
  Runtime ~10 seconds. Outputs land in `figures/`.

## `figures/` — sample outputs

Pre-rendered PNGs from a 2026-04-25 run, suitable for visual comparison
to the paper PDF. The CSVs are the underlying tables.

## Reproducibility expectations

- **AUROC**: from saved pickles, matches paper headline to ±0.005.
- **AUPRC**: from saved pickles, matches paper to ±0.01.
- **Cohort sizes**: 10,784 PSGs / 8,044 participants — exact match.
- **Demographics (Table 2)**: every reported number matches paper exactly.
