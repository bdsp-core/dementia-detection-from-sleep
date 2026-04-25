# 04_model — cross-sectional binary classifiers

Reconstructed `train_classifiers.py` for the SLEEP 2023 paper's three binary
classification tasks (DEM vs CN, MCI vs CN, DEM/MCI vs CN). Three model
families: L2 logistic regression, RBF-kernel SVM, random forest. Stratified
nested cross-validation (outer for unbiased AUROC/AUPRC, inner for
hyperparameter selection) with bootstrap 95% CIs on pooled outer predictions.

## Status

**This is a reconstruction**, not the original training script — that script
hasn't surfaced in any project folder we've searched (see `STATUS.md` and
`ToDo.md` at the repo root). Use the headline metrics from the paper
(AUROC 0.78 / 0.73 / 0.76) as a cross-validation that this reconstruction is
behaviourally equivalent.

## Labeling caveat

The deidentified data we ship corresponds to the **SBOP survival paper
cohort**, which excluded prevalent DEM/MCI cases at PSG entry. As a result:

- `data/features_MGH_deid.csv` (8,673 rows) is missing many of the SLEEP-2023
  paper's prevalent-DEM cases.
- `data/cohort/MGH_dementia_all_04082025_Elissa_rule_made_by_Haoqi.csv`
  contains diagnosis dates for 23,828 patients — joining via
  `HashID → BDSPPatientID` and comparing `DiagnosisDate` to `DOVshifted`
  gives a partial recovery of prevalent-DEM labels.

The smoke-test labeling proxy in `derive_labels()` produces
~45 DEM / 0 explicit MCI / 7,985 CN out of 8,672 rows; the paper has
339 / 514 / 7,263 over 8,044 participants. Replacing the proxy with the
output of `code/01_phenotyping/step2b/c_*ElissaCriteria*.py` should bring
counts in line with the paper.

## Run

```bash
pip install -r requirements.txt   # at repo root
python code/04_model/train_classifiers.py \
    --task all \
    --output-dir models/results
```

Per-task outputs land in `models/results/`:
- `oof_<task>_<model>.csv` — out-of-fold predictions for downstream plotting
- `metrics_<task>.json` — AUROC/AUPRC + bootstrap CIs + per-fold best params
- `metrics_all.json` — combined summary
