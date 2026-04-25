# Data — pointers to the BDSP S3 release

This repository ships **code only**. The deidentified data live in the
Brain Data Science Platform (BDSP) credentialed-access bucket.

## S3 location

```
s3://bdsp-opendata-credentialed/sleep-dementia-detection/
```

| File | Rows × cols | Description |
|------|-------------|-------------|
| `study_groups_deid.csv` | 22,985 × 65 | Per-PSG cohort with Elissa Ye's chart-review labels (`Predicted_Stage`: Excluded / No Dementia / Symptomatic / MCI / Dementia), per-disease evidence flags (`Dementia_Enc/dT/ICD/Med/Prob`, `MCI_*`, `AlzD_*`, `VaD_*`, `FTD_*`, `DLB_*`, `PD_*`, `Symptomatic_*`), CDR/MMSE/MoCA scores. Keyed by `BDSPPatientID`, `HashID`, `FileNameNew`. |
| `dementia_diagnosis_dates.csv` | 23,828 × 3 | `BDSPPatientID, DiagnosisDateShifted, AgeAtDiagnosis` — first chart-review-derived diagnosis date per patient (rule-based, applied to the BDSP-deID release). `DiagnosisDateShifted` is shifted by the same per-patient `ShiftedDays` offset as `DOVshifted` and `DOBshifted`; `AgeAtDiagnosis` is the (true, shift-invariant) age at that diagnosis. The diagnosis date is **not** the PSG date — it is the chart-review event date and typically precedes or follows the PSG by a variable interval (median ~4 years from any PSG). |
| `psg_manifest.csv` | 10,782 × 9 | One row per PSG in the analytic cohort (8,042 unique participants; matches paper's 8,044). Columns: `BDSPPatientID, HashID, FileNameNew, DOVshifted, Sex, AgeAtPSG, PSGType, group (DEM/MCI/CN), s3_path`. **Use this to fetch the raw PSG signals from `s3://bdsp-opendata-credentialed/I0001-MGB/<FileNameNew>` and re-run feature extraction.** |
| `mastersheet_outcome_deid.xlsx` | 8,672 × 41 | `HashID ↔ BDSPPatientID` crosswalk plus survival outcomes for the SBOP cohort (concurrent paper). |
| `features_macro_deid.csv` | 21,223 × 24 | Sleep architecture: TST, %REM, sleep efficiency, WASO, etc. |
| `features_alpha_deid.csv` | 18,955 × 43 | α₁ / α₂ / α₃ sub-band powers across stages. |
| `features_brain_age_deid.csv` | 19,294 × 485 | Per-channel mean-gradient, kurtosis, and band-power features that feed the brain-age model. |
| `features_coherence_deid.csv` | 73,387 × 305 | EEG coherence per channel pair × stage × band (long form: ~3 rows per PSG). |
| `features_full_deid.csv` | 22,583 × 1,071 | Wide combined feature matrix used by the canonical paper notebooks (architecture + spindles + slow oscillations + bandpowers + α subbands + meds + AHI + RED + BAI + hypoxia burden). |
| `features_MGH_deid.csv` | 8,673 × 158 | The SBOP github-repo feature matrix (subset of the SLEEP 2023 cohort: prevalent-DEM cases excluded). Kept here for compatibility with `prediction-example.ipynb`. |

All tables use the following deidentified keys:

- `BDSPPatientID` — stable per-patient ID in the BDSP namespace.
- `HashID` — SHA-256 hash of the source PSG; one per recording.
- `FileNameNew` — `<HashID>_<YYYYMMDD>_<HHMMSSmmm>` — matches the deidentified PSG folder layout in
  `s3://bdsp-opendata-credentialed/I0001-MGB/...` so you can join feature
  rows directly to EEG `.h5`/`.edf` files.
- `DOVshifted`, `DOBshifted`, `ShiftedDays` — per-patient random date offset (±365 d) applied
  consistently to every date for that patient.

## How to download

### Credentialing

Access requires a credentialed BDSP account. See https://bdsp.io/credentialing/
for the application flow.

### With the AWS CLI

```bash
aws s3 sync s3://bdsp-opendata-credentialed/sleep-dementia-detection/ ./data/
```

### With rclone

```bash
rclone sync s3:bdsp-opendata-credentialed/sleep-dementia-detection/ ./data/
```

### Subset (small files only — ~40 MB)

```bash
aws s3 sync s3://bdsp-opendata-credentialed/sleep-dementia-detection/ ./data/ \
    --exclude "*" \
    --include "study_groups_deid.csv" \
    --include "dementia_diagnosis_dates.csv" \
    --include "psg_manifest.csv" \
    --include "mastersheet_outcome_deid.xlsx" \
    --include "features_macro_deid.csv" \
    --include "features_alpha_deid.csv" \
    --include "features_MGH_deid.csv"
```

The full set is ~943 MB (`features_coherence_deid.csv` and `features_full_deid.csv` are the heavy ones).

## How the data were derived

`scripts/deidentify_dementia_cohort.py` and `scripts/deidentify_feature_tables.py` in this
repository document the de-identification process. Both work by joining the
PHI source tables with `mapping.csv` from the BDSP collaboration folder, which
provides the canonical `(PatientID, MRN, DOV) → (BDSPPatientID, HashID, FileNameNew, DOVshifted)`
crosswalk for every PSG in the BDSP-deID release.

Match rates: 99.97 % on the cohort table, 98.3-98.8 % on the feature tables.
The handful of unmatched rows are 2018+ BIDS-format PSGs (recoverable via
`EEG_Master_MGH.csv`) or 5 PatientIDs that don't appear in any current BDSP
inventory (likely Epic-merge legacy IDs).

## Raw PSG recordings

The deidentified raw EEG/PSG signals (~22 k recordings) live in a separate
prefix in the same bucket:

```
s3://bdsp-opendata-credentialed/I0001-MGB/...
```

Use `FileNameNew` from any of the feature tables to locate the matching
recording.
