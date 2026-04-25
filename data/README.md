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
| `dementia_diagnosis_dates.csv` | 23,828 × 3 | `BDSPPatientID, DiagnosisDateShifted, AgeAtDiagnosis`. `DiagnosisDateShifted` is **the date of the clinical encounter where the diagnosis was documented in the medical chart**, shifted by the same per-patient `ShiftedDays` offset as `DOVshifted` and `DOBshifted` (it is *not* the date the chart review was performed — chart review happened in 2025, but the dates here range 1999-2024). Verified shifted by checking `AgeAtDiagnosis = (DiagnosisDateShifted − DOBshifted) / 365.25 − ShiftedDays / 365.25` to floating-point precision across 3,517 cross-joined rows. `AgeAtDiagnosis` is the true, shift-invariant patient age at that visit. The diagnosis date is generally *not* a PSG date (median delta ~4 years from any PSG); use `psg_manifest.csv` for PSG-aligned timing. |
| `psg_manifest.csv` | 10,618 × 10 | One row per PSG in the analytic cohort (97.7% resolve to a BIDS-format file on S3). Columns: `BDSPPatientID, HashID, FileNameNew, DOVshifted, Sex, AgeAtPSG, PSGType, group (DEM/MCI/CN), session (ses-N), s3_path`. **`s3_path` points directly to the raw `.edf`** at `s3://bdsp-opendata-repository/PSG/bids/S0001/sub-S0001<BDSPPatientID>/ses-<N>/eeg/sub-S0001<BDSPPatientID>_ses-<N>_task-psg_eeg.edf`. Sleep stage annotations live in the same folder (`*_task-psg_annotations.csv` and `*_caisr_annotations.csv`). |
| `psg_manifest_unresolved.csv` | 241 × 9 | The 241 PSGs (2.3% of the cohort) that did **not** resolve to a live BIDS .edf, with a `failure_mode` column: 127 cases where the BDSPPatientID has no `sub-S0001<id>/` folder on S3 at all (subject excluded during BIDS conversion); 114 cases where the subject is on S3 but no session matches the manifest's `DOVshifted` date (some of the patient's sessions were converted to BIDS, but not the one used in this study). |
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
- `FileNameNew` — `<HashID>_<YYYYMMDD>_<HHMMSSmmm>` — the original-style hashed filename. The corresponding BIDS-organized recording on S3 is at `s3://bdsp-opendata-repository/PSG/bids/S0001/sub-S0001<BDSPPatientID>/ses-<N>/...` (use `psg_manifest.csv` for the resolved per-PSG path).
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
