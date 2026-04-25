# Reproducibility status

Snapshot of what's currently in this repo, where each piece came from, what's verified, and what's still missing for end-to-end reproduction of the SLEEP 2023 paper.

## Legend

- ✅ Present and complete
- 🟡 Present but needs adaptation, verification, or documentation
- ⚠ Missing — pointer to where it likely lives
- 🔒 PHI — must not enter the repo without de-identification

---

## ⭐ Discovery: the canonical project folder

`(internal project folder)/` turns out to be the **complete project working tree** referenced by `step2b_*ElissaCriteria*.py`. It contains:

- `code/` — ~70 notebooks: `binary_dementia_classification_{DM_v_CN, MCI_v_CN, DM_MCI_v_CN, MCI_v_DM}.ipynb`, `multiclass_dementia_classification_V14.ipynb`, `EEG_Coherence_Analysis.ipynb`, `CDR_classification_V5.ipynb`, `nested_cross_validation_test.ipynb`, `Feature selection Lasso NEW.ipynb`, `cuzick_test.ipynb`, `compile_brain_age_features.ipynb`, `combine_luna_output.ipynb`, `spectral_analysis_V7.ipynb`, `spindle_analysis_V2.ipynb`, `compute_alpha3_alpha2_V2.ipynb`, `assign_study_groups_V3.ipynb`, `wrangle_neuropsychiatric_scores.ipynb`, plus version histories and earlier drafts.
- `medical_data/` (40 GB) — every input the labeling pipeline reads: `study_criteria_table*.{csv,xlsx}`, `EDW_*.csv`, `RPDR_*.csv`, `{MMSE,MoCA,CDR,ACER}_*Final.xlsx`, `Dementia_*ICD.csv`, `MCI_*ICD.csv`, `Neuropsychiatric Scores/`, `Dementia Validation/`, `exclusion_regex`, `dementia_regex`, `MCI_regex.txt`, `medications_regex.txt`, `patient_phenotypes_table*.csv`, `studies_list.{csv,xlsx}`. **All PHI; mirrored locally to `_box_inspect/dementia_detection_other/medical_data/` (gitignored), pending de-identification before publication.**
- `eeg_data/` — feature tables: `coherence_df.csv` (392 MB), `BA_features_df.csv`, `study_features_table_v4.csv`, `combined_luna_output.csv`, `alpha_features_table_V4.csv`, `macrofeatures_df.csv`, plus PSG-system-specific sleep stats (`grass_sleep_stats.csv`, `natus_sleep_stats.csv`, `mat_sleep_stats.csv`). PHI; mirrored to `_box_inspect/dementia_detection_other/eeg_data/`.
- `spindle-luna/` — Haoqi's spindle detector wrapper.
- `figures/` — paper figures.
- Manuscript versions, meeting minutes, task managers, reviewer responses.

This folder resolves all three Tier-1 blockers below. Notebooks are now staged in this repo with cell outputs stripped (outputs frequently contained PHI dataframe previews).

---

## `code/01_phenotyping/` — DEM/MCI/CN labeling

| File | Status | Source |
|---|---|---|
| `step2b/c_*ElissaCriteria*.py` | 🟡 | dropbox `code-haoqi/` — hard-coded paths to `medical_data/`. Adapt path or set up symlink to `_box_inspect/dementia_detection_other/medical_data/`. |
| `Dementia_criteria_Elissa_step{1,2}.ipynb` | 🟡 | dropbox `sleep_stressor/data/sleep_general/examples/` |
| `assign_study_groups_V3.ipynb` | ✅ | `box:.../dementia_detection/code/` — produces the DEM/MCI/CN study-group assignment from the labeled cohort. |
| `wrangle_neuropsychiatric_scores.ipynb`, `mine_neuropsychiatric_scores_V2.ipynb` | ✅ | same |
| `ICD_Code_Analysis.ipynb` | ✅ | same |
| `get_*.ipynb` (5 files) | ✅ | `box:.../brain_age_Dementia/code/` — Elissa's 2018 RPDR extraction notebooks |
| `exclusion_regex`, `dementia_regex`, `MCI_regex.txt`, `medications_regex.txt` | ✅ | `box:.../dementia_detection/medical_data/` |

**Inputs (PHI; staged locally only):** all in `_box_inspect/dementia_detection_other/medical_data/`. Essentials (~5.5 GB) pulled: `study_criteria_table_label_V{1..6}.xlsx`, `study_criteria_table.csv`, `EDW_{EncounterDiagnosis,ProblemList,MedicationDiagnosis}.csv`, `RPDR_{Enc,Med,Dia,Mrn,Dem}_All.csv`, `{MMSE,MoCA,CDR}_{EDW,RPDR}_Final.xlsx`, `EDW_ACER_df_*.xlsx`, dx-by-ICD CSVs, `study_medical_table*.csv`, `studies_list*`, `patient_phenotypes_table*.csv`. Skipped: `EDW_ClinicalNotes.csv` (19.5 GB), `RPDR_{Lno,Prc,Rad,Phy}_All.csv` (~15 GB total) — only needed if rebuilding cognitive-score mining from raw notes.

---

## `code/02_features/` — PSG → feature matrix

| File | Status | Source |
|---|---|---|
| `segment_EEG.py`, `multitaper_spectrogram.py`, `bandpower.py`, `extract_features_parallel.py`, `load_mgh_sleep_dataset.py` | ✅ | `dropbox:.../brainAge/brain_age_Elissa_AllMGH/mycode/` |
| `main_BA.py`, `main_spindle_SO.py` | ✅ | same |
| `compile_brain_age_features.ipynb`, `combine_luna_output.ipynb` | ✅ | `box:.../dementia_detection/code/` — drivers that assemble the final per-PSG feature table |
| `spectral_analysis_V7.ipynb`, `spindle_analysis_V2.ipynb`, `alpha_analysis_V3.ipynb`, `compute_alpha3_alpha2_V2.ipynb` | ✅ | same — per-feature exploratory + paper-ready analysis |
| **`EEG_Coherence_Analysis.ipynb`** | ✅ | same — **the coherence feature analysis** (resolves Tier-1 #3). Uses `coherence_df.csv` with columns like `Wake_O1-O2_Theta` (stage × channel-pair × band). |
| `assess_missing_features.ipynb` | ✅ | same — feature completeness check |
| `Feature_display_name.csv` | ✅ | feature dictionary for plot labels |

**Feature tables (PHI; staged locally only):** `_box_inspect/dementia_detection_other/eeg_data/coherence_df.csv` (392 MB), `study_features_table_v4.csv` (352 MB), `BA_features_df.csv` (180 MB), `combined_luna_output.csv` (131 MB), plus alpha/macro/sleep-stats CSVs.

---

## `code/03_spindle/` — Luna spindle detection

| File | Status | Source |
|---|---|---|
| `LUNAspindles/` | ✅ | `box:.../NoorAdra/Noor's Spindle Project/LUNAspindles/` — Python wrapper around Luna |
| `spindle_extraction_luna_HaoqiVersion.py` | ✅ | `box:.../dementia_detection/spindle-luna/` — Haoqi's spindle extraction script for the SLEEP 2023 paper |
| `summary.py`, `destrat.py`, `fit_run.py`, `fit_boot.py`, `id_pairs.py`, `luna_07*.py`, `spindleCommand.txt` | ✅ | `box:.../Noor's Spindle Project (1)/` |

---

## `code/04_model/` — cross-sectional LR / SVM / RF

| File | Status | Source |
|---|---|---|
| **`binary_dementia_classification_DM_v_CN_V2.ipynb`** | ✅ | `box:.../dementia_detection/code/` — **canonical paper notebook for DEM vs CN** (resolves Tier-1 #1). Uses LR + SVM + RF + Lasso feature selection + Stratified KFold + AUROC/AUPRC. |
| **`binary_dementia_classification_MCI_v_CN.ipynb`** | ✅ | same — MCI vs CN |
| **`binary_dementia_classification_DM_MCI_v_CN.ipynb`** | ✅ | same — DEM/MCI vs CN |
| `binary_dementia_classification_MCI_v_DM.ipynb` | ✅ | same — supplementary MCI vs DEM |
| `multiclass_dementia_classification_V14.ipynb` | ✅ | same — 3-way classifier (latest version) |
| `CDR_classification_V5.ipynb` | ✅ | same — CDR-based classifier |
| `nested_cross_validation_test.ipynb` | ✅ | same — nested CV scaffolding |
| `Feature selection Lasso NEW.ipynb`, `univarite_feature_selection.ipynb` | ✅ | same |
| `benjamini_hochberg.ipynb`, `cuzick_test.ipynb` | ✅ | same — multiple-comparisons correction + Cuzick trend test |
| `neuropsychiatric_score_prediction_V2.ipynb` | ✅ | same — regression for cognitive scores |
| `train_classifiers.py` | 🟡 | reconstruction (clean sklearn LR/SVM/RF + nested CV) — useful as an automated alternative to the canonical notebook flow |

**Resolved:** Tier-1 #1. The canonical notebooks load `study_criteria_table_label_V6.xlsx` (PHI, staged locally) and the per-PSG feature tables in `eeg_data/`.

---

## `code/05_figures/` — figure generation

| File | Status | Source |
|---|---|---|
| `plot_confusion_matrix.ipynb` | ✅ | `box:.../dementia_detection/code/` — paper Fig. 4 |
| `sleepdata_summary_statistics_V3.ipynb` | ✅ | same — paper Table 1 |
| `plot_*.py`, `generate_*.py` (SBOP figure scripts) | 🟡 | dropbox `figures_paper/` — for the SBOP survival paper, kept as templates for figure re-use |

The paper's published Figs. 1A/1B/4A/4B/4I are also available in raw `.svg` form in `box:.../dementia_detection/code/Fig*.svg`.

---

## `data/`

| File | Status | Source |
|---|---|---|
| `data/features_MGH_deid.csv` | ✅ | `dropbox:.../SBOP github-repo/` — 8,673 × 158 deidentified per-PSG feature matrix from the SBOP cohort. **Subset of the SLEEP 2023 cohort** (excludes prevalent DEM/MCI cases). |
| `data/mastersheet_outcome_deid.xlsx` | ✅ | same — `HashID ↔ BDSPPatientID` crosswalk + per-PSG survival outcomes. |
| `s3://.../sleep-dementia-detection/dementia_diagnosis_dates.csv` | ✅ | Renamed and column-cleaned from `MGH_dementia_all_04082025_Elissa_rule_made_by_Haoqi.csv` (orig source: `box:.../BDSP_deID/I0001-MGB/data_Outcomes/mild_cognitive_impairment/old/`). 23,828 rows; columns `BDSPPatientID, DiagnosisDateShifted, AgeAtDiagnosis`. **`DiagnosisDateShifted` is the date of the clinical encounter where dementia was documented** (not the chart-review date — review happened in 2025, but the dates here range 1999-2024). Verified shifted by the same per-patient `ShiftedDays` offset as `DOVshifted`/`DOBshifted` (validated by checking `Age == (DiagDate − DOB)/365.25 − ShiftedDays/365.25` to 1e-15 precision). |
| `s3://.../sleep-dementia-detection/psg_manifest.csv` | ✅ | 10,618 rows × 10 cols. One row per analytic-cohort PSG. Columns: `BDSPPatientID, HashID, FileNameNew, DOVshifted, Sex, AgeAtPSG, PSGType, group, session, s3_path`. **`s3_path` points to the resolved BIDS .edf** at `s3://bdsp-opendata-repository/PSG/bids/S0001/sub-S0001<BDSPPatientID>/ses-<N>/eeg/sub-S0001<BDSPPatientID>_ses-<N>_task-psg_eeg.edf`. 10,377/10,618 (97.7%) PSGs successfully resolved against the live BIDS-formatted S3 release. 15/15 randomly sampled paths verified to exist. |
| `s3://.../sleep-dementia-detection/comorbidities_deid.csv` | ✅ | 8,042 rows × 11 cols. BDSPPatientID-keyed boolean flags for 9 comorbidity categories (cardiovascular disease, OSA, mood, obesity, insomnia, diabetes, anxiety, psychotic, alcoholism) derived from RPDR diagnosis-name regex + ICD prefix matching. Per-group prevalences and z-test significance directions reproduce the paper's Table 2 (* / *** asterisks for p<0.05 / p<0.001). |
| **`data/` (in this repo)** | ✅ | Single `data/README.md` only — points users to the S3 location below. Per BDSP convention, deidentified data does not live in git. |
| **`s3://bdsp-opendata-credentialed/sleep-dementia-detection/study_groups_deid.csv`** | ✅ | derived locally by `scripts/deidentify_dementia_cohort.py` from `study_criteria_table_label_V6.xlsx` joined with `mapping.csv` from the BDSP collab folder. **22,985 rows × 65 columns; 99.97% match (6 unmatched, retained PHI-side under `_box_inspect/`).** Keyed by `BDSPPatientID`, `HashID`, and `FileNameNew` (matches the BDSP-deID PSG folder layout). Includes `Predicted_Stage` (Excluded 9,944 / No Dementia 9,661 / Symptomatic 2,259 / MCI 672 / Dementia 449), per-disease evidence flags, cognitive scores, demographics. |
| **`s3://bdsp-opendata-credentialed/sleep-dementia-detection/features_*.csv`** | ✅ | Five deidentified feature tables produced by `scripts/deidentify_feature_tables.py`: `features_full_deid.csv` (22,583 × 1,071), `features_coherence_deid.csv` (73,387 × 305), `features_brain_age_deid.csv` (19,294 × 485), `features_macro_deid.csv` (21,223 × 24), `features_alpha_deid.csv` (18,955 × 43). Match rates 98.3-98.8%. |
| **`s3://.../sleep-dementia-detection/mastersheet_outcome_deid.xlsx`** | ✅ | The SBOP crosswalk (HashID ↔ BDSPPatientID + per-PSG survival outcomes). |

## Unmatched audit

After joining `study_criteria_table_label_V6.xlsx` with `mapping.csv`:

- **6 (PSG, MRN, DOV) tuples** unmatched out of 22,991 (0.03%). All are 2018+ BIDS-format files (e.g., `Shaita~ Hamid_04328776-...`) whose mapping lives in `EEG_Master_MGH.csv` rather than the 2022 `mapping.csv`.
- **5 PatientIDs** unrecoverable from any current BDSP inventory: `Z10518942, Z10654155, Z12734743, Z16984218, Z7952699`. Likely Epic-merge legacy IDs. `Z12734743` matches the entry in `PatientID_not_found_in_BDSP_table.txt`.

These 11 rows are negligible vs. the cohort size (~0.05%).

---

## `models/`

⚠ Empty in the repo. Trained pickles exist in `box:.../dementia_detection/code/`:
`DM_v_CN_LG_model.pickle`, `DM_v_CN_LG_scores.pickle`, `DM_v_CN_RF_scores.pickle`, `DM_v_CN_SVM_scores.pickle`, plus `DM_v_MCI_v_CN_*` and `DM_MCI_v_CN_*`. These contain learned weights from PHI training data; whether to ship them depends on whether the input data was deidentified at training time. Verify before adding.

---

## Open questions (significantly fewer)

1. Are `features_MGH_deid.csv` (SBOP github-repo) and the canonical `study_features_table_v4.csv` (`dementia_detection/eeg_data/`) the same feature set, just with different keys (HashID vs PatientID)? Probably yes; verify column-by-column.
2. Why does `features_MGH_deid.csv` lack coherence columns when the canonical pipeline produces a `coherence_df.csv`? Is the deidentified release a deliberate subset, or was coherence joined separately?
3. Did the trained pickles in `dementia_detection/code/` get fit on PHI-keyed or deidentified features? Determines whether they can be redistributed.
4. Is there an authoritative latest version of the labeled cohort (`study_criteria_table_label_V7+.xlsx`) beyond V6?
