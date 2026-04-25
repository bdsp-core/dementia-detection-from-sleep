# Dementia Detection from Brain Activity During Sleep

Reproducibility package for:

> Ye EM, Sun H, Krishnamurthy PV, Adra N, Ganglberger W, Thomas RJ, Lam AD, Westover MB.
> **Dementia detection from brain activity during sleep.** *SLEEP* 46(3):zsac286, 2023.
> https://doi.org/10.1093/sleep/zsac286

## What's in this paper

Cross-sectional binary classification of dementia (DEM) and mild cognitive impairment (MCI) versus cognitively normal (CN), from features engineered out of clinical polysomnograms.

- **Cohort.** 10,784 PSGs from 8,044 participants at the Massachusetts General Hospital sleep laboratory (2009–2019). 339 DEM, 514 MCI, 7,263 CN.
- **Features.** Sleep architecture (TST, sleep efficiency, %REM, WASO, %NREM); spectral band powers (δ/θ/α/σ × frontal/central/occipital × W/NREM/REM, plus ratios and kurtosis); spindle features (LUNA-detected: amplitude, density, duration, peak frequency, slow-oscillation coupling); slow oscillations (amplitude, duration, slope, rate, peak-to-peak); EEG coherence; brain-age index (BAI), AHI, hypoxia burden.
- **Models.** Logistic regression, support vector machine, random forest. Three binary tasks: DEM vs CN, MCI vs CN, DEM/MCI vs CN.
- **Headline performance.** DEM vs CN: AUROC 0.78, AUPRC 0.22. MCI vs CN: AUROC 0.73, AUPRC 0.18. DEM/MCI vs CN: AUROC 0.76, AUPRC 0.32.

## Quick start

```bash
git clone https://github.com/bdsp-core/dementia-detection-from-sleep.git
cd dementia-detection-from-sleep
pip install -r requirements.txt

# Pull the deidentified data from S3 (~943 MB; credentialed access required)
aws s3 sync s3://bdsp-opendata-credentialed/sleep-dementia-detection/ ./data/

# Open any of the canonical paper notebooks, e.g.
jupyter notebook code/04_model/binary_dementia_classification_DM_v_CN_V2.ipynb
```

## Repository layout

```
sleep-dementia-detection/
├── code/
│   ├── 01_phenotyping/   Elissa's chart-review labeling rules + EHR extraction
│   │                     (regex files, step2b/c criteria scripts, study-group notebooks)
│   ├── 02_features/      Feature pipeline + canonical analysis notebooks
│   │                     (segment_EEG, multitaper, bandpower, main_BA, main_spindle_SO,
│   │                      EEG_Coherence_Analysis, spectral/spindle/alpha analysis V*)
│   ├── 03_spindle/       Noor Adra's LUNAspindles wrapper around Luna +
│   │                     spindle_extraction_luna_HaoqiVersion.py
│   ├── 04_model/         Canonical paper notebooks (binary DM_v_CN_V2, MCI_v_CN,
│   │                     DM_MCI_v_CN, MCI_v_DM; multiclass V14; CDR_classification V5)
│   │                     plus train_classifiers.py (clean from-scratch alternative)
│   └── 05_figures/       plot_confusion_matrix, sleepdata_summary_statistics, +
│                         SBOP figure scripts as templates
├── data/                 README.md only — actual data lives in S3 (see below)
├── docs/                 Methods/feature notes (work-in-progress)
├── models/               Empty — trained models live in S3 alongside the data
├── notebooks/            prediction-example.ipynb (template from the SBOP repo)
├── scripts/              De-identification + notebook-staging utilities
├── LICENSE               CC BY-NC 4.0
├── README.md             This file
├── STATUS.md             Per-component reproducibility status
├── ToDo.md               Remaining integration / publishing work
└── requirements.txt
```

## Data access

The deidentified data live in the BDSP credentialed-access bucket:

```
s3://bdsp-opendata-credentialed/sleep-dementia-detection/
```

Contents (see [data/README.md](data/README.md) for the full inventory):

- `study_groups_deid.csv` — 22,985 rows of per-PSG chart-review labels
  (BDSPPatientID, HashID, FileNameNew, DOVshifted, Age, Sex, dx flags, cognitive scores)
- `features_full_deid.csv` — 22,583 × 1,071 wide feature matrix
- `features_coherence_deid.csv` — coherence per channel pair × stage × band
- `features_brain_age_deid.csv` — spectral / kurtosis / gradient features
- `features_macro_deid.csv` — sleep architecture
- `features_alpha_deid.csv` — α₁/α₂/α₃ sub-band powers
- `mastersheet_outcome_deid.xlsx` — `HashID ↔ BDSPPatientID` crosswalk + survival outcomes (SBOP cohort)
- `MGH_dementia_all_04082025_*.csv` — 23,828 deidentified diagnosis dates

Raw deidentified PSG signals are at:

```
s3://bdsp-opendata-credentialed/I0001-MGB/...
```

Use `FileNameNew` from any of the feature tables to locate the matching PSG.

### Credentialed access

You need a BDSP credentialed account. Apply at https://bdsp.io/credentialing/. Once approved you'll receive AWS access keys; configure them with `aws configure` (or rclone / boto3) before `aws s3 sync`.

## How the deidentified data were derived

`scripts/deidentify_dementia_cohort.py` and `scripts/deidentify_feature_tables.py` document the full process. Both join the PHI source tables (Elissa's `study_criteria_table_label_V6.xlsx` plus the `eeg_data/` per-PSG feature tables) with `mapping.csv` from the BDSP collaboration folder, which provides the canonical
`(PatientID, MRN, DOV) → (BDSPPatientID, HashID, FileNameNew, DOVshifted)`
crosswalk for every PSG in the BDSP-deID release.

Match rates: 99.97 % on the cohort table, 98.3-98.8 % on the feature tables. Five PatientIDs and a handful of 2018+ BIDS-format PSGs are unrecoverable through the 2022 mapping; they're flagged in `STATUS.md`.

## Reproducibility status

This repository assembles the project's working tree from Box (`(internal)/(internal)/Projects/dementia_detection/`) and Dropbox (internal collaboration folders) into a single layout. **All paper notebooks are present**, with cell outputs stripped to remove dataframe-preview PHI. The remaining work (path adaptation, end-to-end test, figure regeneration on the deidentified data) is tracked in [STATUS.md](STATUS.md) and [ToDo.md](ToDo.md).

## How to cite

```bibtex
@article{ye2023dementia,
  title={Dementia detection from brain activity during sleep},
  author={Ye, Elissa M and Sun, Haoqi and Krishnamurthy, Parimala V and Adra, Noor and
          Ganglberger, Wolfgang and Thomas, Robert J and Lam, Alice D and Westover, M Brandon},
  journal={Sleep},
  volume={46},
  number={3},
  pages={zsac286},
  year={2023},
  publisher={Oxford University Press},
  doi={10.1093/sleep/zsac286}
}
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International). See [LICENSE](LICENSE). Both the code in this repo and the deidentified data on S3 are released under this license. Some files retained from upstream open-source projects (`code/02_features/segment_EEG.py`, `code/02_features/extract_features_parallel.py`, `code/03_spindle/LUNAspindles/`) keep their original licenses; the LICENSE file documents this.

## Acknowledgements

This work was carried out at the Massachusetts General Hospital Department of Neurology and the Clinical Data Animation Center (CDAC). Thanks to Niels Turley and the BDSP engineering team for the deidentification crosswalk; to Haoqi Sun and Wolfgang Ganglberger for the modeling and feature-engineering tooling; to Noor Adra for the LUNAspindles package; and to Robert Thomas, Alice Lam, and Parimala Krishnamurthy for clinical and methodological guidance.
