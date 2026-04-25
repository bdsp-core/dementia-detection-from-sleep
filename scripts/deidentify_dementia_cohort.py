"""De-identify the SLEEP 2023 dementia-detection cohort table.

Joins `study_criteria_table_label_V6.xlsx` (PHI cohort with chart-review
labels) against `mapping.csv` (PatientID/MRN/DOV → BDSPPatientID/HashID/
DOVshifted), drops every direct identifier, and writes a row-per-PSG
deidentified table.

Modeled on `_box_inspect/bdsp_collab_essentials/create_deid_outcome_mastersheet.py`.

Run from repo root:
    python3 scripts/deidentify_dementia_cohort.py

Inputs (PHI; staged locally, gitignored):
    _box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx
    _box_inspect/bdsp_collab_essentials/mapping.csv

Output (deidentified, safe to commit):
    sleep-dementia-detection/data/cohort/study_groups_deid.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Resolves to the parent of `sleep-dementia-detection/`, which contains the
# gitignored `_box_inspect/` staging tree alongside this repo.
ROOT = Path(__file__).resolve().parent.parent.parent
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
MAPPING = ROOT / "_box_inspect/bdsp_collab_essentials/mapping.csv"
OUT = ROOT / "sleep-dementia-detection/data/cohort/study_groups_deid.csv"
UNMATCHED = ROOT / "_box_inspect/dementia_detection_other/study_groups_unmatched.csv"

# ID / PHI columns to drop. Anything not listed here is presumed to be either
# clinical metadata (kept) or a label/feature (kept).
DROP_PHI = [
    "FolderName", "PatientID", "EMPI",
    "MRN", "MRN_key", "MRN_x", "MRN_y",  # raw MRN + pandas-auto-suffixed dupes
    "LastName", "FirstName",
    "DateOfBirth", "DateOfVisit",
    "Path",
    # Free-text notes can leak names/dates extracted from charts:
    "Note", "CDR_Note", "Low_CDR_Note", "MMSE_Note", "High_MMSE_Note",
    "MoCA_Note", "High_MoCA_Note", "Neuropsych_Note",
]


def main():
    print(f"loading cohort: {COHORT}")
    cohort = pd.read_excel(COHORT)
    cohort["DateOfVisit"] = pd.to_datetime(cohort["DateOfVisit"])
    print(f"  cohort shape: {cohort.shape}")

    print(f"loading mapping: {MAPPING}")
    mapping = pd.read_csv(MAPPING, dtype={"PatientID": str, "MRN": str})
    mapping["DOV"] = pd.to_datetime(mapping["DOV"])
    # The mapping should be unique on (PatientID, MRN, DOV); enforce.
    if mapping[["PatientID", "MRN", "DOV"]].duplicated().any():
        n_dup = mapping[["PatientID", "MRN", "DOV"]].duplicated().sum()
        print(f"  WARN: {n_dup} duplicate (PatientID, MRN, DOV) rows in mapping; keeping first")
        mapping = mapping.drop_duplicates(subset=["PatientID", "MRN", "DOV"])

    cohort["PatientID"] = cohort["PatientID"].astype(str)
    # The cohort's MRN is hyphenated (e.g., "424-66-43"); the mapping uses the
    # un-hyphenated MRN_key (e.g., "4246643"). Join on MRN_key.
    cohort["MRN_key"] = cohort["MRN_key"].astype(str).str.replace(r"\.0$", "", regex=True)
    mapping["MRN"] = mapping["MRN"].astype(str).str.replace(r"\.0$", "", regex=True)
    print(f"  mapping shape: {mapping.shape}")

    # Inner-join brings only matched rows; track unmatched separately.
    join_cols = ["PatientID", "MRN", "DOV"]
    deid_cols = ["HashID", "BDSPPatientID", "DOVshifted", "ShiftedDays",
                 "DOBshifted", "FileNameNew"]
    matched = cohort.merge(
        mapping[join_cols + deid_cols],
        left_on=["PatientID", "MRN_key", "DateOfVisit"],
        right_on=join_cols,
        how="inner",
    )
    print(f"  matched: {len(matched)} / {len(cohort)} ({100*len(matched)/len(cohort):.2f}%)")

    # Track unmatched (also deid-stripped, for diagnostic purposes only — local file)
    unmatched = cohort[~cohort.index.isin(
        cohort.merge(mapping[join_cols], left_on=["PatientID", "MRN_key", "DateOfVisit"],
                     right_on=join_cols, how="inner").index
    )]
    print(f"  unmatched: {len(unmatched)} rows (will be written to {UNMATCHED.name})")

    # Drop PHI columns
    drop_present = [c for c in DROP_PHI if c in matched.columns]
    deid = matched.drop(columns=drop_present + ["DOV"])  # DOV is the joined-in raw date

    # Reorder so deid keys come first
    front = ["BDSPPatientID", "HashID", "FileNameNew", "DOVshifted",
             "ShiftedDays", "DOBshifted", "Sex", "Age", "TypeOfTest"]
    front = [c for c in front if c in deid.columns]
    other = [c for c in deid.columns if c not in front]
    deid = deid[front + other]

    # Final integrity check: no remaining PHI patterns in column names
    phi_in_cols = [c for c in deid.columns
                   if any(k in c for k in ["PatientID", "MRN", "EMPI",
                                           "FolderName", "FirstName", "LastName",
                                           "DateOfBirth", "DateOfVisit"])
                   and c not in ("BDSPPatientID",)]
    assert not phi_in_cols, f"PHI columns leaked: {phi_in_cols}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    deid.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({deid.shape[0]} rows × {deid.shape[1]} cols)")

    UNMATCHED.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(UNMATCHED, index=False)
    print(f"wrote {UNMATCHED}  ({len(unmatched)} rows; PHI; gitignored)")


if __name__ == "__main__":
    main()
