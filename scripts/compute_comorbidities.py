"""Compute per-patient comorbidity flags from RPDR_Dia_All.csv and join to
the analytic cohort, producing comorbidities_deid.csv keyed by BDSPPatientID.

Comorbidity categories follow paper Table 2: cardiovascular disease, OSA,
mood disorder, obesity, insomnia, diabetes, anxiety disorder, psychotic
disorder, alcoholism. Matching uses Diagnosis_Name regex AND ICD code
prefix matching.

Run from repo root:
    python3 _repro/compute_comorbidities.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
RPDR = ROOT / "_box_inspect/dementia_detection_other/medical_data/RPDR_Dia_All.csv"
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
MAPPING = ROOT / "_box_inspect/bdsp_collab_essentials/mapping.csv"
OUT_DIR = ROOT / "_s3_stage/sleep-dementia-detection"
OUT_DEID = OUT_DIR / "comorbidities_deid.csv"
FIGS = ROOT / "_repro/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# (category, name-regex, ICD-code-prefix-list) — ICD9 + ICD10 mixed
# Code matching is left-anchored against the Code column.
RULES = [
    ("Cardiovascular_disease",
     r"(?i)cardiovasc|coronary|myocardial|angina pectoris|heart failure|"
     r"arrhythm|atrial fibrillat|atrial flutter|hypertensi|cerebrovasc|"
     r"ischemic heart|peripheral arterial|aortic aneurysm",
     ["I10", "I11", "I12", "I13", "I20", "I21", "I22", "I25", "I48", "I50",
      "I60", "I61", "I63", "I65", "I66", "I67", "I70", "I71",
      "401", "402", "403", "404", "405", "410", "411", "412", "413",
      "414", "415", "416", "427", "428", "430", "431", "433", "434",
      "435", "436", "440", "441", "442"]),

    ("Obstructive_sleep_apnea",
     r"(?i)obstructive sleep apnea|sleep apnea|apnea, obstructive|"
     r"sleep-related breathing disorder|hypopnea",
     ["G4733", "G4730", "32723"]),

    ("Mood_disorder",
     r"(?i)major depress|depressive|dysthym|bipolar|cyclothym|mood disorder",
     ["F30", "F31", "F32", "F33", "F34", "F39",
      "296", "300.4", "311"]),

    ("Obesity",
     r"(?i)\bobesit|morbidly obese|body mass index 3[0-9]|body mass index 4|"
     r"body mass index 5|overweight",
     ["E66", "278.0", "278.00", "278.01", "278.02", "278.03"]),

    ("Insomnia",
     r"(?i)insomni",
     ["G470", "G4700", "G4701", "G4709", "F510", "78052", "780.52"]),

    ("Diabetes",
     r"(?i)diabetes mellitus|diabetic|type 2 diabet|type 1 diabet|"
     r"diabetes type|hyperglycaemia|hyperglycemia",
     ["E08", "E09", "E10", "E11", "E13", "250"]),

    ("Anxiety_disorder",
     r"(?i)anxiety disorder|generalized anxiety|panic disorder|"
     r"phobic|obsessive-compulsive|post-traumatic stress|acute stress",
     ["F40", "F41", "F42", "F43", "F44", "F45",
      "300.0", "300.00", "300.01", "300.02", "300.21", "300.22", "300.23",
      "300.3", "308", "309.81"]),

    ("Psychotic_disorder",
     r"(?i)schizophren|schizoaffective|delusional|psychotic|"
     r"psychosis|paranoia",
     ["F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29",
      "295", "297", "298"]),

    ("Alcoholism",
     r"(?i)alcohol use disorder|alcohol abuse|alcohol dependen|"
     r"alcohol-induced|alcoholism|alcohol withdrawal",
     ["F10", "303", "305.0"]),
]


def code_match(code, prefixes) -> bool:
    if not isinstance(code, str):
        return False
    code = code.strip()
    return any(code.startswith(p) for p in prefixes)


def main():
    t0 = time.time()
    print(f"loading {RPDR.name} (~14.7M rows)…")
    rpdr = pd.read_csv(RPDR, dtype={"EMPI": str, "Code": str},
                       usecols=["EMPI", "Diagnosis_Name", "Code"],
                       low_memory=False)
    print(f"  loaded {len(rpdr):,} rows in {time.time()-t0:.1f}s")

    # Load cohort and mapping for the EMPI → BDSPPatientID chain
    print("loading cohort + mapping…")
    cohort = pd.read_excel(COHORT, dtype={"PatientID": str, "EMPI": str})
    mapping = pd.read_csv(MAPPING, dtype={"PatientID": str},
                          usecols=["PatientID", "BDSPPatientID"]).drop_duplicates("PatientID")
    cohort["EMPI"] = cohort["EMPI"].astype(str).str.replace(r"\.0$", "", regex=True)
    cohort = cohort[["EMPI", "PatientID", "Predicted_Stage"]].drop_duplicates()
    cohort = cohort[cohort["Predicted_Stage"].isin(["Dementia", "MCI", "No Dementia"])]
    cohort = cohort.merge(mapping, on="PatientID", how="left")
    cohort = cohort.dropna(subset=["BDSPPatientID", "EMPI"])
    cohort["BDSPPatientID"] = cohort["BDSPPatientID"].astype(str)
    cohort = cohort.drop_duplicates(["EMPI", "BDSPPatientID"])
    print(f"  cohort EMPI→BDSPPatientID rows: {len(cohort):,}")

    # Build per-patient flags
    print("\ncomputing per-EMPI flags...")
    cohort_empi = set(cohort["EMPI"])
    rpdr = rpdr[rpdr["EMPI"].isin(cohort_empi)]
    print(f"  RPDR rows for our cohort EMPIs: {len(rpdr):,}")

    flags = pd.DataFrame({"EMPI": list(cohort_empi)})
    for cat, name_pat, code_prefixes in RULES:
        t1 = time.time()
        m_name = rpdr["Diagnosis_Name"].str.contains(name_pat, na=False, regex=True)
        m_code = rpdr["Code"].apply(lambda c: code_match(c, code_prefixes))
        m = m_name | m_code
        empis_with = set(rpdr.loc[m, "EMPI"])
        flags[cat] = flags["EMPI"].isin(empis_with)
        print(f"  {cat:30s}  positives={int(flags[cat].sum()):>6,}  ({time.time()-t1:.1f}s)")

    # Join to BDSPPatientID
    flags = flags.merge(cohort[["EMPI", "BDSPPatientID", "Predicted_Stage"]],
                        on="EMPI", how="left")

    # Deidentified output: BDSPPatientID + comorbidity bools (no EMPI/PatientID)
    deid = flags.drop(columns=["EMPI"]).rename(columns={"Predicted_Stage": "group"})
    deid["group"] = deid["group"].map({"Dementia": "DEM", "MCI": "MCI", "No Dementia": "CN"})
    deid = deid.drop_duplicates("BDSPPatientID").reset_index(drop=True)
    cols = ["BDSPPatientID", "group"] + [c for c, *_ in RULES]
    deid = deid[cols]
    deid.to_csv(OUT_DEID, index=False)
    print(f"\nwrote {OUT_DEID}  ({deid.shape[0]:,} × {deid.shape[1]})")

    # Build paper Table 2 comorbidity rows: counts (%) by group + z-test p-value
    from statsmodels.stats.proportion import proportions_ztest
    cn_n = (deid["group"] == "CN").sum()
    mci_n = (deid["group"] == "MCI").sum()
    dem_n = (deid["group"] == "DEM").sum()
    rows = []
    for cat, *_ in RULES:
        cn_pos = int(deid.loc[deid["group"] == "CN", cat].sum())
        mci_pos = int(deid.loc[deid["group"] == "MCI", cat].sum())
        dem_pos = int(deid.loc[deid["group"] == "DEM", cat].sum())
        # 2-sample z-tests vs CN
        _, p_mci = proportions_ztest([mci_pos, cn_pos], [mci_n, cn_n])
        _, p_dem = proportions_ztest([dem_pos, cn_pos], [dem_n, cn_n])

        def stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return ""
        rows.append({
            "Comorbidity": cat.replace("_", " "),
            "Total": f"{cn_pos+mci_pos+dem_pos} ({100*(cn_pos+mci_pos+dem_pos)/(cn_n+mci_n+dem_n):.0f}%)",
            "CN": f"{cn_pos} ({100*cn_pos/cn_n:.0f}%)",
            "MCI": f"{mci_pos} ({100*mci_pos/mci_n:.0f}%){stars(p_mci)}",
            "DEM": f"{dem_pos} ({100*dem_pos/dem_n:.0f}%){stars(p_dem)}",
            "p_MCI_vs_CN": f"{p_mci:.2e}",
            "p_DEM_vs_CN": f"{p_dem:.2e}",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(FIGS / "table2_comorbidities.csv", index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nwrote {FIGS / 'table2_comorbidities.csv'}")
    print(f"per-group n: CN={cn_n}, MCI={mci_n}, DEM={dem_n}")


if __name__ == "__main__":
    main()
