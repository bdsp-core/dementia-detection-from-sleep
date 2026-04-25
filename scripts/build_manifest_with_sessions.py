"""Augment psg_manifest.csv with the BIDS session number and full S3 path
to the .edf signal, derived from per-subject scans.tsv files in
s3://bdsp-opendata-repository/PSG/bids/S0001/.

Output overwrites _s3_stage/sleep-dementia-detection/psg_manifest.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
SCANS_DIR = ROOT / "_box_inspect/s3_scans_tsv"
MANIFEST = ROOT / "_s3_stage/sleep-dementia-detection/psg_manifest.csv"

S3_BASE = "s3://bdsp-opendata-repository/PSG/bids/S0001"


def main():
    print(f"reading {len(list(SCANS_DIR.rglob('*scans.tsv'))):,} scans.tsv files...")
    rows = []
    for tsv_path in SCANS_DIR.rglob("*scans.tsv"):
        # path: .../sub-S0001<BDSP>/ses-<N>/sub-S0001<BDSP>_ses-<N>_scans.tsv
        rel = tsv_path.relative_to(SCANS_DIR)
        parts = rel.parts
        # parts[0] = sub-S0001<BDSP>, parts[1] = ses-<N>
        if len(parts) < 3 or not parts[0].startswith("sub-S0001"):
            continue
        bdsp = parts[0].replace("sub-S0001", "")
        ses = parts[1]
        try:
            tsv = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
            tsv["acq_time"] = pd.to_datetime(tsv["acq_time"], errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            continue
        for _, r in tsv.iterrows():
            if pd.isna(r["acq_time"]): continue
            rows.append({
                "BDSPPatientID": bdsp,
                "session": ses,
                "acq_time": r["acq_time"],
                "edf_path_relative": r["filename"],
            })
    sessions = pd.DataFrame(rows)
    sessions["acq_date"] = sessions["acq_time"].dt.date.astype(str)
    print(f"  unique (subject, session) records: {len(sessions):,}")
    print(f"  unique BDSPPatientIDs: {sessions['BDSPPatientID'].nunique():,}")

    # Load manifest, normalize date for join
    m = pd.read_csv(MANIFEST, low_memory=False)
    m["BDSPPatientID"] = m["BDSPPatientID"].astype(str)
    m["DOVshifted_date"] = pd.to_datetime(m["DOVshifted"], errors="coerce").dt.date.astype(str)

    # Match by (BDSPPatientID, date)
    j = m.merge(sessions, left_on=["BDSPPatientID", "DOVshifted_date"],
                right_on=["BDSPPatientID", "acq_date"], how="left")
    print(f"manifest rows: {len(m):,}, with session match: {j['session'].notna().sum():,}")

    # Build the s3_path
    def build_path(row):
        if pd.isna(row.get("session")):
            return ""
        bdsp = row["BDSPPatientID"]
        ses = row["session"]
        return (f"{S3_BASE}/sub-S0001{bdsp}/{ses}/eeg/"
                f"sub-S0001{bdsp}_{ses}_task-psg_eeg.edf")

    j["s3_path"] = j.apply(build_path, axis=1)
    j = j.drop(columns=["acq_time", "acq_date", "edf_path_relative", "DOVshifted_date"])

    # Ensure column order matches the prior manifest plus the new session col
    cols = ["BDSPPatientID", "HashID", "FileNameNew", "DOVshifted",
            "Sex", "AgeAtPSG", "PSGType", "group", "session", "s3_path"]
    cols = [c for c in cols if c in j.columns] + [c for c in j.columns if c not in cols]
    j = j[cols].drop_duplicates(subset=["BDSPPatientID", "FileNameNew"], keep="first")
    j.to_csv(MANIFEST, index=False)
    print(f"\nwrote {MANIFEST}  ({j.shape[0]:,} × {j.shape[1]})")
    print(j.head(3).to_string())
    print()
    print(f"PSGs with resolved s3_path: {(j['s3_path'] != '').sum():,} / {len(j):,}")
    print(f"PSGs without S3 match: {(j['s3_path'] == '').sum():,} (BIDS conversion may have excluded these)")


if __name__ == "__main__":
    main()
