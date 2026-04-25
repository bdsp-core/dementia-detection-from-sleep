"""Strip outputs from canonical SLEEP 2023 notebooks and stage them in the repo.

Reads notebooks from `_box_inspect/dementia_detection_code/`, removes all cell
outputs and execution counts (which can contain PHI from prior runs), and
writes them under `sleep-dementia-detection/code/<group>/`.

Run from repo root: `python3 scripts/stage_canonical_notebooks.py`
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "_box_inspect/dementia_detection_code"
DST = ROOT / "sleep-dementia-detection/code"

# Map: source notebook → destination subfolder
STAGING = {
    "01_phenotyping": [
        "assign_study_groups_V3.ipynb",
        "wrangle_neuropsychiatric_scores.ipynb",
        "mine_neuropsychiatric_scores_V2.ipynb",
        "ICD_Code_Analysis.ipynb",
    ],
    "02_features": [
        "compile_brain_age_features.ipynb",
        "combine_luna_output.ipynb",
        "spectral_analysis_V7.ipynb",
        "spindle_analysis_V2.ipynb",
        "alpha_analysis_V3.ipynb",
        "compute_alpha3_alpha2_V2.ipynb",
        "EEG_Coherence_Analysis.ipynb",
        "assess_missing_features.ipynb",
        "Feature_display_name.csv",  # feature dictionary, no outputs
    ],
    "04_model": [
        "binary_dementia_classification_DM_v_CN_V2.ipynb",
        "binary_dementia_classification_MCI_v_CN.ipynb",
        "binary_dementia_classification_DM_MCI_v_CN.ipynb",
        "binary_dementia_classification_MCI_v_DM.ipynb",
        "multiclass_dementia_classification_V14.ipynb",
        "CDR_classification_V5.ipynb",
        "nested_cross_validation_test.ipynb",
        "neuropsychiatric_score_prediction_V2.ipynb",
        "univarite_feature_selection.ipynb",
        "Feature selection Lasso NEW.ipynb",
        "benjamini_hochberg.ipynb",
        "cuzick_test.ipynb",
    ],
    "05_figures": [
        "plot_confusion_matrix.ipynb",
        "sleepdata_summary_statistics_V3.ipynb",
    ],
}

PHI_PATTERNS = ["PatientID", "MRN", "EMPI", "DateOfBirth",
                "FolderName", "FirstName", "LastName"]


def strip_notebook(nb: dict) -> tuple[dict, int]:
    """Strip outputs/execution_count from every code cell. Return (nb, n_phi_hits_pre_strip)."""
    phi_hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        # check for PHI in outputs before stripping
        for output in cell.get("outputs", []):
            text = json.dumps(output)
            if any(p in text for p in PHI_PATTERNS):
                phi_hits += 1
                break
        cell["outputs"] = []
        cell["execution_count"] = None
    # Reset notebook-level metadata that can leak paths
    nb.setdefault("metadata", {})
    return nb, phi_hits


def main():
    DST.mkdir(parents=True, exist_ok=True)
    summary = []
    for subfolder, files in STAGING.items():
        out_dir = DST / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = SRC / fname
            dst = out_dir / fname
            if not src.exists():
                summary.append(f"  MISSING: {fname}")
                continue
            if fname.endswith(".ipynb"):
                with open(src) as f:
                    nb = json.load(f)
                nb, phi_hits = strip_notebook(nb)
                with open(dst, "w") as f:
                    json.dump(nb, f, indent=1)
                summary.append(f"  staged {subfolder}/{fname} (stripped {phi_hits} PHI-bearing output cells)")
            else:
                shutil.copy2(src, dst)
                summary.append(f"  copied {subfolder}/{fname}")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
