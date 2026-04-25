"""Reproduce the SLEEP 2023 paper's three binary classification AUROC/AUPRC.

Mirrors the recipe in `binary_dementia_classification_DM_v_CN_V2.ipynb`:
  - Cohort filter on study_criteria_table_label_V6.xlsx Predicted_Stage
  - Random 1000-CN undersample (for DM_v_CN and MCI_v_CN)
  - Inner 5-fold GridSearchCV / Outer 5-fold StratifiedKFold
  - Pipeline: KNNImputer -> StandardScaler -> SelectKBest(k=350) -> SelectFromModel(RF) -> classifier
  - Three classifiers: LR (elasticnet/saga), SVM (RBF), RF
  - Three tasks: DM_v_CN, MCI_v_CN, DM_MCI_v_CN

Compares to paper headlines:
  DEM vs CN:      AUROC 0.78  AUPRC 0.22
  MCI vs CN:      AUROC 0.73  AUPRC 0.18
  DEM/MCI vs CN:  AUROC 0.76  AUPRC 0.32

Run from repo root (Elissa-bai), with the staged PHI in `_box_inspect/`:
    python3 _repro/verify_paper.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
FEATURES = ROOT / "_box_inspect/dementia_detection_other/eeg_data/study_features_table_v4.csv"
OUT = ROOT / "_repro/verify_paper_results.json"

SEED = 7  # the notebook uses random_state=7 for the classifier; we set seed everywhere
N_CN_SAMPLE = 1000  # undersampled in the notebook for class balance

PAPER = {
    "DM_v_CN":     {"AUROC": 0.78, "AUPRC": 0.22},
    "MCI_v_CN":    {"AUROC": 0.73, "AUPRC": 0.18},
    "DM_MCI_v_CN": {"AUROC": 0.76, "AUPRC": 0.32},
}

# Reduced parameter grids for verification speed (notebook scans 10 thresholds × {l1,l2})
PARAM_GRIDS = {
    "LR": {
        "feature_selection__threshold": [0.0005, 0.005],
        "classification__C": [0.1, 1.0],
        "classification__l1_ratio": [0.5],
    },
    "SVM": {
        "feature_selection__threshold": [0.0005, 0.005],
        "classification__C": [0.1, 1.0],
        "classification__gamma": ["scale"],
    },
    "RF": {
        "feature_selection__threshold": [0.0005, 0.005],
        "classification__max_depth": [None, 8],
        "classification__min_samples_leaf": [1, 5],
    },
}


def load_cohort_and_features():
    print(f"loading cohort: {COHORT.name}")
    cohort = pd.read_excel(COHORT)
    print(f"  cohort: {cohort.shape}")
    print(f"  Predicted_Stage counts: {cohort['Predicted_Stage'].value_counts(dropna=False).to_dict()}")
    print(f"loading features: {FEATURES.name}  (~352 MB; takes ~30s)")
    feats = pd.read_csv(FEATURES, low_memory=False)
    print(f"  features: {feats.shape}")
    return cohort, feats


def build_task(cohort, feats, task: str, rng):
    """Return X (DataFrame, features only), y (np.ndarray, 0/1), label_counts dict."""
    if task == "DM_v_CN":
        pos_stage, neg_stage = "Dementia", "No Dementia"
        balance_neg_to = N_CN_SAMPLE
    elif task == "MCI_v_CN":
        pos_stage, neg_stage = "MCI", "No Dementia"
        balance_neg_to = N_CN_SAMPLE
    elif task == "DM_MCI_v_CN":
        pos_stage = ("Dementia", "MCI")
        neg_stage = "No Dementia"
        balance_neg_to = N_CN_SAMPLE
    else:
        raise ValueError(task)

    is_pos = cohort["Predicted_Stage"].isin(pos_stage if isinstance(pos_stage, tuple) else (pos_stage,))
    is_neg = cohort["Predicted_Stage"] == neg_stage

    pos_df = cohort.loc[is_pos, ["FolderName"]].assign(_y=1)
    neg_df = cohort.loc[is_neg, ["FolderName"]].assign(_y=0)
    if balance_neg_to and len(neg_df) > balance_neg_to:
        neg_df = neg_df.sample(n=balance_neg_to, random_state=rng.integers(1, 1_000_000))

    label_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Merge with features
    merged = label_df.merge(feats, on="FolderName", how="left")
    # Notebook drops rows with too many NaNs (>800)
    merged = merged.loc[merged.isna().sum(axis=1) <= 800].copy()
    merged = merged.drop_duplicates(subset=["FolderName"])

    feature_cols = [c for c in merged.columns if c not in ("FolderName", "_y")]
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = merged["_y"].to_numpy()

    return X, y, {
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_total": len(y),
        "n_features": X.shape[1],
    }


def build_pipeline(model_name: str, rng_seed: int) -> Pipeline:
    skb = SelectKBest(f_classif, k=350)
    fs_model = RandomForestClassifier(random_state=rng_seed, class_weight="balanced", n_jobs=-1)

    if model_name == "LR":
        clf = LogisticRegression(random_state=rng_seed, penalty="elasticnet", solver="saga",
                                 class_weight="balanced", max_iter=500)
    elif model_name == "SVM":
        clf = SVC(probability=True, kernel="rbf", random_state=rng_seed,
                  class_weight="balanced")
    elif model_name == "RF":
        clf = RandomForestClassifier(random_state=rng_seed, class_weight="balanced",
                                      n_estimators=200, n_jobs=-1)
    else:
        raise ValueError(model_name)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=10)),
        ("filter", skb),
        ("feature_selection", SelectFromModel(fs_model)),
        ("classification", clf),
    ])


def run_task(task: str, X: pd.DataFrame, y: np.ndarray, model_name: str, n_outer=5, n_inner=3):
    """Outer K-fold AUROC/AUPRC with inner GridSearchCV. Returns dict."""
    cv_outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(y), dtype=float)
    oof_y = np.zeros(len(y), dtype=int)

    for fold_i, (tr, te) in enumerate(cv_outer.split(X, y)):
        cv_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold_i)
        pipe = build_pipeline(model_name, rng_seed=SEED)
        gs = GridSearchCV(pipe, PARAM_GRIDS[model_name], cv=cv_inner,
                          scoring="roc_auc", n_jobs=-1, refit=True)
        t0 = time.time()
        gs.fit(X.iloc[tr], y[tr])
        prob = gs.predict_proba(X.iloc[te])[:, 1]
        oof_prob[te] = prob
        oof_y[te] = y[te]
        print(f"    fold {fold_i+1}/{n_outer} ({time.time()-t0:.1f}s) "
              f"AUC={roc_auc_score(y[te], prob):.3f}  best={gs.best_params_}")

    auroc = roc_auc_score(y, oof_prob)
    auprc = average_precision_score(y, oof_prob)
    return {"AUROC": float(auroc), "AUPRC": float(auprc)}


def main():
    cohort, feats = load_cohort_and_features()
    rng = np.random.default_rng(SEED)

    results = {}
    for task in ["DM_v_CN", "MCI_v_CN", "DM_MCI_v_CN"]:
        print(f"\n=== {task} ===")
        X, y, counts = build_task(cohort, feats, task, rng)
        print(f"  cohort: {counts}")
        results[task] = {"counts": counts, "models": {}}
        for model_name in ["LR", "SVM", "RF"]:
            print(f"  --- {model_name} ---")
            r = run_task(task, X, y, model_name)
            results[task]["models"][model_name] = r
            paper = PAPER[task]
            print(f"  >> {model_name}: AUROC={r['AUROC']:.3f} (paper {paper['AUROC']:.2f})  "
                  f"AUPRC={r['AUPRC']:.3f} (paper {paper['AUPRC']:.2f})")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'task':<12s} {'model':<5s} {'AUROC':>7s} {'paper':>7s} {'AUPRC':>7s} {'paper':>7s}")
    for task, blob in results.items():
        for model_name, r in blob["models"].items():
            paper = PAPER[task]
            print(f"{task:<12s} {model_name:<5s} {r['AUROC']:>7.3f} {paper['AUROC']:>7.2f} "
                  f"{r['AUPRC']:>7.3f} {paper['AUPRC']:>7.2f}")


if __name__ == "__main__":
    main()
