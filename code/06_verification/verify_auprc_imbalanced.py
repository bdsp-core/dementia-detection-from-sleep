"""Verify AUPRC matches the paper by evaluating on the FULL imbalanced CN
cohort while training is still balanced via undersampling.

Single-task (DM_v_CN), single-model (LR) sanity check. Outer 5-fold split
of the FULL set (no CN subsample); inside each fold, training subsamples
CN to 1000 to balance the gradient, but testing is on the full held-out
CN slice (which preserves the natural prevalence).
"""
from __future__ import annotations
import time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
FEATURES = ROOT / "_box_inspect/dementia_detection_other/eeg_data/study_features_table_v4.csv"
SEED = 7
N_CN_SAMPLE_TRAIN = 1000

def main():
    print("loading...")
    cohort = pd.read_excel(COHORT)
    feats = pd.read_csv(FEATURES, low_memory=False)

    # Build full DM_v_CN dataset (no undersample)
    dem = cohort.loc[cohort["Predicted_Stage"] == "Dementia", ["FolderName"]].assign(_y=1)
    cn = cohort.loc[cohort["Predicted_Stage"] == "No Dementia", ["FolderName"]].assign(_y=0)
    df = pd.concat([dem, cn], ignore_index=True)
    df = df.merge(feats, on="FolderName", how="left")
    df = df.loc[df.isna().sum(axis=1) <= 800].drop_duplicates("FolderName")

    feat_cols = [c for c in df.columns if c not in ("FolderName", "_y")]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = df["_y"].to_numpy()
    print(f"FULL DM_v_CN cohort: n={len(y)}, n_dem={int(y.sum())}, n_cn={int((y==0).sum())}, "
          f"prevalence={y.mean():.3f}, features={X.shape[1]}")
    print(f"Paper cohort: ~7,602 (339 DEM + 7,263 CN), prevalence=0.045")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    oof_prob = np.zeros(len(y), dtype=float)

    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        # Undersample CN within the training fold only
        tr_dem = tr[y[tr] == 1]
        tr_cn_all = tr[y[tr] == 0]
        tr_cn = rng.choice(tr_cn_all, size=min(N_CN_SAMPLE_TRAIN, len(tr_cn_all)), replace=False)
        tr_balanced = np.concatenate([tr_dem, tr_cn])
        rng.shuffle(tr_balanced)

        skb = SelectKBest(f_classif, k=350)
        rf_for_fs = RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1)
        clf = LogisticRegression(random_state=SEED, penalty="elasticnet", solver="saga",
                                 l1_ratio=0.5, C=0.1, class_weight="balanced", max_iter=500)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=10)),
            ("filter", skb),
            ("feature_selection", SelectFromModel(rf_for_fs, threshold=0.005)),
            ("classification", clf),
        ])
        t0 = time.time()
        pipe.fit(X.iloc[tr_balanced], y[tr_balanced])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        oof_prob[te] = prob
        auc_te = roc_auc_score(y[te], prob)
        ap_te = average_precision_score(y[te], prob)
        print(f"  fold {fold_i+1}/5 ({time.time()-t0:.1f}s)  "
              f"n_train={len(tr_balanced)} (n_dem={len(tr_dem)}, n_cn={len(tr_cn)})  "
              f"AUROC={auc_te:.3f}  AUPRC={ap_te:.3f}  test prev={y[te].mean():.3f}")

    auroc = roc_auc_score(y, oof_prob)
    auprc = average_precision_score(y, oof_prob)
    print(f"\nOVERALL DM_v_CN (LR, train-balanced/test-imbalanced):")
    print(f"  AUROC = {auroc:.3f}   (paper 0.78)")
    print(f"  AUPRC = {auprc:.3f}   (paper 0.22)")

if __name__ == "__main__":
    main()
