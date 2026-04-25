"""Figure 4 — age × diagnostic-group interaction effects across top features.

Implements paper Eq. (1):
    Feature = β0 + β1·Age + β2·DEM + β3·Age·DEM + β4·MCI + β5·Age·MCI

Per-feature OLS fit on z-scored features (KNN-imputed where missing). Top
features are then selected by |β / SE| (i.e. |t-statistic|) for each of the
four terms of interest (β2 = DEM main, β3 = DEM×Age, β4 = MCI main,
β5 = MCI×Age) and plotted as Feature vs Age trajectories by group.

Output: _repro/figures/figure4_age_interaction.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
FEATURES = ROOT / "_box_inspect/dementia_detection_other/eeg_data/study_features_table_v4.csv"
FIGS = ROOT / "_repro/figures"
COL = {"DEM": "#d92a2a", "MCI": "#e78a2a", "CN": "#3a7eb8"}


def load():
    cohort = pd.read_excel(COHORT)
    cohort = cohort[cohort["Predicted_Stage"].isin(["Dementia", "MCI", "No Dementia"])]
    cohort = cohort[["FolderName", "Age", "Predicted_Stage"]].drop_duplicates("FolderName")
    cohort["Age"] = pd.to_numeric(cohort["Age"], errors="coerce")
    cohort["DEM"] = (cohort["Predicted_Stage"] == "Dementia").astype(int)
    cohort["MCI"] = (cohort["Predicted_Stage"] == "MCI").astype(int)
    cohort["group"] = cohort["Predicted_Stage"].map(
        {"Dementia": "DEM", "MCI": "MCI", "No Dementia": "CN"})

    feats = pd.read_csv(FEATURES, low_memory=False)
    if "Age" in feats.columns:
        feats = feats.drop(columns=["Age"])
    df = cohort.merge(feats, on="FolderName", how="inner")
    df = df.dropna(subset=["Age"])
    print(f"merged: {df.shape}")

    feat_cols = [c for c in feats.columns if c not in ("FolderName", "Age", "Sex_Male")]
    return df, feat_cols


def fit_per_feature(df, feat_cols):
    """Fit Eq. (1) per feature on z-scored values; return DataFrame of effects."""
    print("imputing + standardizing...")
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X_imp = KNNImputer(n_neighbors=10).fit_transform(X)
    X_z = StandardScaler().fit_transform(X_imp)

    age = df["Age"].to_numpy(dtype=float)
    age_z = (age - age.mean()) / age.std(ddof=0)
    dem = df["DEM"].to_numpy()
    mci = df["MCI"].to_numpy()
    age_dem = age_z * dem
    age_mci = age_z * mci

    print("fitting per-feature OLS...")
    rows = []
    for i, fname in enumerate(feat_cols):
        y = X_z[:, i]
        if np.std(y) < 1e-9: continue
        Xmat = np.column_stack([np.ones_like(y), age_z, dem, age_dem, mci, age_mci])
        try:
            res = sm.OLS(y, Xmat).fit()
            b = res.params
            se = res.bse
            rows.append({
                "feature": fname,
                "b_age": b[1], "b_DEM": b[2], "b_AgeDEM": b[3],
                "b_MCI": b[4], "b_AgeMCI": b[5],
                "t_DEM": b[2]/se[2] if se[2] > 0 else np.nan,
                "t_AgeDEM": b[3]/se[3] if se[3] > 0 else np.nan,
                "t_MCI": b[4]/se[4] if se[4] > 0 else np.nan,
                "t_AgeMCI": b[5]/se[5] if se[5] > 0 else np.nan,
            })
        except Exception:
            continue
    eff = pd.DataFrame(rows)
    print(f"  fit {len(eff)} features")
    return eff, X_z, age, age_z


def make_figure4():
    df, feat_cols = load()
    eff, X_z, age, age_z = fit_per_feature(df, feat_cols)

    # Pick top 3 features per term by |t|
    panels = []
    for label, col in [("DEM (β₂)", "t_DEM"),
                       ("DEM × Age (β₃)", "t_AgeDEM"),
                       ("MCI (β₄)", "t_MCI"),
                       ("MCI × Age (β₅)", "t_AgeMCI")]:
        top = eff.iloc[eff[col].abs().sort_values(ascending=False).index[:3]]
        for _, r in top.iterrows():
            panels.append((label, r["feature"]))

    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    age_grid = np.linspace(45, 95, 50)
    age_grid_z = (age_grid - age.mean()) / age.std(ddof=0)
    feat_idx = {f: i for i, f in enumerate(feat_cols)}

    for i, (term, feat) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        # Refit
        y = X_z[:, feat_idx[feat]]
        dem = df["DEM"].to_numpy(); mci = df["MCI"].to_numpy()
        Xmat = np.column_stack([np.ones_like(y), age_z, dem, age_z*dem, mci, age_z*mci])
        b = sm.OLS(y, Xmat).fit().params

        # Predicted trajectories per group
        for grp, dem_v, mci_v in [("CN", 0, 0), ("MCI", 0, 1), ("DEM", 1, 0)]:
            pred = (b[0] + b[1]*age_grid_z + b[2]*dem_v + b[3]*age_grid_z*dem_v
                    + b[4]*mci_v + b[5]*age_grid_z*mci_v)
            ax.plot(age_grid, pred, color=COL[grp], lw=2, label=grp)
            # Scatter mean per age-decile
            dfp = df.assign(y=y)
            sub = dfp[dfp["group"] == grp]
            for lo in range(45, 95, 10):
                msub = (sub["Age"] >= lo) & (sub["Age"] < lo + 10)
                if msub.sum() >= 10:
                    ax.plot(lo + 5, sub.loc[msub, "y"].mean(),
                            "o", color=COL[grp], alpha=0.3, ms=4)

        # Term annotation
        ax.set_title(f"{term}\n{feat[:36]+'…' if len(feat) > 36 else feat}",
                     fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Age (years)")
        if i % 3 == 0:
            ax.set_ylabel("Feature (z)")
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 4 — Age × diagnostic-group interaction across top features"
                 " (per term of paper Eq. 1)", y=0.995)
    fig.tight_layout()
    out = FIGS / "figure4_age_interaction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # also save the effect table
    eff.sort_values("t_DEM", key=lambda s: s.abs(), ascending=False).head(50).to_csv(
        FIGS / "figure4_top_terms.csv", index=False)
    print(f"wrote {FIGS / 'figure4_top_terms.csv'}")


if __name__ == "__main__":
    make_figure4()
