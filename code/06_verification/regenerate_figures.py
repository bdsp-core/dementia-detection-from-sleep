"""Regenerate the SLEEP 2023 paper's tables and figures.

Outputs land under `_repro/figures/`. Each figure is a side-by-side
visual comparison candidate against the PDF.

Inputs (PHI; staged locally):
  _box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx
  _box_inspect/dementia_detection_code/{CN,MCI,Dementia}_{Wake,N1,N2,N3,REM}_EEG_specs.npy
  _box_inspect/dementia_detection_code/freq.npy
  _box_inspect/dementia_detection_code/{DM_v_CN,MCI_v_CN,DM_MCI_v_CN}_{LG,SVM,RF}_scores.pickle
  _box_inspect/dementia_detection_code/{DM_v_CN,MCI_v_CN}_OR_df_V2.csv

Run from repo root:
  python3 _repro/regenerate_figures.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")

ROOT = Path("/Users/mwestover/GithubRepos/Elissa-bai")
COHORT = ROOT / "_box_inspect/dementia_detection_other/medical_data/study_criteria_table_label_V6.xlsx"
CODE = ROOT / "_box_inspect/dementia_detection_code"
FIGS = ROOT / "_repro/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Group colors that match the paper (Fig 3, Fig 2)
COL = {"DEM": "#d92a2a", "MCI": "#e78a2a", "CN": "#3a7eb8"}


# ---------------------------------------------------------------------------
# Table 2 — Group characteristics
# ---------------------------------------------------------------------------

def make_table2():
    print("\n[Table 2] Group characteristics")
    cohort = pd.read_excel(COHORT)
    ana = cohort[cohort["Predicted_Stage"].isin(["Dementia", "MCI", "No Dementia"])].copy()
    ana["group"] = ana["Predicted_Stage"].map(
        {"Dementia": "DEM", "MCI": "MCI", "No Dementia": "CN"}
    )

    rows = []
    rows.append({"row": "Number of PSGs", "Total": len(ana),
                 **{g: int((ana["group"] == g).sum()) for g in ["CN", "MCI", "DEM"]}})
    n_part = {g: ana.loc[ana["group"] == g, "PatientID"].nunique() for g in ["CN", "MCI", "DEM"]}
    rows.append({"row": "Number of participants", "Total": ana["PatientID"].nunique(), **n_part})

    age = pd.to_numeric(ana["Age"], errors="coerce")
    rows.append({"row": "Age (median [IQR])",
                 "Total": f"{age.median():.0f} ({age.quantile(.25):.0f}-{age.quantile(.75):.0f})",
                 **{g: f"{age[ana['group'] == g].median():.0f} "
                       f"({age[ana['group'] == g].quantile(.25):.0f}-"
                       f"{age[ana['group'] == g].quantile(.75):.0f})"
                    for g in ["CN", "MCI", "DEM"]}})

    for sex in ["Female", "Male"]:
        f = ana["Sex"] == sex
        rows.append({"row": f"Sex: {sex}",
                     "Total": f"{f.sum()} ({100*f.mean():.0f}%)",
                     **{g: (lambda gg=g: (lambda sub=ana[ana['group'] == gg]:
                            f"{(sub['Sex'] == sex).sum()} "
                            f"({100*(sub['Sex'] == sex).mean():.0f}%)")())()
                        for g in ["CN", "MCI", "DEM"]}})

    # Normalize TypeOfTest into the paper's three categories
    def norm_type(x):
        x = str(x).strip().lower()
        if "split" in x: return "Split night"
        if "titrat" in x or "cpap" in x: return "All night titration"
        if "diagnostic" in x: return "Diagnostic"
        return None
    ana["_typ"] = ana["TypeOfTest"].apply(norm_type)
    for ttype in ["Diagnostic", "All night titration", "Split night"]:
        tt = ana["_typ"] == ttype
        rows.append({"row": f"Type: {ttype}",
                     "Total": f"{tt.sum()} ({100*tt.mean():.0f}%)",
                     **{g: (lambda gg=g: (lambda sub=ana[ana['group'] == gg]:
                            f"{(sub['_typ'] == ttype).sum()} "
                            f"({100*(sub['_typ'] == ttype).mean():.0f}%)")())()
                        for g in ["CN", "MCI", "DEM"]}})

    out = pd.DataFrame(rows)[["row", "Total", "CN", "MCI", "DEM"]]
    out.to_csv(FIGS / "table2_group_characteristics.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n  wrote {FIGS / 'table2_group_characteristics.csv'}")


# ---------------------------------------------------------------------------
# Figure 3 — Power spectra by stage (occipital), DEM/MCI/CN
# ---------------------------------------------------------------------------

def make_figure3():
    print("\n[Figure 3] Power spectra by sleep stage (occipital), DEM vs MCI vs CN")
    freq = np.load(CODE / "freq.npy")

    stages = ["Wake", "N1", "N2", "N3", "REM"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    axes = axes.flatten()

    for i, stage in enumerate(stages):
        ax = axes[i]
        for grp, fname in [("DEM", f"Dementia_{stage}_EEG_specs.npy"),
                           ("MCI", f"MCI_{stage}_EEG_specs.npy"),
                           ("CN", f"CN_{stage}_EEG_specs.npy")]:
            spec = np.load(CODE / fname)
            # spec shape varies by stage; reduce to occipital channels mean
            # Conventionally last 2 channels are O1M2/O2M1 in 6-channel montage
            if spec.ndim == 3:  # (n_subj, n_freq, n_ch)
                # average occipital channels across subjects
                occ = spec[..., -2:].mean(axis=(0, 2))
            elif spec.ndim == 2:  # already (n_freq,) or (n_freq, n_ch)
                occ = spec.mean(axis=-1) if spec.shape[-1] != len(freq) else spec
            else:
                occ = spec
            occ_db = 10 * np.log10(np.maximum(occ, 1e-10))
            ax.plot(freq, occ_db, label=grp, color=COL[grp], lw=2)
        ax.set_xlim(0, 20)
        ax.set_ylim(-15, 45)
        ax.set_title(f"{stage} Occipital")
        ax.set_xlabel("Frequency (Hz)")
        if i % 3 == 0:
            ax.set_ylabel("dB")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Figure 3 — Occipital EEG power spectra by sleep stage and diagnostic group",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(FIGS / "figure3_spectra_by_stage.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "figure3_spectra_by_stage.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'figure3_spectra_by_stage.png'}")


# ---------------------------------------------------------------------------
# Figure 5 — ROC + PR curves for the three binary tasks
# ---------------------------------------------------------------------------

def load_pickle_scores(prefix: str, model: str):
    path = CODE / f"{prefix}_{model}_scores.pickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        y_test, y_prob, y_pred, cm = pickle.load(f)
    return np.asarray(y_test), np.asarray(y_prob), np.asarray(y_pred), np.asarray(cm)


def make_figure5():
    print("\n[Figure 5] ROC + PR curves")
    tasks = [("DM_v_CN", "DEM vs CN"),
             ("MCI_v_CN", "MCI vs CN"),
             ("DM_MCI_v_CN", "DEM/MCI vs CN")]
    models = [("LG", "LR"), ("SVM", "SVM"), ("RF", "RF")]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))

    summary_rows = []
    for col, (task_key, task_label) in enumerate(tasks):
        # ROC
        ax_roc = axes[0, col]
        ax_pr = axes[1, col]

        for mfile, mlabel in models:
            res = load_pickle_scores(task_key, mfile)
            if res is None:
                print(f"  WARN: missing {task_key}_{mfile}_scores.pickle")
                continue
            y_test, y_prob, y_pred, cm = res
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            au = roc_auc_score(y_test, y_prob)
            ax_roc.plot(fpr, tpr, lw=2, label=f"{mlabel} (AUC={au:.2f})")
            # PR
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            ax_pr.plot(rec, prec, lw=2, label=f"{mlabel} (AP={ap:.2f})")

            summary_rows.append({"task": task_label, "model": mlabel,
                                 "AUROC": au, "AUPRC": ap})

        ax_roc.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax_roc.set_xlabel("False positive rate")
        if col == 0:
            ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"{task_label}\nROC")
        ax_roc.legend(loc="lower right", fontsize=9, frameon=False)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

        ax_pr.set_xlabel("Recall")
        if col == 0:
            ax_pr.set_ylabel("Precision")
        ax_pr.set_title("PR")
        ax_pr.legend(loc="upper right", fontsize=9, frameon=False)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    fig.suptitle("Figure 5 — ROC and Precision-Recall curves for the three binary tasks "
                 "(LR/SVM/RF, balanced eval from saved pickle scores)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGS / "figure5_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "figure5_roc_pr_curves.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'figure5_roc_pr_curves.png'}")

    # Summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(FIGS / "figure5_summary.csv", index=False)
    print("\n  Summary (from saved pickled fold predictions):")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Figure 2 — Top discriminative features by OR
# ---------------------------------------------------------------------------

def parse_feature_band(name) -> str:
    """Coarse band classification by feature-name keyword."""
    if not isinstance(name, str):
        return "General"
    n = name.lower()
    if "alpha" in n or "iaf" in n:
        if "alpha1" in n: return "α1 (8-12 Hz)"
        if "alpha2" in n: return "α2"
        if "alpha3" in n: return "α3"
        return "α (8-12 Hz)"
    if "theta" in n: return "θ (4-8 Hz)"
    if "delta" in n: return "δ (0.5-4 Hz)"
    if "sigma" in n or "spindle" in n: return "σ (11-15 Hz)"
    if "slow" in n or "so_" in n: return "Slow (<0.5 Hz)"
    return "General"


BAND_COL = {"σ (11-15 Hz)": "#3eb55a", "α (8-12 Hz)": "#d92a2a",
            "α1 (8-12 Hz)": "#d92a2a", "α2": "#d92a2a", "α3": "#d92a2a",
            "θ (4-8 Hz)": "#e6c33b", "δ (0.5-4 Hz)": "#3a7eb8",
            "Slow (<0.5 Hz)": "#cf57a8", "General": "#777777"}


STAGE_KEYS = [("Wake", ["W_", "_W_", "_W "]),
              ("N1", ["N1_", "_N1_"]),
              ("N2", ["N2_", "_N2_"]),
              ("N3", ["N3_", "_N3_"]),
              ("REM", ["R_", "_R_", "REM"]),
              ("Macro", ["TST", "TRT", "WASO", "SE", "SOL", "REML", "N1%", "N2%", "N3%", "REM%", "SFI"])]


def stage_of(name):
    if not isinstance(name, str): return "Macro"
    for stage, keys in STAGE_KEYS:
        for k in keys:
            if k in name:
                return stage
    return "Macro"


def make_figure2():
    print("\n[Figure 2] Top discriminative features by OR (stage-grouped)")
    stages = ["Wake", "N1", "N2", "N3", "REM", "Macro"]

    fig, axes = plt.subplots(len(stages), 2, figsize=(14, 13),
                             sharex=True, gridspec_kw={"hspace": 0.35})
    for col, (task_key, task_label) in enumerate([("DM_v_CN", "A. DEM vs CN"),
                                                   ("MCI_v_CN", "B. MCI vs CN")]):
        df = pd.read_csv(CODE / f"{task_key}_OR_df_V2.csv")
        df = df.dropna(subset=["Odds_Ratio", "p", "feature_name"])
        df = df[df["p"] < 0.05]
        df["log_OR"] = np.log(df["Odds_Ratio"])
        df["abs_log_OR"] = df["log_OR"].abs()
        df["band"] = df["feature_name"].apply(parse_feature_band)
        df["stage"] = df["feature_name"].apply(stage_of)

        for r, stage in enumerate(stages):
            ax = axes[r, col]
            sub = df[df["stage"] == stage]
            n_per_side = 6
            top_pos = sub[sub["Odds_Ratio"] > 1].nlargest(n_per_side, "abs_log_OR")
            top_neg = sub[sub["Odds_Ratio"] < 1].nlargest(n_per_side, "abs_log_OR")
            top = pd.concat([top_neg.iloc[::-1], top_pos]).reset_index(drop=True)
            if len(top) == 0:
                ax.set_yticks([]); ax.set_xticks([0])
                ax.text(0, 0.5, "no significant features", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                if col == 0: ax.set_ylabel(stage, fontsize=10, fontweight="bold")
                ax.set_xlim(-0.8, 0.8); ax.axvline(0, color="k", lw=0.7)
                continue
            y_pos = np.arange(len(top))
            colors = [BAND_COL.get(b, "#777") for b in top["band"]]
            ax.barh(y_pos, top["log_OR"], color=colors, edgecolor="none", height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [f[:34] + "…" if len(f) > 34 else f for f in top["feature_name"]],
                fontsize=6.5,
            )
            if col == 0: ax.set_ylabel(stage, fontsize=10, fontweight="bold")
            ax.axvline(0, color="black", lw=0.7)
            ax.set_xlim(-0.8, 0.8)
            ax.grid(True, axis="x", alpha=0.3)
            if r == 0: ax.set_title(task_label, fontsize=11)
            if r == len(stages) - 1: ax.set_xlabel("log(Odds Ratio)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["#3eb55a", "#d92a2a", "#e6c33b", "#3a7eb8", "#cf57a8", "#777777"]]
    labels = ["σ (11-15 Hz)", "α (8-12 Hz)", "θ (4-8 Hz)", "δ (0.5-4 Hz)",
              "Slow (<0.5 Hz)", "General"]
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)

    fig.suptitle("Figure 2 — Top discriminative features by odds ratio (p<0.05), "
                 "grouped by sleep stage", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "figure2_top_discriminative_features.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "figure2_top_discriminative_features.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'figure2_top_discriminative_features.png'}")


# ---------------------------------------------------------------------------
# Confusion matrices (one per task, best model)
# ---------------------------------------------------------------------------

def make_confusion_matrices():
    print("\n[Confusion matrices]")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (task_key, task_label) in zip(axes, [("DM_v_CN", "DEM vs CN"),
                                                  ("MCI_v_CN", "MCI vs CN"),
                                                  ("DM_MCI_v_CN", "DEM/MCI vs CN")]):
        # Use best model per task: paper says SVM best for DM_v_CN, LR for the others
        best = {"DM_v_CN": "SVM", "MCI_v_CN": "LG", "DM_MCI_v_CN": "LG"}[task_key]
        res = load_pickle_scores(task_key, best)
        if res is None:
            ax.set_title(f"{task_label}\n(no scores)"); continue
        _, _, _, cm = res
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{task_label}\n({best.replace('LG', 'LR')}, balanced eval)")
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'confusion_matrices.png'}")


# ---------------------------------------------------------------------------

def make_figure1_age_distribution():
    """Figure 1B-style age histogram by group."""
    print("\n[Figure 1B] Age distribution by group")
    cohort = pd.read_excel(COHORT)
    ana = cohort[cohort["Predicted_Stage"].isin(["Dementia", "MCI", "No Dementia"])].copy()
    ana["group"] = ana["Predicted_Stage"].map({"Dementia": "DEM", "MCI": "MCI", "No Dementia": "CN"})
    ana["Age"] = pd.to_numeric(ana["Age"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(20, 100, 2)
    for grp in ["CN", "MCI", "DEM"]:
        sub = ana.loc[ana["group"] == grp, "Age"].dropna()
        ax.hist(sub, bins=bins, alpha=0.55, label=f"{grp} (n={len(sub):,})",
                color=COL[grp], edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of PSGs")
    ax.set_title("Figure 1B — Age distribution at PSG by diagnostic group")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "figure1b_age_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'figure1b_age_distribution.png'}")


def make_table_s4():
    """Per-classifier full metrics table."""
    print("\n[Table S4] Per-classifier full metrics")
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score)
    rows = []
    tasks = [("DM_v_CN", "DEM vs CN"), ("MCI_v_CN", "MCI vs CN"),
             ("DM_MCI_v_CN", "DEM/MCI vs CN")]
    for task_key, task_label in tasks:
        for mfile, mlabel in [("LG", "LR"), ("SVM", "SVM"), ("RF", "RF")]:
            res = load_pickle_scores(task_key, mfile)
            if res is None: continue
            y, prob, pred, _ = res
            rows.append({
                "Task": task_label, "Model": mlabel,
                "AUROC": round(roc_auc_score(y, prob), 3),
                "AUPRC": round(average_precision_score(y, prob), 3),
                "Accuracy": round(accuracy_score(y, pred), 3),
                "Precision": round(precision_score(y, pred, zero_division=0), 3),
                "Recall": round(recall_score(y, pred, zero_division=0), 3),
                "F1": round(f1_score(y, pred, zero_division=0), 3),
                "Cohen's K": round(cohen_kappa_score(y, pred), 3),
                "MCC": round(matthews_corrcoef(y, pred), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(FIGS / "table_S4_classifier_metrics.csv", index=False)
    print(df.to_string(index=False))


def make_supplementary_bh():
    """Supplementary: BH-corrected per-feature OR p-values."""
    print("\n[Supplementary] BH correction of per-feature p-values")
    from statsmodels.stats.multitest import multipletests
    rows = []
    for task_key, task_label in [("DM_v_CN", "DEM vs CN"), ("MCI_v_CN", "MCI vs CN")]:
        df = pd.read_csv(CODE / f"{task_key}_OR_df_V2.csv").dropna(subset=["p", "Odds_Ratio"])
        rej, p_adj, _, _ = multipletests(df["p"], alpha=0.05, method="fdr_bh")
        df["p_adj"] = p_adj
        df["sig_BH"] = rej
        n_sig_unc = int((df["p"] < 0.05).sum())
        n_sig_bh = int(rej.sum())
        n_pos_bh = int(((rej) & (df["Odds_Ratio"] > 1)).sum())
        n_neg_bh = int(((rej) & (df["Odds_Ratio"] < 1)).sum())
        rows.append({
            "Task": task_label,
            "Total features": len(df),
            "Significant (p<0.05 uncorrected)": n_sig_unc,
            "Significant (BH q<0.05)": n_sig_bh,
            "BH-significant positive (OR>1)": n_pos_bh,
            "BH-significant negative (OR<1)": n_neg_bh,
        })
        df.sort_values("p_adj").head(50).to_csv(
            FIGS / f"supp_top50_BH_{task_key}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(FIGS / "supp_BH_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaper says: 'For DEM vs CN, 499 features had significant ORs (177 DEM, 322 CN). "
          "For MCI vs CN, 386 features (95 MCI, 291 CN).'")


def main():
    make_table2()
    make_figure1_age_distribution()
    make_figure3()
    make_figure5()
    make_figure2()
    make_confusion_matrices()
    make_table_s4()
    make_supplementary_bh()
    print("\nAll outputs in", FIGS)
    for p in sorted(FIGS.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
