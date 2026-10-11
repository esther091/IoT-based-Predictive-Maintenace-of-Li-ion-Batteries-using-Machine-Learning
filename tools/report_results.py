"""
report_results.py
---------------------------------------------------------------
Regenerates every number and figure in Chapter 4 of the project
report from the saved models in ml/models/, so the report always
matches the code.

Writes to report_results/ (project root):
  results.md                          all tables, ready to copy
  fig_4_3_soh_pred_vs_actual.png      Figure 4.3
  fig_4_4_rul_pred_vs_actual.png      Figure 4.4
  fig_4_5_model_r2.png                Figure 4.5
  fig_4_6_feature_importance.png      Figure 4.6

USAGE (from the project root, venv active, after ml/train_models.py):
  python tools/report_results.py
---------------------------------------------------------------
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ML_DIR = os.path.join(ROOT, "ml")
OUT_DIR = os.path.join(ROOT, "report_results")
sys.path.insert(0, ML_DIR)
os.chdir(ML_DIR)  # train_models uses paths relative to ml/

from utils import mean_absolute_percentage_error
from train_models import (
    BATTERIES, TRAIN_BATTERIES, TEST_BATTERY, FEATURE_COLS,
    extract_discharge_cycles, compute_targets, remove_outliers, split_data,
)

COLORS = {"Random Forest": "#2A7BD6", "XGBoost": "#E8683A", "Ensemble": "#1DB07E"}
FEATURE_LABELS = {"mean_voltage": "Mean voltage", "mean_current": "Mean current",
                  "mean_power": "Mean power", "energy_wh": "Energy delivered"}


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p),
            "MAPE": mean_absolute_percentage_error(y, p), "R2": r2_score(y, p)}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    md = []

    # ---------------- data (exactly as training) ----------------
    rows = []
    for b in BATTERIES:
        rows.extend(extract_discharge_cycles(b))
    raw = compute_targets(pd.DataFrame(rows))
    df = remove_outliers(raw.copy())
    train_df, val_df, test_df = split_data(df)

    m = {n: joblib.load(os.path.join("models", f"{n}.pkl"))
         for n in ("scaler", "rf_soh", "rf_rul", "xgb_soh", "xgb_rul")}
    X_train = m["scaler"].transform(train_df[FEATURE_COLS])
    X_test = m["scaler"].transform(test_df[FEATURE_COLS])

    # ---------------- Table 4.3 ----------------
    md.append("## Table 4.3: Discharge Cycle Dataset Preparation Summary\n")
    md.append("| Battery | Cycles extracted | Cycles retained | Initial SOH (%) | Minimum SOH (%) | End-of-life cycle |")
    md.append("|---|---|---|---|---|---|")
    for b in BATTERIES:
        g, kept = raw[raw.battery == b], df[df.battery == b]
        eol = g[g.soh < 70]
        eol_txt = str(int(eol.cycle_number.iloc[0])) if len(eol) else f"Not reached ({len(g)})"
        md.append(f"| {b} | {len(g)} | {len(kept)} | {g.soh.iloc[0]:.1f} | {g.soh.min():.1f} | {eol_txt} |")
    md.append(f"| Total | {len(raw)} | {len(df)} | | | |\n")

    # ---------------- scaler ranges (Section 3.2.4) ----------------
    md.append("## Training feature ranges (Min-Max scaler, Section 3.2.4)\n")
    md.append("| Feature | Min | Max |")
    md.append("|---|---|---|")
    for f, lo, hi in zip(FEATURE_COLS, m["scaler"].data_min_, m["scaler"].data_max_):
        md.append(f"| {FEATURE_LABELS[f]} | {lo:.3f} | {hi:.3f} |")
    md.append("")

    # ---------------- Table 4.4: CV scores of the selected configurations ----------------
    md.append("## Table 4.4: Selected Hyperparameters and Mean Cross-Validation Scores\n")
    md.append("| Model | Target | Selected hyperparameters | Mean CV R² | Mean CV MAE |")
    md.append("|---|---|---|---|---|")
    targets = {"soh": train_df["soh"].values, "rul": train_df["rul"].values}
    for name, key, keys in [
        ("Random Forest", "rf", ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"]),
        ("XGBoost", "xgb", ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]),
    ]:
        for tgt in ("soh", "rul"):
            est = m[f"{key}_{tgt}"]
            params = est.get_params()
            p_txt = ", ".join(f"{k} = {params[k]}" for k in keys)
            cv = cross_validate(clone(est), X_train, targets[tgt], cv=5,
                                scoring=("r2", "neg_mean_absolute_error"))
            unit = "%" if tgt == "soh" else " days"
            md.append(f"| {name} | {tgt.upper()} | {p_txt} | {cv['test_r2'].mean():.3f} | "
                      f"{-cv['test_neg_mean_absolute_error'].mean():.2f}{unit} |")
    md.append("")

    # ---------------- Tables 4.5 / 4.6 ----------------
    preds, results = {}, {}
    for tgt in ("soh", "rul"):
        rf = m[f"rf_{tgt}"].predict(X_test)
        xgb = m[f"xgb_{tgt}"].predict(X_test)
        preds[tgt] = {"Random Forest": rf, "XGBoost": xgb, "Ensemble": (rf + xgb) / 2}
        results[tgt] = {k: metrics(test_df[tgt].values, v) for k, v in preds[tgt].items()}

    for tgt, num, title, unit in [("soh", "4.5", "State of Health", "%"), ("rul", "4.6", "Remaining Useful Life", "days")]:
        md.append(f"## Table {num}: {title} Prediction Performance on Test Battery {TEST_BATTERY}\n")
        md.append(f"| Model | MAE ({unit}) | RMSE ({unit}) | MAPE (%) | R² |")
        md.append("|---|---|---|---|---|")
        for k, r in results[tgt].items():
            md.append(f"| {k} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['MAPE']:.2f} | {r['R2']:.4f} |")
        md.append("")

    # ---------------- Feature importance (Section 4.2.6) ----------------
    imp = {tgt: m[f"rf_{tgt}"].feature_importances_ for tgt in ("soh", "rul")}
    md.append("## Section 4.2.6: Random Forest Feature Importance\n")
    md.append("| Feature | RF-SOH | RF-RUL |")
    md.append("|---|---|---|")
    for i, f in enumerate(FEATURE_COLS):
        md.append(f"| {FEATURE_LABELS[f]} | {imp['soh'][i]:.2f} | {imp['rul'][i]:.2f} |")
    md.append("")

    with open(os.path.join(OUT_DIR, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("# Report results (generated by tools/report_results.py)\n\n" + "\n".join(md))

    # ---------------- Figures ----------------
    cyc = test_df["cycle_number"].values
    for tgt, fname, ylabel in [("soh", "fig_4_3_soh_pred_vs_actual.png", "State of Health (%)"),
                               ("rul", "fig_4_4_rul_pred_vs_actual.png", "Remaining Useful Life (days)")]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
        ax.plot(cyc, test_df[tgt], color="black", linewidth=2.2, label="Actual")
        for k, v in preds[tgt].items():
            ax.plot(cyc, v, color=COLORS[k], linewidth=1.4, label=k,
                    linestyle="--" if k == "Ensemble" else "-")
        ax.set_xlabel(f"Discharge cycle number (battery {TEST_BATTERY})")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=4, frameon=False, loc="upper right")
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    width, x = 0.26, np.arange(2)
    for j, k in enumerate(("Random Forest", "XGBoost", "Ensemble")):
        vals = [results["soh"][k]["R2"], results["rul"][k]["R2"]]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=COLORS[k], label=k)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, ["SOH prediction", "RUL prediction"])
    ax.set_ylabel("Coefficient of determination (R²)")
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_4_5_model_r2.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    order = np.argsort(imp["soh"])
    y = np.arange(len(FEATURE_COLS))
    for off, tgt, color, label in [(0.2, "soh", "#2A7BD6", "RF – SOH model"), (-0.2, "rul", "#E8683A", "RF – RUL model")]:
        bars = ax.barh(y + off, imp[tgt][order], 0.38, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_yticks(y, [FEATURE_LABELS[FEATURE_COLS[i]] for i in order])
    ax.set_xlabel("Relative importance")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_4_6_feature_importance.png"))
    plt.close(fig)

    print(f"\nWrote results.md and 4 figures to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
