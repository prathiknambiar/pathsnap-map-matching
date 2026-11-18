"""Builds the report assets for sections 3 and 4.

Usage (from the project root):
    python models\\make_report_assets.py results\\features_meghana.csv [ExtraModel ...]

Default confusion matrices: LogReg, LightGBM, LightGBM (tuned), from trip-grouped
5-fold out-of-fold predictions on the train set (test labels are hidden).
Extra names (e.g. RandomForest MLP) must match keys in model_configs.MODELS;
they are slow (RandomForest ~10+ min, MLP ~20 min), so they are optional.

Writes to results\\ and results\\plots\\:
    comparison_table.csv
    plots\\f1_comparison.png, logloss_comparison.png, train_time.png,
    plots\\tuned_vs_untuned.png, optuna_history.png, confusion_matrices.png
    confusion_counts.csv
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier

from harness import SEED, RESULTS_DIR
from model_configs import MODELS

FEATURES_PATH = sys.argv[1]
EXTRA = sys.argv[2:]

RESULTS_DIR = os.path.normpath(RESULTS_DIR)
PLOTS = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- table
cmp = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
cmp = cmp.drop_duplicates("model", keep="last")
order = ["LogReg", "KNN", "RandomForest", "MLP", "LightGBM", "LightGBM (tuned)"]
cmp["_o"] = cmp["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
cmp = cmp.sort_values("_o").reset_index(drop=True)

cols = ["model", "family", "cv_macro_f1", "cv_std", "cv_precision", "cv_recall",
        "cv_log_loss", "cv_top1", "train_time"]
cols = [c for c in cols if c in cmp.columns]
table = cmp[cols].copy()
table["note"] = ""
table.loc[table["model"] == "KNN", "note"] = "trained/CV on 20 of 80 train trips"
table = table.round(4)
table.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
print(table.to_string(index=False))

labels = [m + "*" if m == "KNN" else m for m in cmp["model"]]
colors = ["#4C72B0" if m != "LightGBM (tuned)" else "#C44E52" for m in cmp["model"]]
foot = "*KNN trained and cross-validated on 20 of 80 train trips (not directly comparable)"


def bar(metric, title, ylabel, fname, err=None, log=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, cmp[metric], yerr=None if err is None else cmp[err],
           color=colors, capsize=4)
    for i, v in enumerate(cmp[metric]):
        ax.text(i, v, f"{v:.3g}" if log else f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.text(0.01, 0.01, foot, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(PLOTS, fname), dpi=200)
    plt.close(fig)


bar("cv_macro_f1", "CV macro-F1 by model (error bars = fold std)", "macro-F1",
    "f1_comparison.png", err="cv_std")
bar("cv_log_loss", "CV log-loss by model (lower is better)", "log-loss",
    "logloss_comparison.png")
bar("train_time", "Final-fit train time by model (log scale)", "seconds",
    "train_time.png", log=True)

# ---------------------------------------------------------- tuned vs untuned
pair = cmp[cmp["model"].isin(["LightGBM", "LightGBM (tuned)"])].set_index("model")
if len(pair) == 2:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, (m, t) in zip(axes, [("cv_macro_f1", "CV macro-F1"),
                                 ("cv_log_loss", "CV log-loss (lower better)"),
                                 ("cv_top1", "CV top-1")]):
        vals = [pair.loc["LightGBM", m], pair.loc["LightGBM (tuned)", m]]
        ax.plot(["untuned", "tuned"], vals, "o-", color="#C44E52")
        for x, v in zip(["untuned", "tuned"], vals):
            ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.margins(y=0.3)
    fig.suptitle("LightGBM: untuned vs Optuna-tuned (axes not zero-based)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "tuned_vs_untuned.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------- Optuna history
tp = os.path.join(RESULTS_DIR, "optuna_trials.csv")
if os.path.exists(tp):
    tr = pd.read_csv(tp)
    x = tr["number"] + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, tr["value"], "o", alpha=0.6, label="trial")
    ax.plot(x, tr["value"].cummin(), "-", color="#C44E52", label="best so far")
    ax.axhline(float(pair.loc["LightGBM", "cv_log_loss"]) if len(pair) == 2 else np.nan,
               ls="--", color="gray", label="untuned LightGBM")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV log-loss")
    ax.set_title("Optuna search history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "optuna_history.png"), dpi=200)
    plt.close(fig)

# ------------------------------------------------------ confusion matrices
NON_FEATURES = {"trip_id", "point_idx", "split", "label", "arc_id"}
df = pd.read_csv(FEATURES_PATH)
feats = [c for c in df.columns if c not in NON_FEATURES]
train = df[df["split"] == "train"].reset_index(drop=True)
X, y, g = train[feats], train["label"].astype(int), train["trip_id"]
folds = list(GroupKFold(n_splits=min(5, g.nunique())).split(X, y, g))

with open(os.path.join(RESULTS_DIR, "best_params_lightgbm.json")) as f:
    best = json.load(f)

to_run = {
    "LogReg": MODELS["LogReg"][1],
    "LightGBM": MODELS["LightGBM"][1],
    "LightGBM (tuned)": lambda: LGBMClassifier(**best, random_state=SEED, verbose=-1),
}
for name in EXTRA:
    to_run[name] = MODELS[name][1]

rows, cms = [], {}
for name, make in to_run.items():
    print("out-of-fold predictions:", name, flush=True)
    oof = np.zeros(len(train))
    for tr_i, va_i in folds:
        m = make()
        m.fit(X.iloc[tr_i], y.iloc[tr_i])
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    cm = confusion_matrix(y, (oof >= 0.5).astype(int), labels=[0, 1])
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "confusion_counts.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))

n = len(cms)
fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
for ax, (name, cm) in zip(axes[0], cms.items()):
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(name, fontsize=10)
fig.suptitle("Confusion matrices (out-of-fold, train set; % of each true class)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "confusion_matrices.png"), dpi=200)
plt.close(fig)

print("\nsaved plots to", PLOTS)
