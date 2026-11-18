"""Bayesian (Optuna TPE) tuning of LightGBM with trip-grouped 5-fold CV.

Usage (from the project root):
    python models\\tune_lightgbm.py results\\features_meghana.csv [n_trials]

Objective: minimise mean CV log-loss (smoother than F1 / top-1).
Outputs (in results\\):
    best_params_lightgbm.json   best hyper-parameters
    optuna_trials.csv           every trial (updated after each trial)
    lightgbm_tuned.joblib       final model trained on all train trips
    model_comparison.csv        gets a new row named "LightGBM (tuned)"
"""
import sys
import os
import json
import time

import numpy as np
import pandas as pd
import optuna
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from harness import run_experiment, SEED, RESULTS_DIR

FEATURES_PATH = sys.argv[1]
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

NON_FEATURES = {"trip_id", "point_idx", "split", "label", "arc_id"}

df = pd.read_csv(FEATURES_PATH)
feats = [c for c in df.columns if c not in NON_FEATURES]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "student_test"].reset_index(drop=True)

X, y, g = train[feats], train["label"], train["trip_id"]
n_splits = min(5, g.nunique())
folds = list(GroupKFold(n_splits=n_splits).split(X, y, g))

os.makedirs(RESULTS_DIR, exist_ok=True)
print("rows:", len(train), "| trips:", g.nunique(), "| folds:", n_splits)
print("features:", feats)
print("trials:", N_TRIALS)


def make_model(p):
    return LGBMClassifier(**p, random_state=SEED, verbose=-1)


def objective(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    losses = []
    for tr, va in folds:
        m = make_model(p)
        m.fit(X.iloc[tr], y.iloc[tr])
        losses.append(log_loss(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    trial.set_user_attr("cv_std", float(np.std(losses)))
    return float(np.mean(losses))


start = time.time()


def progress(study, trial):
    print(f"trial {trial.number + 1}/{N_TRIALS} | log-loss {trial.value:.4f} | "
          f"best {study.best_value:.4f} | {(time.time() - start) / 60:.1f} min",
          flush=True)
    study.trials_dataframe().to_csv(
        os.path.join(RESULTS_DIR, "optuna_trials.csv"), index=False)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress])

best = dict(study.best_params)
best["subsample_freq"] = 1
with open(os.path.join(RESULTS_DIR, "best_params_lightgbm.json"), "w") as f:
    json.dump(best, f, indent=2)
print("\nbest CV log-loss:", round(study.best_value, 4))
print("best params:", best)

# Score the tuned model with the same harness as the other models,
# so it lands in model_comparison.csv next to the untuned LightGBM.
final, res = run_experiment("LightGBM (tuned)", "boosting",
                            lambda: make_model(best), train, test, feats,
                            n_splits=n_splits)
joblib.dump(final, os.path.join(RESULTS_DIR, "lightgbm_tuned.joblib"))
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
print("saved model to results\\lightgbm_tuned.joblib")