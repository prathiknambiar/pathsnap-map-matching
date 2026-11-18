import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from harness import RESULTS_DIR, SEED

N_TRIALS = 30


def make_lgbm(params):
    return LGBMClassifier(random_state=SEED, verbose=-1, **params)


def objective_factory(train, feats, n_splits=5):
    X, y, g = train[feats], train["label"], train["trip_id"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        scores = []
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, g):
            m = make_lgbm(params)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = (m.predict_proba(X.iloc[va])[:, 1] >= 0.5).astype(int)
            scores.append(f1_score(y.iloc[va], pred, average="macro"))
        return float(np.mean(scores))

    return objective


def run_tuning(train, feats, n_trials=N_TRIALS, out_name="optuna_trials.csv"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_factory(train, feats), n_trials=n_trials)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(RESULTS_DIR, out_name), index=False)
    return study