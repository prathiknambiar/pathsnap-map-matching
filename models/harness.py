import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEFAULT_LOG = os.path.join(RESULTS_DIR, "model_comparison.csv")
METRICS = ["macro_f1", "precision", "recall", "log_loss", "top1"]


def top1_accuracy(meta, proba):
    """Per GPS point, take the highest-scoring candidate and check if it is the true one."""
    d = meta[["trip_id", "point_idx", "label"]].copy()
    d["p"] = np.asarray(proba)
    best = d.loc[d.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    return best["label"].mean()


def evaluate(model, X, y, meta):
    t = time.time()
    proba = model.predict_proba(X)[:, 1]
    infer = time.time() - t
    pred = (proba >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, proba),
        "top1": top1_accuracy(meta, proba),
        "infer_time": infer,
    }


def run_experiment(name, family, make_model, train, test, feats,
                   n_splits=5, log_path=DEFAULT_LOG):
    X, y, g = train[feats], train["label"].astype(int), train["trip_id"]

    folds = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, g):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        folds.append(evaluate(m, X.iloc[va], y.iloc[va], train.iloc[va]))
    cv = pd.DataFrame(folds)

    t = time.time()
    final = make_model()
    final.fit(X, y)
    train_time = time.time() - t

    if test["label"].notna().all():
        res = evaluate(final, test[feats], test["label"].astype(int), test)
    else:
        # test labels are hidden: no test metrics, only the timing
        t = time.time()
        final.predict_proba(test[feats])
        res = {k: np.nan for k in METRICS}
        res["infer_time"] = time.time() - t

    res.update({
        "model": name, "family": family,
        "cv_macro_f1": float(cv["macro_f1"].mean()),
        "cv_std": float(np.std(cv["macro_f1"])),
        "cv_precision": float(cv["precision"].mean()),
        "cv_recall": float(cv["recall"].mean()),
        "cv_log_loss": float(cv["log_loss"].mean()),
        "cv_top1": float(cv["top1"].mean()),
        "train_time": train_time, "seed": SEED,
    })

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame([res]).to_csv(log_path, mode="a",
                               header=not os.path.exists(log_path), index=False)
    return final, res