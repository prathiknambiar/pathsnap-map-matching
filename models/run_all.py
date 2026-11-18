import sys
import pandas as pd
from harness import run_experiment
from model_configs import MODELS

# Usage:
#   python models\run_all.py path\to\features.csv            (all models)
#   python models\run_all.py path\to\features.csv MLP LightGBM
FEATURES_PATH = sys.argv[1]
ONLY = sys.argv[2:]

NON_FEATURES = {"trip_id", "point_idx", "split", "label", "arc_id"}

df = pd.read_csv(FEATURES_PATH)
feats = [c for c in df.columns if c not in NON_FEATURES]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "student_test"].reset_index(drop=True)

n_train_trips = train.trip_id.nunique()
n_splits = min(5, n_train_trips)

print("rows:", len(train), len(test), "| trips:", n_train_trips, test.trip_id.nunique())
print("features:", feats)
print("cv folds:", n_splits)

for name, (family, make) in MODELS.items():
    if ONLY and name not in ONLY:
        continue
    print("running", name, flush=True)
    _, res = run_experiment(name, family, make, train, test, feats, n_splits=n_splits)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)