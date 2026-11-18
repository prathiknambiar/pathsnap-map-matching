import sys
import pandas as pd
from harness import run_experiment
from model_configs import MODELS

# Usage: python models\run_all.py path\to\features.csv
FEATURES_PATH = sys.argv[1]

# Edit this set to match the data contract with the feature owner
NON_FEATURES = {"trip_id", "point_idx", "split", "label", "arc_id"}

df = pd.read_csv(FEATURES_PATH)   # use pd.read_parquet if they send parquet
feats = [c for c in df.columns if c not in NON_FEATURES]
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "student_test"].reset_index(drop=True)

print("rows:", len(train), len(test), "| trips:", train.trip_id.nunique(), test.trip_id.nunique())
print("features:", feats)

for name, (family, make) in MODELS.items():
    print("running", name)
    _, res = run_experiment(name, family, make, train, test, feats)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})