import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from harness import run_experiment, RESULTS_DIR

rng = np.random.default_rng(42)
rows = []
for trip in range(20):
    split = "train" if trip < 16 else "student_test"
    for p in range(100):
        true = rng.integers(0, 5)
        for c in range(5):
            rows.append({
                "trip_id": trip, "point_idx": p, "split": split,
                "label": int(c == true),
                "f1": rng.normal(), "f2": rng.normal(), "f3": rng.normal(),
                "f4": rng.normal(), "f5": rng.normal(),
            })

df = pd.DataFrame(rows)
train = df[df.split == "train"].reset_index(drop=True)
test = df[df.split == "student_test"].reset_index(drop=True)
feats = ["f1", "f2", "f3", "f4", "f5"]

make = lambda: Pipeline([
    ("sc", StandardScaler()),
    ("m", LogisticRegression(max_iter=500, random_state=42)),
])

model, res = run_experiment("LogReg_dummy", "linear", make, train, test, feats,
                            log_path=os.path.join(RESULTS_DIR, "dummy_test.csv"))
print(res)