import numpy as np
import pandas as pd
from tune import run_tuning

rng = np.random.default_rng(42)
rows = []
for trip in range(16):
    for p in range(100):
        true = rng.integers(0, 5)
        for c in range(5):
            rows.append({"trip_id": trip, "point_idx": p, "label": int(c == true),
                         **{f"f{i}": rng.normal() for i in range(1, 6)}})
train = pd.DataFrame(rows)
feats = [f"f{i}" for i in range(1, 6)]

study = run_tuning(train, feats, n_trials=3, out_name="dummy_optuna.csv")
print(study.best_params, study.best_value)