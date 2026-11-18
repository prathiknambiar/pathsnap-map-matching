
import pandas as pd

files = [
    "data/processed/candidates_train.csv",
    "data/processed/candidates_student_test.csv"
]

for file in files:
    df = pd.read_csv(file)

    groups = df.groupby(
        ["trip_id", "observation_id"]
    ).size()

    print("\nFILE:", file)
    print("Candidate rows:", len(df))
    print("Observations with candidates:", len(groups))
    print("Minimum candidates:", groups.min())
    print("Maximum candidates:", groups.max())
    print("Average candidates:", round(groups.mean(), 2))
    print("Maximum distance (m):", round(df["distance_m"].max(), 2))
    print(
        "Candidates beyond 100m:",
        int((df["distance_m"] > 100).sum())
    )