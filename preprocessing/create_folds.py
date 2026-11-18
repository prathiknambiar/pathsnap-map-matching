
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


def main():
    train_folder = DATASET_ROOT / "train"

    trip_ids = sorted(
        folder.name
        for folder in train_folder.iterdir()
        if folder.is_dir()
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rows = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(trip_ids), start=1
    ):
        train_ids = [trip_ids[i] for i in train_idx]
        val_ids = [trip_ids[i] for i in val_idx]

        assert set(train_ids).isdisjoint(set(val_ids))

        for trip_id in train_ids:
            rows.append({
                "trip_id": trip_id,
                "fold": fold,
                "role": "train"
            })

        for trip_id in val_ids:
            rows.append({
                "trip_id": trip_id,
                "fold": fold,
                "role": "validation"
            })

        print(
            f"Fold {fold}: "
            f"{len(train_ids)} training trips, "
            f"{len(val_ids)} validation trips"
        )

    output = OUTPUT_DIR / "trip_grouped_folds.csv"

    pd.DataFrame(rows).to_csv(output, index=False)

    print("\nCross-validation folds created successfully.")
    print("Total trips:", len(trip_ids))
    print("Number of folds:", N_SPLITS)
    print("Saved:", output)


if __name__ == "__main__":
    main()