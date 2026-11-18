
from pathlib import Path
import csv
import sys
import math

# Original dataset location
DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_FILES = [".track", ".nodes", ".arcs", ".route"]


def read_rows(file_path):
    """Read whitespace-separated values from a dataset file."""
    rows = []
    malformed = 0

    if not file_path.exists():
        return rows, malformed

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            values = line.split()

            try:
                rows.append([float(value) for value in values])
            except ValueError:
                malformed += 1

    return rows, malformed


def inspect_trip(trip_folder, partition):
    """Inspect all files belonging to one trip."""

    trip_id = trip_folder.name

    result = {
        "partition": partition,
        "trip_id": trip_id,
    }

    for extension in EXPECTED_FILES:
        file_path = trip_folder / f"{trip_id}{extension}"

        result[f"{extension[1:]}_exists"] = file_path.exists()

        result[f"{extension[1:]}_size_kb"] = (
            round(file_path.stat().st_size / 1024, 2)
            if file_path.exists() else 0
        )

    # Inspect GPS observations
    track_path = trip_folder / f"{trip_id}.track"
    gps_rows, malformed = read_rows(track_path)

    result["gps_observations"] = len(gps_rows)
    result["malformed_track_rows"] = malformed

    missing_values = 0
    invalid_coordinates = 0
    timestamps = []

    for row in gps_rows:
        if len(row) != 3:
            missing_values += 1
            continue

        longitude, latitude, timestamp = row

        if not all(math.isfinite(v) for v in row):
            missing_values += 1
            continue

        if not (
            -180 <= longitude <= 180
            and -90 <= latitude <= 90
        ):
            invalid_coordinates += 1

        timestamps.append(timestamp)

    result["missing_or_invalid_rows"] = missing_values
    result["invalid_coordinates"] = invalid_coordinates

    # Inspect timestamp ordering and gaps
    timestamps_sorted = sorted(timestamps)

    result["timestamps_ordered"] = (
        timestamps == timestamps_sorted
    )

    gaps = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    result["time_gaps_over_1s"] = sum(
        gap > 1 for gap in gaps
    )

    result["max_time_gap_seconds"] = (
        round(max(gaps), 2) if gaps else 0
    )

    result["duplicate_timestamps"] = (
        len(timestamps) - len(set(timestamps))
    )

    # Inspect road network files
    nodes, node_errors = read_rows(
        trip_folder / f"{trip_id}.nodes"
    )

    arcs, arc_errors = read_rows(
        trip_folder / f"{trip_id}.arcs"
    )

    route, route_errors = read_rows(
        trip_folder / f"{trip_id}.route"
    )

    result["road_nodes"] = len(nodes)
    result["road_arcs"] = len(arcs)
    result["reference_route_segments"] = len(route)

    result["malformed_nodes"] = node_errors
    result["malformed_arcs"] = arc_errors
    result["malformed_route"] = route_errors

    return result


def inspect_partition(partition_path, partition_name):
    """Inspect every trip in a partition."""

    results = []

    trip_folders = sorted(
        [p for p in partition_path.iterdir() if p.is_dir()]
    )

    print(f"\nInspecting {partition_name}...")

    for trip_folder in trip_folders:
        result = inspect_trip(trip_folder, partition_name)
        results.append(result)

    print(f"Trips found: {len(results)}")

    return results


def main():
    if not DATASET_ROOT.exists():
        print("ERROR: Dataset folder not found!")
        print(DATASET_ROOT)
        sys.exit(1)

    all_results = []

    for partition in ["train", "student_test"]:
        partition_path = DATASET_ROOT / partition

        if not partition_path.exists():
            print(f"WARNING: {partition} folder not found")
            continue

        results = inspect_partition(
            partition_path, partition
        )

        all_results.extend(results)

    if not all_results:
        print("No trips found. Check the dataset structure.")
        return

    # Save inventory CSV
    output_file = OUTPUT_DIR / "dataset_inventory.csv"
    fieldnames = list(all_results[0].keys())

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Dataset summary
    print("\n========== DATASET SUMMARY ==========")

    for partition in ["train", "student_test"]:
        trips = [
            r for r in all_results
            if r["partition"] == partition
        ]

        print(f"\n{partition.upper()}")
        print(f"Number of trips: {len(trips)}")

        print(
            "Total GPS observations:",
            sum(r["gps_observations"] for r in trips)
        )

        print(
            "Total road nodes:",
            sum(r["road_nodes"] for r in trips)
        )

        print(
            "Total road arcs:",
            sum(r["road_arcs"] for r in trips)
        )

        print(
            "Trips with missing files:",
            sum(
                not all(
                    r[f"{ext[1:]}_exists"]
                    for ext in EXPECTED_FILES
                )
                for r in trips
            )
        )

    # Data quality summary
    print("\n========== DATA QUALITY SUMMARY ==========")

    for partition in ["train", "student_test"]:
        trips = [
            r for r in all_results
            if r["partition"] == partition
        ]

        print(f"\n{partition.upper()}")

        print(
            "Rows with missing/non-finite values:",
            sum(r["missing_or_invalid_rows"] for r in trips)
        )

        print(
            "Invalid coordinates:",
            sum(r["invalid_coordinates"] for r in trips)
        )

        print(
            "Trips with unordered timestamps:",
            sum(not r["timestamps_ordered"] for r in trips)
        )

        print(
            "Total gaps over 1 second:",
            sum(r["time_gaps_over_1s"] for r in trips)
        )

        print(
            "Trips with duplicate timestamps:",
            sum(
                r["duplicate_timestamps"] > 0
                for r in trips
            )
        )

        print(
            "Largest observed time gap:",
            max(r["max_time_gap_seconds"] for r in trips),
            "seconds"
        )

    print("\nInventory saved to:")
    print(output_file)


if __name__ == "__main__":
    main()