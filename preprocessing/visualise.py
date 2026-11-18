
from pathlib import Path
import csv
from itertools import groupby

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "candidates_train.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "candidate_miss_rate_train.csv"

REFERENCE_DISTANCE_THRESHOLD_M = 100.0


def load_array(path, columns):
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)

    if data.shape[1] != columns:
        raise ValueError(f"Unexpected file format: {path}")

    return data


def load_trip_candidates(group):
    candidates = {}
    crs = None

    for row in group:
        if crs is None:
            crs = row["crs"]

        observation_id = int(row["observation_id"])
        arc_id = int(row["arc_id"])

        candidates.setdefault(
            observation_id, set()
        ).add(arc_id)

    return candidates, crs


def evaluate_trip(trip_folder, candidates, crs):
    trip_id = trip_folder.name

    track = load_array(
        trip_folder / f"{trip_id}.track", 3
    )

    nodes = load_array(
        trip_folder / f"{trip_id}.nodes", 2
    )

    arcs = np.round(
        load_array(
            trip_folder / f"{trip_id}.arcs", 2
        )
    ).astype(int)

    route = np.loadtxt(
        trip_folder / f"{trip_id}.route",
        dtype=int
    ).reshape(-1)

    if len(route) == 0:
        raise ValueError(f"Empty reference route: {trip_id}")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"Invalid route arc ID: {trip_id}")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid arc endpoint: {trip_id}")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    # Keep unique route arcs while preserving their first occurrence.
    route_arc_ids = list(dict.fromkeys(route.tolist()))

    valid_arc_ids = []
    geometries = []

    for arc_id in route_arc_ids:
        start_node, end_node = arcs[arc_id]

        geometry = LineString([
            (node_x[start_node], node_y[start_node]),
            (node_x[end_node], node_y[end_node])
        ])

        if geometry.is_empty or not geometry.is_valid:
            continue

        valid_arc_ids.append(arc_id)
        geometries.append(geometry)

    if not geometries:
        raise ValueError(
            f"No valid reference geometries: {trip_id}"
        )

    tree = STRtree(geometries)

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    total = len(track)
    reference_within_threshold = 0
    reference_too_far = 0
    eligible_misses = 0
    missing_candidate_observations = 0

    reference_distances = []
    candidate_distances = []

    for observation_id, (x, y) in enumerate(
        zip(gps_x, gps_y)
    ):
        point = Point(float(x), float(y))

        nearest_index = int(tree.nearest(point))

        reference_arc_id = valid_arc_ids[nearest_index]

        reference_distance = point.distance(
            geometries[nearest_index]
        )

        reference_distances.append(reference_distance)

        observation_candidates = candidates.get(
            observation_id, set()
        )

        if observation_candidates:
            candidate_geometries = []

            for arc_id in observation_candidates:
                if arc_id < 0 or arc_id >= len(arcs):
                    continue

                start_node, end_node = arcs[arc_id]

                candidate_geometries.append(
                    LineString([
                        (node_x[start_node], node_y[start_node]),
                        (node_x[end_node], node_y[end_node])
                    ])
                )

            if candidate_geometries:
                nearest_candidate_distance = min(
                    point.distance(geometry)
                    for geometry in candidate_geometries
                )
            else:
                nearest_candidate_distance = np.nan

        else:
            nearest_candidate_distance = np.nan
            missing_candidate_observations += 1

        candidate_distances.append(
            nearest_candidate_distance
        )

        # Separate reference-route alignment from candidate recall.
        if reference_distance > REFERENCE_DISTANCE_THRESHOLD_M:
            reference_too_far += 1
            continue

        reference_within_threshold += 1

        if reference_arc_id not in observation_candidates:
            eligible_misses += 1

    eligible_miss_rate = (
        eligible_misses / reference_within_threshold * 100
        if reference_within_threshold else 0
    )

    valid_candidate_distances = np.array(
        candidate_distances
    )
    valid_candidate_distances = (
        valid_candidate_distances[
            np.isfinite(valid_candidate_distances)
        ]
    )

    return {
        "trip_id": trip_id,
        "observations": total,
        "reference_within_100m": reference_within_threshold,
        "reference_over_100m": reference_too_far,
        "eligible_candidate_misses": eligible_misses,
        "eligible_miss_rate_percent": round(
            eligible_miss_rate, 4
        ),
        "missing_candidate_observations":
            missing_candidate_observations,
        "mean_reference_distance_m": round(
            float(np.mean(reference_distances)), 3
        ),
        "p95_reference_distance_m": round(
            float(np.percentile(reference_distances, 95)), 3
        ),
        "mean_nearest_candidate_distance_m": round(
            float(np.nanmean(valid_candidate_distances)), 3
        ) if len(valid_candidate_distances) else ""
    }


def main():
    train_folder = DATASET_ROOT / "train"

    trip_folders = sorted(
        item for item in train_folder.iterdir()
        if item.is_dir()
    )

    results = []

    with CANDIDATE_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        candidate_groups = iter(
            groupby(reader, key=lambda row: row["trip_id"])
        )

        current_group = next(candidate_groups, None)

        for index, trip_folder in enumerate(
            trip_folders, start=1
        ):
            trip_id = trip_folder.name

            if current_group is None:
                raise ValueError(
                    f"Missing candidates for {trip_id}"
                )

            group_trip_id, group_rows = current_group

            if group_trip_id != trip_id:
                raise ValueError(
                    "Candidate CSV trip order mismatch. "
                    f"Expected {trip_id}, found {group_trip_id}."
                )

            candidates, crs = load_trip_candidates(group_rows)

            result = evaluate_trip(
                trip_folder, candidates, crs
            )

            results.append(result)

            print(
                f"{index}/{len(trip_folders)} {trip_id}: "
                f"{result['eligible_miss_rate_percent']}% "
                "eligible miss rate"
            )

            current_group = next(
                candidate_groups, None
            )

    if current_group is not None:
        raise ValueError(
            "Candidate CSV contains trips not found "
            "in the training dataset."
        )

    total_observations = sum(
        row["observations"] for row in results
    )

    total_eligible = sum(
        row["reference_within_100m"] for row in results
    )

    total_far = sum(
        row["reference_over_100m"] for row in results
    )

    total_misses = sum(
        row["eligible_candidate_misses"] for row in results
    )

    total_missing_candidates = sum(
        row["missing_candidate_observations"]
        for row in results
    )

    overall_rate = (
        total_misses / total_eligible * 100
        if total_eligible else 0
    )

    results.append({
        "trip_id": "OVERALL",
        "observations": total_observations,
        "reference_within_100m": total_eligible,
        "reference_over_100m": total_far,
        "eligible_candidate_misses": total_misses,
        "eligible_miss_rate_percent": round(
            overall_rate, 4
        ),
        "missing_candidate_observations":
            total_missing_candidates,
        "mean_reference_distance_m": "",
        "p95_reference_distance_m": "",
        "mean_nearest_candidate_distance_m": ""
    })

    fieldnames = list(results[0].keys())

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nCANDIDATE EVALUATION SUMMARY")
    print("----------------------------")
    print("Training trips:", len(trip_folders))
    print("Total GPS observations:", total_observations)
    print("Reference within 100 m:", total_eligible)
    print("Reference over 100 m:", total_far)
    print("Eligible candidate misses:", total_misses)
    print("Missing candidate observations:", total_missing_candidates)
    print(f"Eligible miss rate: {overall_rate:.4f}%")
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()