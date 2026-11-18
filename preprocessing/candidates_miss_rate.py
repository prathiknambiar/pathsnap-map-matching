
from pathlib import Path
import csv
from itertools import groupby

import numpy as np
import folium
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


def load_array(path, columns):
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)

    if data.shape[1] != columns:
        raise ValueError(f"Unexpected file format: {path}")

    return data


def load_trip_candidates(group):
    candidates = {}

    for row in group:
        observation_id = int(row["observation_id"])
        arc_id = int(row["arc_id"])

        candidates.setdefault(
            observation_id, set()
        ).add(arc_id)

    return candidates


def evaluate_trip(trip_folder, candidates):
    trip_id = trip_folder.name

    track = load_array(
        trip_folder / f"{trip_id}.track", 3
    )

    nodes = load_array(
        trip_folder / f"{trip_id}.nodes", 2
    )

    arcs_raw = load_array(
        trip_folder / f"{trip_id}.arcs", 2
    )

    route = np.loadtxt(
        trip_folder / f"{trip_id}.route",
        dtype=int
    ).reshape(-1)

    arcs = np.round(arcs_raw).astype(int)

    if len(route) == 0:
        raise ValueError(f"Empty reference route: {trip_id}")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"Invalid route arc ID: {trip_id}")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid arc endpoint: {trip_id}")

    # Get the projection used during candidate generation.
    first_candidate = next(iter(candidates.values()), None)

    if not candidates:
        raise ValueError(
            f"No candidates found for trip {trip_id}"
        )

    # CRS is stored in the original candidate CSV.
    # It is read separately from the candidate rows below.
    crs = trip_crs[trip_id]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    # Project road nodes.
    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    # Keep unique route arc IDs while preserving route order.
    route_arc_ids = list(dict.fromkeys(route.tolist()))

    route_geometries = []

    for arc_id in route_arc_ids:
        start_node, end_node = arcs[arc_id]

        geometry = LineString([
            (node_x[start_node], node_y[start_node]),
            (node_x[end_node], node_y[end_node])
        ])

        if geometry.is_empty or not geometry.is_valid:
            continue

        route_geometries.append((arc_id, geometry))

    if not route_geometries:
        raise ValueError(
            f"No valid reference geometries: {trip_id}"
        )

    valid_arc_ids = [
        item[0] for item in route_geometries
    ]

    geometries = [
        item[1] for item in route_geometries
    ]

    tree = STRtree(geometries)

    # Project GPS observations.
    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    total = 0
    misses = 0
    missing_candidate_observations = 0
    reference_distances = []

    for observation_id, (x, y) in enumerate(
        zip(gps_x, gps_y)
    ):
        point = Point(float(x), float(y))

        nearest_index = int(tree.nearest(point))

        reference_arc_id = valid_arc_ids[nearest_index]

        distance = point.distance(
            geometries[nearest_index]
        )

        reference_distances.append(distance)

        candidate_arc_ids = candidates.get(
            observation_id, set()
        )

        if not candidate_arc_ids:
            missing_candidate_observations += 1

        if reference_arc_id not in candidate_arc_ids:
            misses += 1

        total += 1

    miss_rate = (
        misses / total * 100 if total else 0
    )

    return {
        "trip_id": trip_id,
        "observations": total,
        "misses": misses,
        "missing_candidate_observations":
            missing_candidate_observations,
        "miss_rate_percent": round(miss_rate, 4),
        "mean_reference_distance_m": round(
            float(np.mean(reference_distances)), 3
        ),
        "p95_reference_distance_m": round(
            float(np.percentile(reference_distances, 95)), 3
        )
    }


def main():
    global trip_crs
    trip_crs = {}

    # Read CRS values for each trip.
    with CANDIDATE_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            trip_crs[row["trip_id"]] = row["crs"]

    train_folder = DATASET_ROOT / "train"

    trip_folders = sorted(
        item for item in train_folder.iterdir()
        if item.is_dir()
    )

    results = []

    # Stream the candidate CSV by trip to avoid loading
    # the entire 1.36-million-row file into memory.
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
                    "Candidate CSV is not grouped in the "
                    f"expected trip order. Expected {trip_id}, "
                    f"found {group_trip_id}."
                )

            candidates = load_trip_candidates(group_rows)

            result = evaluate_trip(
                trip_folder, candidates
            )

            results.append(result)

            print(
                f"{index}/{len(trip_folders)} "
                f"{trip_id}: "
                f"{result['miss_rate_percent']}% miss rate"
            )

            current_group = next(
                candidate_groups, None
            )

    total_observations = sum(
        row["observations"] for row in results
    )

    total_misses = sum(
        row["misses"] for row in results
    )

    overall_rate = (
        total_misses / total_observations * 100
        if total_observations else 0
    )

    results.append({
        "trip_id": "OVERALL",
        "observations": total_observations,
        "misses": total_misses,
        "missing_candidate_observations": sum(
            row["missing_candidate_observations"]
            for row in results
        ),
        "miss_rate_percent": round(overall_rate, 4),
        "mean_reference_distance_m": "",
        "p95_reference_distance_m": ""
    })

    fieldnames = [
        "trip_id",
        "observations",
        "misses",
        "missing_candidate_observations",
        "miss_rate_percent",
        "mean_reference_distance_m",
        "p95_reference_distance_m"
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nCANDIDATE MISS-RATE SUMMARY")
    print("---------------------------")
    print("Training trips:", len(trip_folders))
    print("GPS observations:", total_observations)
    print("Missing reference candidates:", total_misses)
    print(f"Overall miss rate: {overall_rate:.4f}%")
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()