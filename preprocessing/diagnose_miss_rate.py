
from pathlib import Path
import csv

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

OUTPUT_FILE = (
    OUTPUT_DIR / "candidate_miss_diagnostics.csv"
)

TARGET_TRIPS = {
    "00000033",
    "00000054",
    "00000072"
}


def load_array(path, columns):
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)

    if data.shape[1] != columns:
        raise ValueError(f"Unexpected format: {path}")

    return data


def load_candidates():
    grouped = {}

    with CANDIDATE_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            trip_id = row["trip_id"]

            if trip_id not in TARGET_TRIPS:
                continue

            observation_id = int(row["observation_id"])
            arc_id = int(row["arc_id"])

            grouped.setdefault(
                trip_id, {}
            ).setdefault(
                observation_id, []
            ).append({
                "arc_id": arc_id,
                "distance_m": float(row["distance_m"]),
                "crs": row["crs"]
            })

    return grouped


def diagnose_trip(trip_id, candidates):
    folder = DATASET_ROOT / "train" / trip_id

    track = load_array(
        folder / f"{trip_id}.track", 3
    )

    nodes = load_array(
        folder / f"{trip_id}.nodes", 2
    )

    arcs = np.round(
        load_array(
            folder / f"{trip_id}.arcs", 2
        )
    ).astype(int)

    route = np.loadtxt(
        folder / f"{trip_id}.route",
        dtype=int
    ).reshape(-1)

    if not candidates:
        raise ValueError(
            f"No candidate observations for {trip_id}"
        )

    crs = next(
        iter(next(iter(candidates.values())))
    )["crs"]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    route_arc_ids = list(dict.fromkeys(route.tolist()))

    geometries = []
    valid_arc_ids = []

    for arc_id in route_arc_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            continue

        start_node, end_node = arcs[arc_id]

        if (
            start_node < 0
            or end_node < 0
            or start_node >= len(nodes)
            or end_node >= len(nodes)
        ):
            continue

        geometry = LineString([
            (node_x[start_node], node_y[start_node]),
            (node_x[end_node], node_y[end_node])
        ])

        if geometry.is_empty:
            continue

        geometries.append(geometry)
        valid_arc_ids.append(arc_id)

    if not geometries:
        raise ValueError(
            f"No valid route geometries for {trip_id}"
        )

    tree = STRtree(geometries)

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    candidate_distances = []
    reference_distances = []
    misses = 0
    missing_observations = 0

    for observation_id, (x, y) in enumerate(
        zip(gps_x, gps_y)
    ):
        point = Point(float(x), float(y))

        nearest_index = int(tree.nearest(point))
        reference_arc_id = valid_arc_ids[nearest_index]

        reference_distance = point.distance(
            geometries[nearest_index]
        )

        observation_candidates = candidates.get(
            observation_id, []
        )

        if observation_candidates:
            nearest_candidate_distance = min(
                item["distance_m"]
                for item in observation_candidates
            )

            candidate_arc_ids = {
                item["arc_id"]
                for item in observation_candidates
            }

            if reference_arc_id not in candidate_arc_ids:
                misses += 1
        else:
            nearest_candidate_distance = np.nan
            missing_observations += 1
            misses += 1

        candidate_distances.append(
            nearest_candidate_distance
        )

        reference_distances.append(
            reference_distance
        )

    candidate_distances = np.array(candidate_distances)
    reference_distances = np.array(reference_distances)

    valid_candidates = candidate_distances[
        np.isfinite(candidate_distances)
    ]

    total = len(track)

    return {
        "trip_id": trip_id,
        "observations": total,
        "misses": misses,
        "miss_rate_percent": round(
            misses / total * 100, 4
        ),
        "missing_candidate_observations":
            missing_observations,
        "mean_nearest_candidate_m": round(
            float(np.mean(valid_candidates)), 3
        ) if len(valid_candidates) else "",
        "p95_nearest_candidate_m": round(
            float(np.percentile(valid_candidates, 95)), 3
        ) if len(valid_candidates) else "",
        "mean_reference_distance_m": round(
            float(np.mean(reference_distances)), 3
        ),
        "p95_reference_distance_m": round(
            float(np.percentile(reference_distances, 95)), 3
        )
    }


def main():
    grouped = load_candidates()
    results = []

    for trip_id in sorted(TARGET_TRIPS):
        print(f"\nInvestigating trip {trip_id}...")

        if trip_id not in grouped:
            print("No candidate data found. Skipping.")
            continue

        result = diagnose_trip(
            trip_id, grouped[trip_id]
        )

        results.append(result)

        print("Observations:", result["observations"])
        print("Miss rate:", result["miss_rate_percent"], "%")
        print(
            "Mean nearest candidate distance:",
            result["mean_nearest_candidate_m"], "m"
        )
        print(
            "Mean reference-route distance:",
            result["mean_reference_distance_m"], "m"
        )
        print(
            "95th percentile reference distance:",
            result["p95_reference_distance_m"], "m"
        )

    if not results:
        raise RuntimeError("No trips were diagnosed.")

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nDiagnostic report saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()