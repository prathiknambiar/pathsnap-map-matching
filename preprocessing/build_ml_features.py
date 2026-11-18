from pathlib import Path
import csv
import sys

import numpy as np
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS = PROJECT_ROOT / "experiments"

INPUTS = {
    "train": PROCESSED / "candidates_train.csv",
    "student_test": PROCESSED / "candidates_student_test.csv",
}

OUTPUTS = {
    "train": PROCESSED / "features_train.csv",
    "student_test": PROCESSED / "features_student_test.csv",
}

FOLD_FILE = EXPERIMENTS / "trip_grouped_folds.csv"
REPORT_FILE = EXPERIMENTS / "label_alignment_report.csv"

FEATURES = [
    "longitude",
    "latitude",
    "timestamp",
    "distance_m",
    "gps_heading",
    "road_heading",
    "heading_difference",
    "gps_speed_mps",
    "projected_x",
    "projected_y",
    "crs",
    "fallback_candidate",
]

COLUMNS = [
    "trip_id",
    "point_idx",
    "arc_id",
    "label",
    "arc_len",
    "split",
] + FEATURES

MAX_REVIEW_DISTANCE = 250.0


def load_array(path, columns):
    data = np.atleast_2d(np.loadtxt(path, dtype=float))

    if data.shape[1] != columns:
        raise ValueError(f"Invalid columns in {path}")

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {path}")

    return data


def get_transformer(nodes):
    lon = float(np.median(nodes[:, 0]))
    lat = float(np.median(nodes[:, 1]))

    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone

    crs = f"EPSG:{epsg}"

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    return transformer, crs


def align_track_to_route(track, nodes, arcs, route, transformer):
    """
    Align GPS observations to the ordered route using
    monotonic dynamic programming.

    The route position can stay the same or advance.
    It cannot move backwards.
    """

    nx, ny = transformer.transform(
        nodes[:, 0],
        nodes[:, 1],
    )

    nx = np.asarray(nx)
    ny = np.asarray(ny)

    start = arcs[route, 0]
    end = arcs[route, 1]

    ax = nx[start]
    ay = ny[start]
    bx = nx[end]
    by = ny[end]

    gx, gy = transformer.transform(
        track[:, 0],
        track[:, 1],
    )

    gx = np.asarray(gx)
    gy = np.asarray(gy)

    n = len(track)
    m = len(route)

    previous = np.full(m, np.inf)
    back = np.zeros((n, m), dtype=np.int32)

    distances = np.zeros((n, m), dtype=np.float32)

    for i in range(n):

        dx = bx - ax
        dy = by - ay

        denominator = dx * dx + dy * dy

        t = np.zeros(m)
        valid = denominator > 0

        t[valid] = (
            (gx[i] - ax[valid]) * dx[valid]
            + (gy[i] - ay[valid]) * dy[valid]
        ) / denominator[valid]

        np.clip(t, 0, 1, out=t)

        qx = ax + t * dx
        qy = ay + t * dy

        distance_squared = (
            (gx[i] - qx) ** 2
            + (gy[i] - qy) ** 2
        )

        distances[i] = np.sqrt(distance_squared)

        if i == 0:
            previous = distance_squared
            continue

        prefix = np.minimum.accumulate(previous)

        best_indices = np.empty(m, dtype=np.int32)

        best_idx = 0
        best_value = previous[0]

        for j in range(m):
            if previous[j] < best_value:
                best_idx = j
                best_value = previous[j]

            best_indices[j] = best_idx

        back[i] = best_indices

        previous = prefix + distance_squared

    positions = np.empty(n, dtype=np.int32)

    positions[-1] = int(np.argmin(previous))

    for i in range(n - 1, 0, -1):
        positions[i - 1] = back[i, positions[i]]

    aligned_arcs = route[positions]

    aligned_distances = distances[
        np.arange(n),
        positions,
    ]

    return aligned_arcs, aligned_distances


def read_folds():
    assignments = {}

    with FOLD_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        required = {"trip_id", "fold", "role"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Fold file needs trip_id, fold and role columns"
            )

        for row in reader:
            trip_id = str(row["trip_id"]).zfill(8)

            assignments.setdefault(
                trip_id, []
            ).append((
                row["fold"],
                row["role"],
            ))

    return assignments


def candidate_groups(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        current_trip = None
        rows = []

        for row in reader:

            trip_id = str(row["trip_id"]).zfill(8)

            if current_trip is not None and trip_id != current_trip:
                yield current_trip, rows
                rows = []

            current_trip = trip_id
            rows.append(row)

        if current_trip is not None:
            yield current_trip, rows


def process_partition(partition):

    input_file = INPUTS[partition]
    output_file = OUTPUTS[partition]

    successful = 0
    failed = 0
    report = []

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=COLUMNS,
        )

        writer.writeheader()

        for trip_id, rows in candidate_groups(input_file):

            folder = DATASET_ROOT / partition / trip_id

            try:

                track = load_array(
                    folder / f"{trip_id}.track", 3
                )

                nodes = load_array(
                    folder / f"{trip_id}.nodes", 2
                )

                raw_arcs = load_array(
                    folder / f"{trip_id}.arcs", 2
                )

                if not np.allclose(
                    raw_arcs,
                    np.round(raw_arcs),
                ):
                    raise ValueError(
                        "Arc endpoints must be integers"
                    )

                arcs = np.round(raw_arcs).astype(np.int64)

                if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
                    raise ValueError(
                        "Invalid node IDs in arcs"
                    )

                transformer, crs = get_transformer(nodes)

                if rows[0]["crs"] and rows[0]["crs"] != crs:
                    raise ValueError("CRS mismatch")

                node_x, node_y = transformer.transform(
                    nodes[:, 0],
                    nodes[:, 1],
                )

                node_x = np.asarray(node_x)
                node_y = np.asarray(node_y)

                arc_lengths = np.hypot(
                    node_x[arcs[:, 1]] - node_x[arcs[:, 0]],
                    node_y[arcs[:, 1]] - node_y[arcs[:, 0]],
                )

                if partition == "train":

                    route = np.loadtxt(
                        folder / f"{trip_id}.route",
                        dtype=np.int64,
                    ).reshape(-1)

                    if len(route) == 0:
                        raise ValueError("Empty route")

                    if np.any(route < 0) or np.any(route >= len(arcs)):
                        raise ValueError("Invalid route arc ID")

                    aligned_arcs, aligned_distances = (
                        align_track_to_route(
                            track,
                            nodes,
                            arcs,
                            route,
                            transformer,
                        )
                    )

                else:
                    aligned_arcs = None
                    aligned_distances = None

                positive_points = set()
                seen_points = set()
                max_distance = 0.0

                for row in rows:

                    point_idx = int(row["observation_id"])
                    arc_id = int(row["arc_id"])

                    if not 0 <= point_idx < len(track):
                        raise ValueError("Invalid observation ID")

                    if not 0 <= arc_id < len(arcs):
                        raise ValueError("Invalid candidate arc ID")

                    seen_points.add(point_idx)

                    if partition == "train":

                        label = int(
                            arc_id == int(aligned_arcs[point_idx])
                        )

                        if label == 1:
                            positive_points.add(point_idx)

                        max_distance = max(
                            max_distance,
                            float(aligned_distances[point_idx]),
                        )

                    else:
                        label = ""

                    output_row = {
                        "trip_id": trip_id,
                        "point_idx": point_idx,
                        "arc_id": arc_id,
                        "label": label,
                        "arc_len": round(
                            float(arc_lengths[arc_id]), 3
                        ),
                        "split": partition,
                    }

                    for feature in FEATURES:
                        output_row[feature] = row.get(feature, "")

                    writer.writerow(output_row)

                if partition == "train":

                    report.append({
                        "trip_id": trip_id,
                        "observations": len(track),
                        "candidate_observations": len(seen_points),
                        "observations_with_positive_candidate": len(
                            positive_points
                        ),
                        "observations_without_positive_candidate": (
                            len(seen_points) - len(positive_points)
                        ),
                        "max_aligned_route_distance_m": round(
                            max_distance, 3
                        ),
                        "alignment_status": (
                            "review_distance"
                            if max_distance > MAX_REVIEW_DISTANCE
                            else "generated_monotonic_alignment"
                        ),
                    })

                successful += 1

                print(
                    f"{partition}: {trip_id} completed "
                    f"({len(rows):,} candidate rows)"
                )

            except Exception as error:

                failed += 1

                print(
                    f"FAILED {trip_id}: {error}",
                    file=sys.stderr,
                )

    if partition == "train":

        with REPORT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            fields = [
                "trip_id",
                "observations",
                "candidate_observations",
                "observations_with_positive_candidate",
                "observations_without_positive_candidate",
                "max_aligned_route_distance_m",
                "alignment_status",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(report)

    print(f"\n{partition.upper()} SUMMARY")
    print("Successful trips:", successful)
    print("Failed trips:", failed)
    print("Output:", output_file)


def main():

    for path in [
        *INPUTS.values(),
        FOLD_FILE,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    fold_assignments = read_folds()

    print(
        "Loaded fold assignments for",
        len(fold_assignments),
        "trips",
    )

    process_partition("train")
    process_partition("student_test")

    print("\nFEATURE GENERATION COMPLETE")
    print("Check:", REPORT_FILE)


if __name__ == "__main__":
    main()