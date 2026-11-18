"""Align student-test GPS observations to supplied route sequences.

This follows the monotonic point-to-route alignment used by the data-prep
branch, using a per-trip local equirectangular projection (metre scale) so the
script only needs NumPy. It creates derived development labels, not independently
verified ground truth. Review large alignment distances before reporting scores.

    python align_student_test_labels.py

Outputs:
    Dataset/student_test_observation_labels.csv
    Dataset/student_test_label_alignment_report.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Student folder" / "student_test"
CANDIDATE_FILE = ROOT / "Dataset" / "candidates_student_test.csv"
LABEL_FILE = ROOT / "Dataset" / "student_test_observation_labels.csv"
REPORT_FILE = ROOT / "Dataset" / "student_test_label_alignment_report.csv"
REVIEW_DISTANCE_M = 250.0
EARTH_RADIUS_M = 6_371_000.0


def load_rows(path: Path, names: list[str]) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError(f"Invalid values/columns in {path}")
    return values


def project_local(lon: np.ndarray, lat: np.ndarray, lat0: float):
    """Project a small trip area to a local equirectangular metre plane."""
    x = EARTH_RADIUS_M * np.radians(lon) * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat)
    return x, y


def align_track_to_route(track: np.ndarray, nodes: np.ndarray,
                         arcs: np.ndarray, route: np.ndarray):
    """Assign each GPS point to a nondecreasing position in the route sequence."""
    if len(track) == 0 or len(route) == 0:
        raise ValueError("Track and reference route must be nonempty")
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Route contains arc IDs outside the trip arc table")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Arc endpoint references a missing node")

    lat0 = float(np.mean(track[:, 1]))
    node_x, node_y = project_local(nodes[:, 0], nodes[:, 1], lat0)
    gps_x, gps_y = project_local(track[:, 0], track[:, 1], lat0)
    route_arcs = arcs[route.astype(np.int64)]
    ax, ay = node_x[route_arcs[:, 0]], node_y[route_arcs[:, 0]]
    bx, by = node_x[route_arcs[:, 1]], node_y[route_arcs[:, 1]]
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    valid = denominator > 0

    n_points, n_route = len(track), len(route)
    previous = np.full(n_route, np.inf, dtype=np.float64)
    back = np.zeros((n_points, n_route), dtype=np.int32)
    point_distances = np.zeros((n_points, n_route), dtype=np.float32)

    for i in range(n_points):
        t = np.zeros(n_route, dtype=np.float64)
        t[valid] = ((gps_x[i] - ax[valid]) * dx[valid]
                    + (gps_y[i] - ay[valid]) * dy[valid]) / denominator[valid]
        np.clip(t, 0.0, 1.0, out=t)
        qx, qy = ax + t * dx, ay + t * dy
        d2 = (gps_x[i] - qx) ** 2 + (gps_y[i] - qy) ** 2
        point_distances[i] = np.sqrt(d2)
        if i == 0:
            previous = d2
            continue
        prefix = np.minimum.accumulate(previous)
        best_indices = np.empty(n_route, dtype=np.int32)
        best_index = 0
        best_value = previous[0]
        for j in range(n_route):
            if previous[j] < best_value:
                best_index, best_value = j, previous[j]
            best_indices[j] = best_index
        back[i] = best_indices
        previous = prefix + d2

    positions = np.empty(n_points, dtype=np.int32)
    positions[-1] = int(np.argmin(previous))
    for i in range(n_points - 1, 0, -1):
        positions[i - 1] = back[i, positions[i]]
    return route[positions].astype(np.int64), point_distances[np.arange(n_points), positions]


def read_candidates():
    candidates = {}
    with CANDIDATE_FILE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trip_id = str(row["trip_id"]).zfill(8)
            key = (trip_id, int(row["observation_id"]))
            candidates.setdefault(key, set()).add(int(row["arc_id"]))
    return candidates


def main():
    candidates = read_candidates()
    label_rows, report_rows = [], []
    trip_dirs = sorted((p for p in DATA_ROOT.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    for folder in trip_dirs:
        trip_id = folder.name
        track = load_rows(folder / f"{trip_id}.track", ["lon", "lat", "time"])
        nodes = load_rows(folder / f"{trip_id}.nodes", ["lon", "lat"])
        arcs_float = load_rows(folder / f"{trip_id}.arcs", ["from_node", "to_node"])
        if not np.allclose(arcs_float, np.round(arcs_float)):
            raise ValueError(f"Non-integer arc endpoints in trip {trip_id}")
        arcs = np.round(arcs_float).astype(np.int64)
        route_float = np.loadtxt(folder / f"{trip_id}.route", dtype=float)
        route = np.atleast_1d(route_float).astype(np.int64)

        aligned_arcs, distances = align_track_to_route(track, nodes, arcs, route)
        positive_candidates = 0
        candidate_observations = 0
        candidate_misses = 0
        for obs_id, (arc_id, distance) in enumerate(zip(aligned_arcs, distances)):
            candidate_set = candidates.get((trip_id, obs_id), set())
            candidate_observations += bool(candidate_set)
            found = int(int(arc_id) in candidate_set)
            positive_candidates += found
            candidate_misses += int(bool(candidate_set) and not found)
            label_rows.append({
                "trip_id": trip_id,
                "observation_id": obs_id,
                "arc_id": int(arc_id),
                "aligned_distance_m": round(float(distance), 3),
            })

        maximum = float(np.max(distances))
        report_rows.append({
            "trip_id": trip_id,
            "observations": len(track),
            "route_arcs": len(route),
            "observations_with_candidate": int(candidate_observations),
            "aligned_arc_in_candidate_set": int(positive_candidates),
            "candidate_label_misses": candidate_misses,
            "candidate_miss_rate": (candidate_misses / candidate_observations
                                     if candidate_observations else ""),
            "median_alignment_distance_m": round(float(np.median(distances)), 3),
            "p95_alignment_distance_m": round(float(np.percentile(distances, 95)), 3),
            "max_alignment_distance_m": round(maximum, 3),
            "alignment_status": "review_distance" if maximum > REVIEW_DISTANCE_M else "generated_monotonic_alignment",
        })
        print(f"{trip_id}: {len(track):,} observations; max route distance {maximum:.1f} m; "
              f"candidate misses {candidate_misses:,}/{candidate_observations:,}")

    for path, rows in ((LABEL_FILE, label_rows), (REPORT_FILE, report_rows)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote labels: {LABEL_FILE}")
    print(f"Wrote alignment diagnostics: {REPORT_FILE}")
    print("These are derived student-test development labels; inspect review_distance trips before reporting metrics.")


if __name__ == "__main__":
    main()
