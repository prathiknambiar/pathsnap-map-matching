"""Run PathSnap classifier-only and HMM route decoding from the current CSVs.

Default: decode the 10 student-test trips using candidate features and exported
LightGBM scores, save predictions/continuity/alignment CSVs, and one Folium map
per trip if folium is installed.

    python run_hmm.py
    python run_hmm.py --partition student_test --maps none
    python run_hmm.py --partition student_test --labels aligned_labels.csv

Optional labels must be an independently validated observation-aligned CSV with
columns trip_id,observation_id,arc_id. This script never reads evaluator data.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from hmm_decoder import (
    classification_metrics,
    haversine_m,
    make_graph,
    route_continuity,
    split_indices,
    viterbi_decode,
)


ROOT = Path(__file__).resolve().parent


def trip_key(value: str) -> str:
    return str(value).strip().zfill(8)


def read_candidates(path: Path):
    """Read candidate rows into trip -> observation -> candidate records."""
    candidates: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    row_count = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"trip_id", "observation_id", "arc_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must include columns {sorted(required)}")
        for row in reader:
            key = trip_key(row["trip_id"])
            obs_id = int(row["observation_id"])
            arc_id = int(row["arc_id"])
            candidates[key][obs_id].append({"arc_id": arc_id, "row": row})
            row_count += 1
    return candidates, row_count


def read_probabilities(path: Path):
    """Read and sum exported model scores per (trip, observation, arc)."""
    scores: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    row_count = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"trip_id", "point_idx", "arc_id", "prob"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must include columns {sorted(required)}")
        for row in reader:
            key = trip_key(row["trip_id"])
            obs_id = int(row["point_idx"])
            arc_id = int(row["arc_id"])
            try:
                probability = float(row["prob"])
            except (TypeError, ValueError):
                probability = 0.0
            if not math.isfinite(probability) or probability < 0:
                probability = 0.0
            scores[key][obs_id][arc_id] += probability
            row_count += 1
    return scores, row_count


def read_raw_trip(data_root: Path, partition: str, trip_id: str):
    folder = data_root / partition / trip_id
    track_path = folder / f"{trip_id}.track"
    arcs_path = folder / f"{trip_id}.arcs"
    nodes_path = folder / f"{trip_id}.nodes"
    if not all(p.is_file() for p in (track_path, arcs_path, nodes_path)):
        missing = [str(p) for p in (track_path, arcs_path, nodes_path) if not p.is_file()]
        raise FileNotFoundError("Missing raw trip files: " + ", ".join(missing))

    track = []
    with track_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            fields = line.split()
            if len(fields) >= 3:
                track.append({"observation_id": i, "longitude": float(fields[0]),
                              "latitude": float(fields[1]), "timestamp": float(fields[2])})
    arc_endpoints = {}
    with arcs_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            fields = line.split()
            if len(fields) >= 2:
                arc_endpoints[i] = (int(fields[0]), int(fields[1]))
    nodes = []
    with nodes_path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                nodes.append((float(fields[0]), float(fields[1])))  # lon, lat
    return track, nodes, arc_endpoints, make_graph(arc_endpoints)


def read_reference_route(data_root: Path, partition: str, trip_id: str):
    """Read the provided chronological arc sequence for visualization only."""
    path = data_root / partition / trip_id / f"{trip_id}.route"
    if not path.is_file():
        return []
    route = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if fields:
                route.append(int(fields[0]))
    return route


def read_labels(path: Path | None):
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    if path is None:
        return labels
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"trip_id", "observation_id", "arc_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Label file must include columns {sorted(required)}")
        for row in reader:
            labels[trip_key(row["trip_id"])][int(row["observation_id"])] = int(row["arc_id"])
    return labels


def score_candidates(candidates, observation_scores, score_floor: float):
    """Restrict exported scores to candidate arcs, floor missing scores, normalize."""
    ids, probabilities, missing = [], [], 0
    # Deduplicate candidate IDs while preserving first occurrence.
    seen = set()
    for candidate in candidates:
        arc_id = int(candidate["arc_id"])
        if arc_id in seen:
            continue
        seen.add(arc_id)
        score = observation_scores.get(arc_id)
        if score is None:
            score = score_floor
            missing += 1
        ids.append(arc_id)
        probabilities.append(max(score_floor, float(score)))
    total = sum(probabilities)
    if not ids:
        return [-1], [1.0], missing
    return ids, [x / total for x in probabilities], missing


def alignment_row(partition, candidates, probabilities, candidate_rows, probability_rows):
    candidate_obs = sum(len(obs) for obs in candidates.values())
    prob_obs = sum(len(obs) for obs in probabilities.values())
    shared = set((t, o) for t, obs in candidates.items() for o in obs) & set(
        (t, o) for t, obs in probabilities.items() for o in obs)
    matched = total = 0
    per_obs_coverage = []
    for trip_id, obs_id in shared:
        candidate_ids = {x["arc_id"] for x in candidates[trip_id][obs_id]}
        probability_ids = set(probabilities[trip_id][obs_id])
        total += len(candidate_ids)
        count = len(candidate_ids & probability_ids)
        matched += count
        per_obs_coverage.append(count / len(candidate_ids) if candidate_ids else 0.0)
    return {
        "partition": partition,
        "candidate_rows": candidate_rows,
        "probability_rows": probability_rows,
        "candidate_observations": candidate_obs,
        "probability_observations": prob_obs,
        "shared_observations": len(shared),
        "candidate_arc_score_coverage": matched / total if total else 0.0,
        "mean_observation_candidate_coverage": sum(per_obs_coverage) / len(per_obs_coverage) if per_obs_coverage else 0.0,
        "probability_groups_over_8_arcs": sum(len(arcs) > 8 for trip in probabilities.values() for arcs in trip.values()),
    }


def run_trip(trip_id, candidate_obs, probability_obs, track, nodes, arc_endpoints, node_graph,
             labels, args):
    # Candidate CSV observation IDs / probability point_idx are assumed to be raw .track row indices.
    # This is checked against trip lengths in the summary, but should be confirmed with the exporter.
    n = len(track)
    candidate_lists, prob_lists, classifier = [], [], []
    score_by_obs_arc: dict[int, dict[int, float]] = {}
    missing_score_count = 0
    for obs_id in range(n):
        candidates = candidate_obs.get(obs_id, [])
        ids, probs, missing = score_candidates(candidates, probability_obs.get(obs_id, {}), args.score_floor)
        candidate_lists.append(ids)
        prob_lists.append(probs)
        classifier.append(ids[max(range(len(probs)), key=probs.__getitem__)])
        score_by_obs_arc[obs_id] = dict(zip(ids, probs))
        missing_score_count += missing

    obs_ids = list(range(n))
    times = [x["timestamp"] for x in track]
    breaks = split_indices(obs_ids, times, gap_seconds=args.gap_seconds)
    decoded = [-1] * n
    chunk_starts = [0] + sorted(breaks)
    chunk_ends = sorted(breaks) + [n]
    for start, end in zip(chunk_starts, chunk_ends):
        if start == end:
            continue
        distances = [0.0]
        for i in range(start + 1, end):
            a = (track[i - 1]["longitude"], track[i - 1]["latitude"])
            b = (track[i]["longitude"], track[i]["latitude"])
            distances.append(haversine_m(a, b))
        decoded[start:end] = viterbi_decode(
            candidate_lists[start:end], prob_lists[start:end], arc_endpoints, node_graph,
            distances, max_hops=args.max_hops, meters_per_hop=args.meters_per_hop,
            distance_tolerance_m=args.distance_tolerance_m,
            distance_scale_m=args.distance_scale_m,
            disconnected_log_penalty=args.disconnected_penalty,
        )

    c_direct, c_reachable = route_continuity(classifier, arc_endpoints, breaks,
                                              args.max_hops, node_graph)
    h_direct, h_reachable = route_continuity(decoded, arc_endpoints, breaks,
                                              args.max_hops, node_graph)
    summary = {
        "trip_id": trip_id,
        "gps_observations": n,
        "candidate_observations": len(candidate_obs),
        "missing_candidate_score_rows": missing_score_count,
        "sequence_splits": len(breaks),
        "classifier_direct_continuity": c_direct,
        "classifier_reachable_continuity": c_reachable,
        "hmm_direct_continuity": h_direct,
        "hmm_reachable_continuity": h_reachable,
    }

    truth = labels.get(trip_id, {})
    if truth:
        scored_ids = [i for i in range(n) if i in truth]
        y_true = [truth[i] for i in scored_ids]
        y_classifier = [classifier[i] for i in scored_ids]
        y_hmm = [decoded[i] for i in scored_ids]
        p_true = [score_by_obs_arc[i].get(truth[i], args.score_floor) for i in scored_ids]
        miss_rate = sum(truth[i] not in candidate_lists[i] for i in scored_ids) / len(scored_ids)
        for prefix, pred in (("classifier", y_classifier), ("hmm", y_hmm)):
            metrics = classification_metrics(y_true, pred, p_true if prefix == "classifier" else None)
            summary[f"{prefix}_top1_accuracy"] = metrics["accuracy"]
            summary[f"{prefix}_macro_f1"] = metrics["macro_f1"]
            if prefix == "classifier":
                summary["classifier_log_loss_on_aligned_candidates"] = metrics["log_loss"]
        summary["candidate_miss_rate"] = miss_rate
        summary["labeled_observations"] = len(scored_ids)
    return summary, classifier, decoded, track, nodes, arc_endpoints, score_by_obs_arc


def _route_segments(predictions, endpoints, nodes):
    segments, current, previous_end, previous_arc = [], [], None, None
    for arc_id in predictions:
        if arc_id < 0 or arc_id not in endpoints:
            if len(current) > 1:
                segments.append(current)
            current, previous_end, previous_arc = [], None, None
            continue
        if arc_id == previous_arc:
            continue
        start_node, end_node = endpoints[arc_id]
        if not (0 <= start_node < len(nodes) and 0 <= end_node < len(nodes)):
            continue
        start = [nodes[start_node][1], nodes[start_node][0]]
        end = [nodes[end_node][1], nodes[end_node][0]]
        if previous_end is not None and start_node != previous_end:
            if len(current) > 1:
                segments.append(current)
            current = []
        if not current:
            current = [start]
        current.append(end)
        previous_end = end_node
        previous_arc = arc_id
    if len(current) > 1:
        segments.append(current)
    return segments


def save_folium_map(path, trip_id, track, nodes, arc_endpoints,
                    classifier, decoded, reference_route=None, truth=None):
    try:
        import folium
    except ImportError:
        return False
    center = [sum(x["latitude"] for x in track) / len(track),
              sum(x["longitude"] for x in track) / len(track)]
    # Use no external tile service: OSM has blocked these requests in the current
    # environment. Candidate arcs below provide an offline road-network backdrop.
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles=None)
    road_layer = folium.FeatureGroup(name="Local road network", show=True)
    lons = [x["longitude"] for x in track]
    lats = [x["latitude"] for x in track]
    # Clip the full supplied graph to a small margin around the observed path.
    # This makes the offline map useful without requesting external map tiles.
    lon_margin = max((max(lons) - min(lons)) * .05, .003)
    lat_margin = max((max(lats) - min(lats)) * .05, .003)
    lon_min, lon_max = min(lons) - lon_margin, max(lons) + lon_margin
    lat_min, lat_max = min(lats) - lat_margin, max(lats) + lat_margin
    for start_node, end_node in arc_endpoints.values():
        if not (0 <= start_node < len(nodes) and 0 <= end_node < len(nodes)):
            continue
        start_lon, start_lat = nodes[start_node]
        end_lon, end_lat = nodes[end_node]
        near_map = (
            lon_min <= start_lon <= lon_max and lat_min <= start_lat <= lat_max
        ) or (
            lon_min <= end_lon <= lon_max and lat_min <= end_lat <= lat_max
        )
        if near_map:
            folium.PolyLine(
                [[start_lat, start_lon], [end_lat, end_lon]],
                color="#777777", weight=1, opacity=.35,
            ).add_to(road_layer)
    road_layer.add_to(fmap)
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    # Thin markers for readability on long trajectories.
    stride = max(1, len(track) // 3000)
    for i in range(0, len(track), stride):
        point = track[i]
        folium.CircleMarker([point["latitude"], point["longitude"]], radius=2,
                            color="blue", fill=True, fill_opacity=.65,
                            popup=f"Observation {i}").add_to(gps_layer)
    gps_layer.add_to(fmap)
    layers = [("Classifier only", classifier, "orange", False), ("HMM/Viterbi", decoded, "red", True)]
    if reference_route:
        layers.insert(0, ("Provided reference route", reference_route, "green", False))
    if truth:
        reference = [truth.get(i, -1) for i in range(len(track))]
        layers.insert(0, ("Aligned reference", reference, "green", False))
    for name, predictions, color, show in layers:
        layer = folium.FeatureGroup(name=name, show=show)
        for segment in _route_segments(predictions, arc_endpoints, nodes):
            folium.PolyLine(segment, color=color, weight=4, opacity=.85, tooltip=name).add_to(layer)
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(path))
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root")
    parser.add_argument("--partition", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--candidate-file", type=Path, default=None)
    parser.add_argument("--probability-file", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional validated observation-aligned labels CSV")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--maps", choices=("all", "none"), default="all")
    parser.add_argument("--gap-seconds", type=float, default=30.0)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--meters-per-hop", type=float, default=80.0)
    parser.add_argument("--distance-tolerance-m", type=float, default=150.0)
    parser.add_argument("--distance-scale-m", type=float, default=150.0)
    parser.add_argument("--disconnected-penalty", type=float, default=-12.0)
    parser.add_argument("--score-floor", type=float, default=1e-12)
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = (args.data_root or root / "Student folder").resolve()
    output_dir = (args.output_dir or root / "hmm_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.candidate_file:
        candidate_path = args.candidate_file.resolve()
    else:
        candidate_path = root / "Dataset" / ("candidates_train.csv" if args.partition == "train"
                                              else "candidates_student_test.csv")
    if args.probability_file:
        probability_path = args.probability_file.resolve()
    else:
        probability_path = root / "model" / ("LightGBM_train_oof_probs.csv" if args.partition == "train"
                                             else "LightGBM_test_probs.csv")
    if not candidate_path.is_file() or not probability_path.is_file():
        raise FileNotFoundError(f"Can't find candidate/probability CSV: {candidate_path} / {probability_path}")

    print(f"Loading {args.partition} candidates and exported scores...")
    candidates, candidate_rows = read_candidates(candidate_path)
    probabilities, probability_rows = read_probabilities(probability_path)
    alignment = alignment_row(args.partition, candidates, probabilities, candidate_rows, probability_rows)
    write_csv(output_dir / "alignment_audit.csv", [alignment])
    print(f"Candidate/probability arc coverage: {alignment['candidate_arc_score_coverage']:.1%}")
    print(f"Mean per-observation candidate coverage: {alignment['mean_observation_candidate_coverage']:.1%}")
    print(f"Probability groups with >8 arcs: {alignment['probability_groups_over_8_arcs']:,}")
    print("Adapter assumption: filter exported scores to candidate arcs, floor missing scores, and renormalize per observation.")

    labels = read_labels(args.labels.resolve() if args.labels else None)
    trip_ids = sorted(candidates, key=int)
    summaries, prediction_rows = [], []
    start_time = time.perf_counter()
    for trip_id in trip_ids:
        print(f"Decoding {trip_id}...", flush=True)
        trip_start = time.perf_counter()
        track, nodes, arc_endpoints, node_graph = read_raw_trip(data_root, args.partition, trip_id)
        summary, classifier, decoded, track, nodes, arc_endpoints, score_by_obs = run_trip(
            trip_id, candidates[trip_id], probabilities.get(trip_id, {}), track, nodes,
            arc_endpoints, node_graph, labels, args)
        summary["inference_seconds_including_data_load"] = time.perf_counter() - trip_start
        summaries.append(summary)
        truth = labels.get(trip_id, {})
        for i, point in enumerate(track):
            prediction_rows.append({"trip_id": trip_id, "observation_id": i,
                                    "classifier_arc_id": classifier[i], "hmm_arc_id": decoded[i],
                                    "reference_arc_id": truth.get(i, "")})
        if args.maps == "all":
            map_path = output_dir / f"{trip_id}_prediction_map.html"
            reference_route = read_reference_route(data_root, args.partition, trip_id)
            if not save_folium_map(map_path, trip_id, track, nodes, arc_endpoints,
                                   classifier, decoded, reference_route, truth):
                print("Folium is not installed; skipping map files. Install folium to create maps.")
                args.maps = "none"
            else:
                print(f"  saved map {map_path.name}")
    elapsed = time.perf_counter() - start_time
    write_csv(output_dir / "route_continuity.csv", summaries)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    print(f"\nDecoded {len(trip_ids)} trips in {elapsed:.2f}s")
    print(f"Wrote {output_dir / 'alignment_audit.csv'}")
    print(f"Wrote {output_dir / 'route_continuity.csv'}")
    print(f"Wrote {output_dir / 'predictions.csv'}")
    if args.labels is None:
        print("No observation-aligned labels supplied: accuracy and Macro F1 are not computed.")
    else:
        print("Label-based metrics included in route_continuity.csv; verify labels and partition before reporting.")


if __name__ == "__main__":
    main()
