"""Small training-only search for PathSnap decoder parameters.

Uses LightGBM out-of-fold scores and route-aligned labels from training trips.
Trips whose route alignment exceeds 250 m or whose true arc is missing from more
than 10% of candidate sets are excluded from tuning and listed in the log.
Student-test trips are not used to select parameters.

    python align_student_test_labels.py --partition train
    python tune_hmm.py

Writes hmm_outputs/decoder_tuning.csv and decoder_selected.json. Apply the
selected settings to student-test once with run_hmm.py and the observation labels.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from hmm_decoder import classification_metrics, haversine_m, route_continuity, split_indices, viterbi_decode
from run_hmm import read_raw_trip, score_candidates


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hmm_outputs"
OUT.mkdir(exist_ok=True)

SEARCH = [
    {"name": "baseline", "max_hops": 6, "meters_per_hop": 80.0, "distance_tolerance_m": 150.0, "distance_scale_m": 150.0, "disconnected_penalty": -12.0},
    {"name": "hops_3", "max_hops": 3, "meters_per_hop": 80.0, "distance_tolerance_m": 150.0, "distance_scale_m": 150.0, "disconnected_penalty": -12.0},
    {"name": "hops_8", "max_hops": 8, "meters_per_hop": 80.0, "distance_tolerance_m": 150.0, "distance_scale_m": 150.0, "disconnected_penalty": -12.0},
    {"name": "tolerance_75m", "max_hops": 6, "meters_per_hop": 80.0, "distance_tolerance_m": 75.0, "distance_scale_m": 150.0, "disconnected_penalty": -12.0},
    {"name": "tolerance_300m", "max_hops": 6, "meters_per_hop": 80.0, "distance_tolerance_m": 300.0, "distance_scale_m": 150.0, "disconnected_penalty": -12.0},
    {"name": "strong_transitions", "max_hops": 6, "meters_per_hop": 80.0, "distance_tolerance_m": 150.0, "distance_scale_m": 75.0, "disconnected_penalty": -20.0},
    {"name": "weak_transitions", "max_hops": 6, "meters_per_hop": 80.0, "distance_tolerance_m": 150.0, "distance_scale_m": 300.0, "disconnected_penalty": -6.0},
]


def grouped_csv(path: Path, trip_field: str):
    """Yield consecutive trip groups from trip-sorted CSVs without loading all rows."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current = None
        rows = []
        for row in reader:
            trip_id = str(row[trip_field]).zfill(8)
            if current is not None and trip_id != current:
                yield current, rows
                rows = []
            current = trip_id
            rows.append(row)
        if current is not None:
            yield current, rows


def load_labels(path: Path):
    labels = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[str(row["trip_id"]).zfill(8)][int(row["observation_id"])] = int(row["arc_id"])
    return labels


def load_alignment_quality(path: Path):
    quality = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            quality[str(row["trip_id"]).zfill(8)] = row
    return quality


def main():
    root = ROOT
    candidates_iter = iter(grouped_csv(root / "Dataset" / "candidates_train.csv", "trip_id"))
    probabilities_iter = iter(grouped_csv(root / "model" / "LightGBM_train_oof_probs.csv", "trip_id"))
    labels = load_labels(root / "Dataset" / "train_observation_labels.csv")
    alignment = load_alignment_quality(root / "Dataset" / "train_label_alignment_report.csv")
    per_config = {entry["name"]: defaultdict(list) for entry in SEARCH}
    excluded = []
    count = 0

    while True:
        try:
            trip_c, candidate_rows = next(candidates_iter)
            trip_p, probability_rows = next(probabilities_iter)
        except StopIteration:
            break
        if trip_c != trip_p:
            raise ValueError(f"Training trip order differs: candidates={trip_c}, probabilities={trip_p}")
        trip_id = trip_c
        q = alignment[trip_id]
        max_distance = float(q["max_alignment_distance_m"])
        miss_rate = float(q["candidate_miss_rate"] or 0.0)
        if max_distance > 250.0 or miss_rate > 0.10:
            excluded.append({"trip_id": trip_id, "max_alignment_distance_m": max_distance,
                             "candidate_miss_rate": miss_rate,
                             "reason": "alignment_distance_over_250m" if max_distance > 250 else "candidate_miss_rate_over_10pct"})
            continue

        candidate_obs = defaultdict(list)
        for row in candidate_rows:
            obs_id = int(row["observation_id"])
            candidate_obs[obs_id].append({"arc_id": int(row["arc_id"]), "row": row})
        score_obs = defaultdict(dict)
        for row in probability_rows:
            score_obs[int(row["point_idx"])][int(row["arc_id"])] = float(row["prob"])

        track, _, endpoints, graph = read_raw_trip(root / "Student folder", "train", trip_id)
        candidate_lists, probability_lists, classifier = [], [], []
        for obs_id in range(len(track)):
            ids, probs, _ = score_candidates(candidate_obs.get(obs_id, []), score_obs.get(obs_id, {}), 1e-12)
            candidate_lists.append(ids)
            probability_lists.append(probs)
            classifier.append(ids[max(range(len(probs)), key=probs.__getitem__)])
        truth_map = labels[trip_id]
        obs_ids = [i for i in range(len(track)) if i in truth_map]
        y_true = [truth_map[i] for i in obs_ids]
        y_classifier = [classifier[i] for i in obs_ids]
        gps_distances = [0.0]
        for i in range(1, len(track)):
            a = (track[i-1]["longitude"], track[i-1]["latitude"])
            b = (track[i]["longitude"], track[i]["latitude"])
            gps_distances.append(haversine_m(a, b))
        breaks = split_indices(list(range(len(track))), [p["timestamp"] for p in track], 30.0)

        for params in SEARCH:
            decoded = [-1] * len(track)
            starts = [0] + sorted(breaks)
            ends = sorted(breaks) + [len(track)]
            for start, end in zip(starts, ends):
                if start == end:
                    continue
                chunk_distances = [0.0] + gps_distances[start+1:end]
                decoded[start:end] = viterbi_decode(
                    candidate_lists[start:end], probability_lists[start:end], endpoints, graph,
                    chunk_distances, max_hops=params["max_hops"],
                    meters_per_hop=params["meters_per_hop"],
                    distance_tolerance_m=params["distance_tolerance_m"],
                    distance_scale_m=params["distance_scale_m"],
                    disconnected_log_penalty=params["disconnected_penalty"],
                )
            y_hmm = [decoded[i] for i in obs_ids]
            metrics = classification_metrics(y_true, y_hmm)
            direct, reachable = route_continuity(decoded, endpoints, breaks,
                                                   params["max_hops"], graph)
            per_config[params["name"]]["hmm_accuracy"].append(metrics["accuracy"])
            per_config[params["name"]]["hmm_macro_f1"].append(metrics["macro_f1"])
            per_config[params["name"]]["direct_continuity"].append(direct)
            per_config[params["name"]]["reachable_continuity"].append(reachable)
        # Baseline per-trip metrics for paired comparison.
        baseline = classification_metrics(y_true, y_classifier)
        per_config["baseline"]["classifier_accuracy"].append(baseline["accuracy"])
        per_config["baseline"]["classifier_macro_f1"].append(baseline["macro_f1"])
        count += 1
        if count % 10 == 0:
            print(f"Tuned {count} eligible training trips...", flush=True)

    results = []
    config_by_name = {x["name"]: x for x in SEARCH}
    for params in SEARCH:
        metrics = per_config[params["name"]]
        row = dict(params)
        row["eligible_training_trips"] = count
        for metric, values in metrics.items():
            if metric == "classifier_accuracy" or metric == "classifier_macro_f1":
                continue
            row[f"mean_{metric}"] = sum(values) / len(values) if values else math.nan
        results.append(row)
    baseline = results[0]
    for row in results:
        row["mean_classifier_accuracy"] = sum(per_config["baseline"]["classifier_accuracy"]) / count
        row["mean_classifier_macro_f1"] = sum(per_config["baseline"]["classifier_macro_f1"]) / count
        row["objective"] = "mean per-trip HMM Macro F1 on eligible train OOF trips"
    results.sort(key=lambda r: (r["mean_hmm_macro_f1"], r["mean_hmm_accuracy"]), reverse=True)
    best = results[0]
    best_config = {k: best[k] for k in ("max_hops", "meters_per_hop", "distance_tolerance_m",
                                         "distance_scale_m", "disconnected_penalty")}
    (OUT / "decoder_selected.json").write_text(json.dumps({
        "selected_config": best_config,
        "selected_name": best["name"],
        "selection_objective": best["objective"],
        "eligible_training_trips": count,
        "excluded_trips": excluded,
        "note": "Selected on train OOF predictions only; student-test was not used for tuning.",
    }, indent=2))
    with (OUT / "decoder_tuning.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (OUT / "decoder_tuning_exclusions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trip_id", "max_alignment_distance_m", "candidate_miss_rate", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    print(f"\nEligible training trips: {count}; excluded: {len(excluded)}")
    print("Selected:", best["name"], best_config)
    print(f"Train mean Macro F1: classifier={best['mean_classifier_macro_f1']:.4f}, HMM={best['mean_hmm_macro_f1']:.4f}")
    print("Wrote decoder_tuning.csv, decoder_selected.json, and decoder_tuning_exclusions.csv")


if __name__ == "__main__":
    main()
