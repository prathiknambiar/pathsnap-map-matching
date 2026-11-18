"""HMM/Viterbi route decoder for PathSnap candidate arcs.

Hidden states are candidate arc IDs. Classifier scores are emissions. Directed
road-network reachability and a rough network-distance/GPS-distance agreement
score define transitions. This is a starter decoder: calibrate parameters with
training/student-test data and report the choices and limitations.
"""

from __future__ import annotations

import math
from collections import deque
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

EPS = 1e-12


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between (longitude, latitude) coordinates."""
    lon1, lat1 = a
    lon2, lat2 = b
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def make_graph(arc_endpoints: Mapping[int, tuple[int, int]]) -> dict[int, list[int]]:
    """Create a directed node adjacency list from arc_id -> (from, to)."""
    graph: dict[int, list[int]] = {}
    for start, end in arc_endpoints.values():
        graph.setdefault(start, []).append(end)
    return graph


def _make_bfs(graph: Mapping[int, Sequence[int]], max_hops: int):
    """Build a bounded, cached BFS lookup for repeated transition scoring."""

    @lru_cache(maxsize=4096)
    def distances_from(source: int) -> dict[int, int]:
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if distance[node] >= max_hops:
                continue
            for nxt in graph.get(node, ()):
                if nxt not in distance:
                    distance[nxt] = distance[node] + 1
                    queue.append(nxt)
        return distance

    def get(source: int, target: int) -> int | None:
        return distances_from(source).get(target)

    return get


def viterbi_decode(
    candidate_arc_ids: Sequence[Sequence[int]],
    emission_probabilities: Sequence[Sequence[float]],
    arc_endpoints: Mapping[int, tuple[int, int]],
    node_graph: Mapping[int, Sequence[int]],
    observed_distances_m: Sequence[float],
    *,
    max_hops: int = 6,
    meters_per_hop: float = 80.0,
    distance_tolerance_m: float = 150.0,
    distance_scale_m: float = 150.0,
    disconnected_log_penalty: float = -12.0,
) -> list[int]:
    """Decode the most likely arc sequence using the Viterbi recurrence.

    An empty candidate row becomes an unknown state (-1). Probabilities are
    clipped and normalized per observation before taking log emissions.
    """
    if len(candidate_arc_ids) != len(emission_probabilities):
        raise ValueError("Candidate and probability sequences differ in length")
    n = len(candidate_arc_ids)
    if n == 0:
        return []
    if len(observed_distances_m) != n:
        raise ValueError("Need one observed distance per observation")

    states: list[list[int]] = []
    emissions: list[list[float]] = []
    for ids, scores in zip(candidate_arc_ids, emission_probabilities):
        if len(ids) != len(scores):
            raise ValueError("Candidate IDs and probabilities must align")
        if not ids:
            ids, scores = [-1], [1.0]
        values = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in scores]
        total = sum(values)
        values = [x / total for x in values] if total > 0 else [1 / len(values)] * len(values)
        states.append([int(x) for x in ids])
        emissions.append([math.log(max(EPS, x)) for x in values])

    hop_distance = _make_bfs(node_graph, max_hops)
    dynamic = [emissions[0][:]]
    backpointers = [[-1] * len(states[0])]

    for t in range(1, n):
        current_scores = [float("-inf")] * len(states[t])
        current_back = [-1] * len(states[t])
        for j, current_arc in enumerate(states[t]):
            for i, previous_arc in enumerate(states[t - 1]):
                if (previous_arc < 0 or current_arc < 0 or
                        previous_arc not in arc_endpoints or current_arc not in arc_endpoints):
                    transition = disconnected_log_penalty
                else:
                    _, previous_end = arc_endpoints[previous_arc]
                    current_start, _ = arc_endpoints[current_arc]
                    hops = 0 if previous_arc == current_arc else hop_distance(previous_end, current_start)
                    if hops is None:
                        transition = disconnected_log_penalty
                    else:
                        network_distance = hops * meters_per_hop
                        gps_distance = max(0.0, float(observed_distances_m[t]))
                        mismatch = max(0.0, abs(network_distance - gps_distance) - distance_tolerance_m)
                        transition = -mismatch / max(distance_scale_m, EPS)
                score = dynamic[t - 1][i] + transition + emissions[t][j]
                if score > current_scores[j]:
                    current_scores[j], current_back[j] = score, i
        dynamic.append(current_scores)
        backpointers.append(current_back)

    chosen = [max(range(len(dynamic[-1])), key=dynamic[-1].__getitem__)]
    for t in range(n - 1, 0, -1):
        chosen.append(backpointers[t][chosen[-1]])
    chosen.reverse()
    return [states[t][index] for t, index in enumerate(chosen)]


def split_indices(
    observation_ids: Sequence[int], timestamps: Sequence[float], gap_seconds: float = 30.0
) -> set[int]:
    """Return positions that begin a new sequence after missing IDs/large gaps."""
    breaks = set()
    for i in range(1, len(observation_ids)):
        if int(observation_ids[i]) != int(observation_ids[i - 1]) + 1:
            breaks.add(i)
            continue
        try:
            dt = float(timestamps[i]) - float(timestamps[i - 1])
        except (TypeError, ValueError):
            dt = 0.0
        if math.isfinite(dt) and dt > gap_seconds:
            breaks.add(i)
    return breaks


def route_continuity(
    predictions: Sequence[int],
    arc_endpoints: Mapping[int, tuple[int, int]],
    break_before: Iterable[int] = (),
    max_hops: int = 6,
    node_graph: Mapping[int, Sequence[int]] | None = None,
) -> tuple[float, float]:
    """Return direct-connectivity and optional bounded-reachability rates."""
    breaks = set(break_before)
    pairs = [(i, predictions[i - 1], predictions[i])
             for i in range(1, len(predictions))
             if i not in breaks and predictions[i - 1] >= 0 and predictions[i] >= 0]
    if not pairs:
        return float("nan"), float("nan")
    direct = 0
    reachable = 0
    bfs = _make_bfs(node_graph, max_hops) if node_graph is not None else None
    for _, previous, current in pairs:
        if previous == current:
            direct += 1
            if bfs:
                reachable += 1
            continue
        previous_end = arc_endpoints.get(previous, (None, None))[1]
        current_start = arc_endpoints.get(current, (None, None))[0]
        direct += int(previous_end == current_start)
        if bfs and previous_end is not None:
            reachable += int(bfs(previous_end, current_start) is not None)
    return direct / len(pairs), reachable / len(pairs) if bfs else float("nan")


def classification_metrics(
    truth: Sequence[int], predictions: Sequence[int], probabilities_for_truth: Sequence[float] | None = None
) -> dict[str, float]:
    """Compute point accuracy, macro-F1, and optional clipped log loss."""
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction lengths differ")
    labels = sorted(set(int(x) for x in truth))
    if not truth:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "log_loss": float("nan")}
    correct = sum(int(int(y) == int(p)) for y, p in zip(truth, predictions))
    f1s = []
    for label in labels:
        tp = sum(int(y == label and p == label) for y, p in zip(truth, predictions))
        fp = sum(int(y != label and p == label) for y, p in zip(truth, predictions))
        fn = sum(int(y == label and p != label) for y, p in zip(truth, predictions))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    logloss = float("nan")
    if probabilities_for_truth is not None and len(probabilities_for_truth) == len(truth):
        logloss = -sum(math.log(max(EPS, min(1.0, float(p)))) for p in probabilities_for_truth) / len(truth)
    return {"accuracy": correct / len(truth), "macro_f1": sum(f1s) / len(f1s), "log_loss": logloss}
