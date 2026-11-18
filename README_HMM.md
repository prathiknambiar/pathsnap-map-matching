# PathSnap HMM/Viterbi scripts

`hmm_decoder.py` contains the reusable Viterbi decoder and route metrics.
`run_hmm.py` joins the current candidate and LightGBM probability CSVs, audits
their alignment, decodes trips, and writes prediction and continuity files.

Run from the project folder:

```bash
python run_hmm.py
```

By default it processes only the permitted `student_test` partition and writes
to `hmm_outputs/`. Outputs are:

- `alignment_audit.csv`: candidate/probability overlap and coverage.
- `predictions.csv`: classifier-only and HMM arc predictions per GPS row.
- `route_continuity.csv`: direct connectivity, bounded reachability, and
  per-trip inference time.
- One `*_prediction_map.html` per trip if Folium is installed. Maps avoid
  external tile servers and show the local road graph, GPS observations,
  classifier route, HMM route, and provided reference route where available.

The decoder has no required third-party packages. Install Folium to generate
interactive maps:

```bash
python -m pip install -r requirements_hmm.txt
```

The score adapter matches on trip ID, observation index, and arc ID; restricts
exported scores to the generated candidate arcs; assigns `1e-12` to candidate
arcs with no matching score; then renormalizes within each observation. This is
a provisional assumption because the supplied probability files do not match
the candidate CSVs one-to-one. Confirm what the score export represents with
the model teammate before using these results as final evidence.

The current source files do not include validated labels aligned to GPS rows,
so the default run reports continuity only. If the team prepares a validated
development label CSV with columns `trip_id,observation_id,arc_id`, pass it with
`--labels labels.csv` to calculate Top-1 accuracy, Macro F1, candidate miss
rate, and classifier log loss. The `.route` sequence is not automatically
treated as point-aligned labels.

Useful options:

```bash
python run_hmm.py --partition student_test --maps none
python run_hmm.py --partition train --maps none
python run_hmm.py --gap-seconds 30 --max-hops 6 --meters-per-hop 80
```

The gap threshold and transition parameters are starter settings. Select and
justify final settings using training and student-test data only; never use
evaluator-test data to tune them.
