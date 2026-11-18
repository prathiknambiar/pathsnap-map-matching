# 5. HMM/Viterbi decoder and evaluation

## Method

Each candidate directed road arc is a hidden state for a GPS observation. The classifier's exported arc scores provide the emission probabilities; scores are restricted to the generated candidates and normalized within each observation. The Viterbi decoder works in log space and combines these emissions with transitions based on directed road-network reachability and distance between consecutive observations. Transitions search up to six directed network hops, use 80 m per hop with a 150 m distance tolerance and 75 m distance scale, and assign a penalty of -20 to disconnected transitions. Sequences are split at missing observations and timestamp gaps over 30 seconds. If an observation has no usable candidate state, it is retained as an unknown state rather than forcing an unsupported arc.

Parameters were selected using out-of-fold training classifier scores. Seven configurations were compared by mean per-trip Macro F1. The six training trips with route alignment distance above 250 m or candidate miss rate above 10% were excluded. The selected configuration was `max_hops=6`, `meters_per_hop=80`, `distance_tolerance_m=150`, `distance_scale_m=75`, and `disconnected_penalty=-20`. On the 74 eligible training trips, mean per-trip Macro F1 was 0.8642 for classifier-only and 0.8891 for HMM; mean Top-1 accuracy was 0.8169 and 0.8499, respectively. The complete search is in `hmm_outputs/decoder_tuning.csv`; excluded trips and reasons are recorded in `hmm_outputs/decoder_tuning_exclusions.csv`.

## Student-test development evaluation

The selected settings were applied once to the ten `student_test` trips. Labels were produced by monotonically aligning GPS observations to each supplied route sequence. These are route-alignment-derived proxy labels, not independently verified GPS ground truth. Across the 19,160 observations, mean and median alignment distances were 6.67 m and 5.26 m, the 95th percentile was 16.85 m, and the maximum was 157.85 m. There were 67 aligned labels (0.35%) whose arc was absent from the candidate set.

| Mean per trip | Classifier only | HMM/Viterbi | Change |
|---|---:|---:|---:|
| Top-1 accuracy | 0.8257 | 0.8535 | +2.77 pp |
| Macro F1 | 0.8617 | 0.8908 | +2.91 pp |
| Direct transition continuity | 0.8905 | 0.9280 | +3.75 pp |
| Reachable within six hops | 0.9669 | 0.9993 | +3.24 pp |

The means are unweighted averages of the ten trip-level values. The results suggest improved route continuity and proxy-label agreement for these trips; the small, route-derived evaluation set does not establish generalization. Per-trip metrics, coverage, and inference time are in `hmm_outputs/route_continuity.csv`; aligned predictions are in `hmm_outputs/predictions.csv`.

# 6. Prediction maps

Interactive Folium maps are generated for all ten student-test trips in `hmm_outputs/`. Each map includes selectable layers for the local candidate road network, GPS observations, supplied reference route, classifier-only route, and HMM/Viterbi route. The maps use the local network geometry and do not request OpenStreetMap tiles, avoiding the blocked tile-server behavior observed during development. Open a `*_prediction_map.html` file in a browser and use the layer control to compare the routes. The map is a visual diagnostic; route scores should be read from the CSV metrics above.
