# PathSnap evaluator UI and LAN API

The UI runs in a browser on the evaluator laptop. It gets trip metrics and
maps from the backend laptop over the local network. The API serves the saved
`hmm_outputs/route_continuity.csv` results and generated map files; it does not
run live model inference or retrain a model.

## Backend laptop

From the repository root, install the API dependencies and start the API:

```bash
python -m pip install -r requirements_api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Find this laptop's local network IP address (for example, `192.168.1.10`).
The backend needs `hmm_outputs/route_continuity.csv` and the corresponding
`*_prediction_map.html` files. Generate them with `python run_hmm.py` if needed.

## UI laptop

Keep the repository files on this laptop, then from the repository root run:

```bash
python -m http.server 8000 --bind 0.0.0.0
```

Open `http://localhost:8000/ui/`. In **Backend API URL**, enter
`http://<backend-laptop-ip>:8001/api` (replace the example with the backend
laptop's actual IP), then click **Connect**. Select a trip to request its
metrics and map from the backend.

Both laptops must be on a network that allows them to reach each other. If
connection fails, check the backend IP and allow inbound connections on port
8001. Accuracy and F1 use route-alignment-derived proxy labels, not
independently verified ground truth.
