# CheatDetect: Behavioral Anomaly Detection

Detects cheating behavior through mouse and keyboard actions collected from students during live exam sessions.

## Overview

Extracts kinematic and event-based features from raw session data, then classifies each time window as `anomalous` or `normal` using classical unsupervised ML (Isolation Forest, One-Class SVM, weighted ensemble). Built for small datasets (~20 sessions).

## Project Structure

```
CheatDetect/
├── dataset/
│   ├── raw/                          # Raw CSV session files
│   │   ├── mixed/                    # Sessions with cheating segments
│   │   └── pure normal/              # Clean sessions
│   └── processed/                    # Feature-engineered cache (.pkl)
├── best_models/                      # Serialized model + inference config
├── notebooks/                        # EDA + train/eval (jupytext .py ↔ .ipynb)
├── reports/                          # Generated plots and metrics
├── plans/                            # Architecture and design docs
├── src/cheatdetect/
│   ├── config.py                     # Static paths + ExperimentConfig
│   ├── utils.py                      # _find_project_root
│   ├── data/                         # Raw events → model-ready array
│   │   ├── loader.py                 #   Load CSVs (single or directory)
│   │   ├── cleaning.py               #   Raw session + feature-matrix cleaning
│   │   ├── pipeline.py               #   load → clean → extract orchestration
│   │   ├── build.py                  #   Column merging (window_switch_events)
│   │   ├── transform.py              #   Log1pSkewed + find_skewed_features
│   │   ├── selection.py              #   Zero-var + correlation feature selection
│   │   ├── augment.py                #   Gaussian-noise augmentation
│   │   └── features/
│   │       ├── extract.py            #   Chunking & feature orchestration
│   │       ├── mouse.py              #   Mouse kinematic features (24)
│   │       ├── keyboard.py           #   Keyboard typing features (3)
│   │       └── actions.py            #   Action event features (6)
│   ├── models/                       # Anomaly detectors (shared ABC)
│   │   ├── base.py                   #   AnomalyDetector interface
│   │   ├── isolation_forest.py       #   IF + grid_search
│   │   ├── ocsvm.py                  #   OCSVM + grid_search
│   │   ├── ensemble.py               #   Weighted IF+OCSVM ensemble
│   │   └── threshold.py              #   tune_threshold
│   ├── eval/                         # Evaluation
│   │   ├── metrics.py                #   evaluate_model, compare_models
│   │   └── plots.py                  #   PR/ROC/confusion/score-distributions
│   ├── pipeline/
│   │   └── train.py                  #   prepare_data + train_pipeline orchestration
│   └── app/                          # FastAPI inference API
│       ├── app.py                    #   FastAPI app, lifespan, routes
│       ├── schemas.py                #   Pydantic request/response models
│       ├── service.py                #   PredictionService (stateful inference)
│       ├── buffer.py                 #   WindowBuffer (per-session sliding window)
│       └── dependencies.py           #   Model loading + DI
└── tests/
    ├── test_schema.py                #   Pydantic model validation
    ├── test_buffer.py                #   Sliding window logic
    ├── test_service.py               #   Service orchestration
    └── test_api.py                   #   HTTP layer (TestClient)
```

## Feature Set (34 features per chunk + label)

| Category | Features | Count |
|---|---|---|
| Mouse kinematics | velocity (mean/std/max), acceleration (mean/std/max), jerk (mean/std) | 8 |
| Mouse spatial | path length, straightness, direction changes, angular velocity (mean/std/min/max), curvature (mean/std/min/max), direction class, sum of angles, largest deviation, sharp angles | 15 |
| Mouse other | click count, idle time ratio | 2 |
| Keyboard | typing rate, burst count, pause count | 3 |
| Actions | elapsed time, copy/paste/blur/focus/tab-switch events | 6 |
| Label | is_cheating | 1 |

Events are chunked by count (configurable, default: 50 events per window, sliding by 25) and labeled by the cheating-event ratio within the window.

## Quick Start

```bash
# Setup
uv venv cheatdetect
uv sync

# Run the full training pipeline
uv run python -c "
from cheatdetect.config import ExperimentConfig
from cheatdetect.pipeline.train import train_pipeline

results = train_pipeline(ExperimentConfig())
print(results['metrics_df'])
"

# Run the API
uv run uvicorn cheatdetect.app.app:app --reload

# Run tests
uv run pytest
```

## API

The inference API exposes two endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/predict` | Accepts a batch of events, returns verdict + per-chunk scores |

### `POST /predict` — Request

```json
{
  "session_id": "student_123",
  "events": [
    {"time": 0.0, "event_type": "mousemove", "x": 100.0, "y": 200.0},
    {"time": 0.1, "event_type": "click", "x": 100.0, "y": 200.0, "action": "copy"}
  ]
}
```

### `POST /predict` — Response

```json
{
  "session_id": "student_123",
  "verdict": "anomalous",
  "chunks_pred": [
    {"chunk_index": 0, "score": 0.85, "isanomalous": true}
  ]
}
```

Events are buffered per `session_id` using a sliding window (`chunk_size` events, `step_size` overlap). The API returns a verdict once enough events have accumulated to form a window.

## Notebooks

Notebooks are stored as paired `.py` scripts (Jupytext, `py:percent` format). Edit the `.py`, then sync:

```bash
jupytext --sync notebooks/eda.py
jupytext --sync notebooks/train_evaluate.py
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_service.py -v
```

Tests are organized bottom-up by layer:

| File | Layer | Tests |
|------|-------|-------|
| `test_schema.py` | Data contracts | 12 |
| `test_buffer.py` | Windowing logic | 4 |
| `test_service.py` | Inference orchestration | 5 |
| `test_api.py` | HTTP transport | 4 |

## Tech Stack

- Python 3.12, uv, FastAPI, Pydantic v2, scikit-learn, pandas, numpy, joblib
- Jupytext for notebook sync (`.py` ↔ `.ipynb`)
- pytest for testing
