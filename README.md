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
│   └── processed/                    # Regenerated split metadata (no caches)
├── best_models/                      # Serialized model + inference config
├── notebooks/                        # EDA + train/eval (jupytext .py ↔ .ipynb)
├── reports/                          # Generated plots and metrics
├── plans/                            # Architecture and design docs
├── src/cheatdetect/
│   ├── config.py                     # Static paths + ExperimentConfig
│   ├── utils.py                      # _find_project_root
│   ├── data/                         # Raw events → fixed-schema matrices
│   │   ├── loader.py                 #   Load CSVs (single or directory)
│   │   ├── cleaning.py               #   Raw session + feature-matrix cleaning
│   │   ├── build.py                  #   Column merging (window_switch_events)
│   │   ├── augment.py                #   Gaussian-noise augmentation
│   │   ├── feature_schema.py         #   Frozen 25-source → 33-encoded schema
│   │   ├── preprocessing.py          #   FeaturePreprocessor recipes + checks
│   │   ├── paired.py                 #   Aligned flat/micro views per window
│   │   ├── dataset.py                #   Manifest-guarded PreparedStudy builder
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
│   ├── pipeline/                     # Training orchestration
│   │   ├── train.py                  #   prepare_data + train_pipeline
│   │   ├── search.py                 #   Validation-only model search
│   │   └── report.py                 #   Held-out test reporting
│   ├── experiments/                  # Tracked experiment framework
│   │   └── experiment_1/             #   Original-training audit + docs
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

## Feature Set

Production models consume the frozen **25-source schema**: the extractor's 34
raw features with blur/focus/tab-switch merged into `window_switch_events`, and
`FeaturePreprocessor` encodes direction into 9 indicator columns (33 outputs).

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

### Shared data preparation

Reusable behavioral preprocessing and aligned extraction are available from
`cheatdetect.data`:

- `FeaturePreprocessor`: original-training median imputation, selective
  base/log/Yeo–Johnson/quantile transforms, block scaling, and fixed direction
  encoding. Fit on original normal training once, then reuse for synthetic and
  held-out inputs. The fixed 25-source-feature schema produces 33 encoded columns.
- `build_paired_representation` / `extract_paired_features`: parent aggregates
  and micro-feature sequences extracted from the same realized original/noisy
  window, with configurable geometry and stable parent/copy identities.
- `load_parent_sessions`: ordered loading with explicit zero-window exclusion
  reports; fails if a required partition has no usable windows.

These implementations and their regression tests are tracked. Local study
orchestration imports them; no production module imports `study/`. Existing
`train_pipeline` defaults and saved deployed models are not automatically replaced
by an experimental recipe. Model-specific wiring and recipe selection remain
separate from reusable data preparation.

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
| `test_service.py` | Inference orchestration | 6 |
| `test_api.py` | HTTP transport | 4 |
| `test_preprocessing.py` | Fitted feature preprocessing | 20 |
| `test_paired.py` | Aligned extraction and session loading | 14 |
| `test_mouse_features.py` | Circular turn-angle regression | 8 |

## Tech Stack

- Python 3.12, uv, FastAPI, Pydantic v2, scikit-learn, pandas, numpy, joblib
- Jupytext for notebook sync (`.py` ↔ `.ipynb`)
- pytest for testing
