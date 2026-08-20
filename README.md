# Proctoring Behavioral Anomaly Detection

Behavioral anomaly detection through mouse and keyboard actions collected from students during live exam sessions.

## Overview

Detects cheating behavior by extracting kinematic and event-based features from raw session data, then classifying each time window as `anomalous` or `normal`. Built for small datasets (~20 sessions) using classical unsupervised ML approaches (Isolation Forest, One-Class SVM, and a weighted ensemble of both).

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
└── tests/
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
python -c "
from cheatdetect.config import ExperimentConfig
from cheatdetect.pipeline import train_pipeline

results = train_pipeline(ExperimentConfig())
print(results['metrics_df'])
"

# Run the API
uv run uvicorn src.cheatdetect.app.app:app --reload

# Run tests
uv run pytest
```

## Notebooks

Notebooks are stored as paired `.py` scripts (Jupytext, `py:percent` format). Edit the `.py`, then sync:

```bash
jupytext --sync notebooks/eda.py
jupytext --sync notebooks/train_evaluate.py
```

## Tech Stack

- Python 3.12, uv, FastAPI, Pydantic, scikit-learn, pandas, numpy, joblib
- Jupytext for notebook sync (`.py` ↔ `.ipynb`)
