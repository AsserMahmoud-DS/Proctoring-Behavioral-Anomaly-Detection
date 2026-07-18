# CheatDetect

Behavioral anomaly detection through mouse and keyboard actions collected from students during live exam sessions.

## Overview

Detects cheating behavior by extracting kinematic and event-based features from raw session data, then classifying each time window as `anomalous` or `normal`. Built for small datasets (~20 sessions) using classical unsupervised ML approaches (Isolation Forest, GMM, One-Class SVM).

## Project Structure

```
CheatDetect/
├── data/
│   ├── raw/                          # Raw CSV session files
│   │   ├── mixed/                    # Sessions with cheating segments
│   │   └── pure normal/              # Clean sessions
│   └── processed/                    # Feature-engineered outputs
├── models/                           # Serialized models (.joblib/.pkl)
├── notebooks/                        # EDA, experiments, benchmarking
├── src/cheatdetect/
│   ├── data/                         # Data pipeline
│   │   ├── loader.py                 #   Load CSVs (single or directory)
│   │   ├── cleaning.py              #   Raw session cleaning & flagging
│   │   ├── pipeline.py              #   End-to-end orchestration
│   │   └── features/
│   │       ├── extract.py           #   Chunking & feature orchestration
│   │       ├── mouse.py             #   Mouse kinematic features (24)
│   │       ├── keyboard.py          #   Keyboard typing features (3)
│   │       └── actions.py            #   Action event features (6)
│   ├── preprocessing/
│   │   └── transform.py             #   Post-extraction cleaning for models
│   ├── app/                          # FastAPI application (planned)
│   ├── train/                        # Model training (planned)
│   ├── eval/                         # Evaluation & reporting (planned)
│   └── config/                       # Configuration (planned)
└── tests/
```

## Feature Set (34 features per chunk)

| Category | Features | Count |
|---|---|---|
| Mouse kinematics | velocity (mean/std/max), acceleration (mean/std/max), jerk (mean/std) | 8 |
| Mouse spatial | path length, straightness, direction changes, angular velocity (mean/std/min/max), curvature (mean/std/min/max), direction class, sum of angles, largest deviation, sharp angles | 13 |
| Mouse other | click count, idle time ratio | 2 |
| Keyboard | typing rate, burst count, pause count | 3 |
| Actions | elapsed time, copy/paste/blur/focus/tab-switch events | 6 |
| Label | is_cheating | 1 |

Events are chunked by count (default: 5 events per window, sliding by 1) and labeled by cheating-event ratio within the window.

## Quick Start

```bash
# Setup
uv venv cheatdetect
uv sync

# Run the full pipeline
python -c "
from cheatdetect.data import process_sessions
df = process_sessions('data/raw/pure normal/')
print(df.head())
"

# Run API (when ready)
uv run uvicorn src.cheatdetect.app.app:app --reload

# Run tests
uv run pytest
```

## Tech Stack

- Python 3.12, uv, FastAPI, scikit-learn, pandas, numpy
- Jupytext for notebook sync (`.py` → `.ipynb`)

