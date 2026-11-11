# CheatDetect

CheatDetect is a FastAPI service that scores mouse, keyboard, and browser interaction events for behavioral anomalies during online exams. The deployed **Isolation Forest** scores overlapping windows of 50 events, advancing 25 events at a time. An anomaly flag can help an instructor decide which activity to review.

[Live API and Swagger UI](https://proctoring-anomaly-detection-production-0036.up.railway.app/docs) · [Railway health endpoint](https://proctoring-anomaly-detection-production-0036.up.railway.app/health)

## Model evaluation

Experiment 1 compared four anomaly detectors on held-out session data. AP is average precision, and FPR is the false positive rate. The experiment snapshot includes the split, model recipes, and complete evaluation.

| Model | ROC-AUC | AP | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.7138 | 0.7500 | 0.7391 | **0.6634** | 0.6992 | 0.2981 |
| One-Class SVM | **0.7323** | 0.7461 | **0.7798** | 0.6390 | 0.7024 | **0.2298** |
| LSTM autoencoder | 0.6314 | 0.6607 | 0.7099 | 0.4537 | 0.5536 | 0.2360 |
| Weighted ensemble | 0.7191 | **0.7587** | 0.7670 | 0.6585 | **0.7087** | 0.2547 |

The deployed Isolation Forest is loaded from `best_models/best_model.joblib`. Its saved configuration separately records a test PR-AUC of **0.7578** for that artifact.

[Experiment 1 report](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/e97158922154062bf856b1f579550493690028b4/reports/experiment1/README.md) · [Experiment metrics CSV](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/e97158922154062bf856b1f579550493690028b4/reports/experiment1/tables/final_metrics.csv) · [Deployed model configuration](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/faeb1bf8b75bba123c51cbba85f961473d214784/best_models/model_config.json)

## Railway load test

With **30 concurrent simulated exam sessions**, the API sustained **10.20 scoring requests/sec** for approximately three minutes. Mean end-to-end scoring latency was **430 ms**, p95 was **870 ms**, and the saved results recorded **zero failures** across **1,826 scoring requests**.

| Active students | Scoring requests | Successful scores/sec | Mean latency | p95 latency | Failures |
|---|---:|---:|---:|---:|---:|
| 10 | 645 | 3.66 | 229 ms | 290 ms | 0 |
| 30 | 1,826 | 10.20 | 430 ms | 870 ms | 0 |

Locust used staggered starts at two students per second. Each session sent batches of 25 synthetic mouse events and waited 2.5 seconds after each response. Synthetic timestamps advanced by 0.2 seconds per event, so each scoring window represented roughly 10 seconds of simulated behavior. The table uses the scoring request row and measures time from sending the HTTP request to receiving the prediction. The 30-student run exceeded the provisional p95 target of 500 ms.

[10-student results](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/faeb1bf8b75bba123c51cbba85f961473d214784/tests/locust/results/railway-10_stats.csv) · [30-student results](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/faeb1bf8b75bba123c51cbba85f961473d214784/tests/locust/results/railway-30_stats.csv) · [Locust workload](https://github.com/AsserMahmoud-DS/Proctoring-Behavioral-Anomaly-Detection/blob/faeb1bf8b75bba123c51cbba85f961473d214784/tests/locust/locustfile.py)

## Repository structure

```text
src/cheatdetect/
├── app/        FastAPI routes, request schemas, session buffers, inference
├── data/       Event cleaning, feature extraction, and preprocessing
├── models/     Anomaly detectors and threshold selection
├── pipeline/   Training and experiment orchestration
└── eval/       Evaluation metrics and plots
best_models/   Saved detector and inference configuration
notebooks/     Exploratory analysis and training evaluation
tests/         Unit and integration tests
```

`main` contains the application and research entry points. `deploy` contains the standalone Docker build, smoke test, and Locust suite. The `experiment1` branch preserves the study and its reproduction guide.

## Quick start

From the repository root, use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-dev
uv run --no-sync uvicorn cheatdetect.app.app:app --reload
```

Open [local Swagger UI](http://127.0.0.1:8000/docs) or check `GET /health`. Run the project tests with `uv run pytest`.

## Prediction API

Send a JSON batch to `POST /predict`. Reuse the same `session_id` and keep event timestamps increasing across batches.

```json
{
  "session_id": "demo-student",
  "events": [
    {"time": 0.0, "event_type": "mousemove", "x": 100, "y": 200}
  ]
}
```

For a fresh session, a first batch of 25 events returns an empty `chunks_pred`. A second batch of 25 events produces one scored window. Each additional 25 events produces another window. Each prediction includes a score and an `isanomalous` flag. Use `chunks_pred` to tell whether any window has been scored.

## Deployment and benchmarking

Check out `deploy` for the Docker image that includes the trained model and configuration.

```bash
git switch deploy
docker build -t cheatdetect:deploy .
docker run --rm -p 8000:8000 cheatdetect:deploy
```

In a second terminal, run the smoke test against the container. The script also accepts the Railway base URL.

```bash
python3 scripts/smoke_test.py http://127.0.0.1:8000
```

On `deploy`, use the following to reproduce the 30-user workload against your own running instance. Set `API_URL` to its base URL. For the 10-user run, use `-u 10` and `-t 3m5s`.

```bash
uv run locust -f tests/locust/locustfile.py --host "$API_URL" \
  --headless -u 30 -r 2 -t 3m15s --reset-stats --stop-timeout 10
```
