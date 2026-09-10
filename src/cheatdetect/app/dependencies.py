import json
from functools import lru_cache
from pathlib import Path

import joblib

from cheatdetect.app.schemas import InferenceConfig
from cheatdetect.app.service import PredictionService
from cheatdetect.models import AnomalyDetector
from cheatdetect.config import MODELS_DIR


@lru_cache
def get_predictor() -> PredictionService:
    """Load model artifacts once and return a PredictionService.

    Uses @lru_cache so the model is deserialized exactly once across
    all requests. Subsequent calls return the cached instance.
    """
    config_path = MODELS_DIR / "model_config.json"
    model_path = MODELS_DIR / "best_model.joblib"

    with open(config_path) as f:
        config = InferenceConfig(**json.load(f))

    detector: AnomalyDetector = joblib.load(model_path)
    return PredictionService(detector, config)