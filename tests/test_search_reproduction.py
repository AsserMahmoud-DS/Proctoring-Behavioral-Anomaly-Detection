"""The winner found by validation search must survive through train_pipeline."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

import cheatdetect.pipeline.search as search_mod
import cheatdetect.pipeline.train as tr
from cheatdetect.config import ExperimentConfig
from cheatdetect.data.preprocessing import FeaturePreprocessor

from synthetic import make_source_frame


def _prepared_frames() -> dict:
    """Deterministic fixed-schema matrices shaped like prepare_data's output."""
    X_train = make_source_frame(60, seed=1)
    X_val_normal = make_source_frame(20, seed=2)
    X_val_mixed = make_source_frame(20, seed=3)
    return {
        "X_train": X_train,
        "X_val_normal": X_val_normal,
        "X_val": pd.concat([X_val_normal, X_val_mixed], ignore_index=True),
        "y_val": np.concatenate([np.zeros(20, dtype=int), np.array([0, 1] * 10)]),
        "X_test": make_source_frame(30, seed=4),
        "y_test": np.array([0] * 15 + [1] * 15),
        "features_to_keep": ["elapsed_time"],
    }


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        if_n_estimators=(10, 20),
        if_max_samples=(0.8,),
        if_contamination=(0.1,),
        ocsvm_nu=(0.1,),
        ocsvm_gamma=("scale",),
        ocsvm_kernel=("rbf",),
        ensemble_weights=(0.0, 0.5, 1.0),
        random_state=42,
    )


def test_search_winner_reproduces_through_train_pipeline(tmp_path, monkeypatch):
    data = _prepared_frames()
    config = _config()
    monkeypatch.setattr(tr, "prepare_data", lambda config: data)
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)

    search = search_mod.search_validation(
        data["X_train"], data["X_val"], data["X_val_normal"], data["y_val"], config
    )
    results = tr.train_pipeline(config)

    assert results["best_name"] == search["best_name"]
    assert results["best_threshold"] == pytest.approx(search["best_threshold"])
    assert results["selection_score_val"] == pytest.approx(
        search["selection_scores"][search["best_name"]]
    )
    np.testing.assert_allclose(
        results["val_scores"][search["best_name"]],
        search["val_scores"][search["best_name"]],
    )

    saved = json.loads((tmp_path / "model_config.json").read_text())
    assert saved["model"] == search["best_name"]
    assert saved["threshold"] == pytest.approx(search["best_threshold"])

    restored = joblib.load(tmp_path / "best_model.joblib")
    np.testing.assert_allclose(
        restored.decision_function(data["X_test"]),
        search["best_detector"].decision_function(data["X_test"]),
    )
    components = (
        [restored.if_detector, restored.ocsvm_detector]
        if search["best_name"] == "Ensemble"
        else [restored]
    )
    for detector in components:
        assert isinstance(
            detector.pipeline.named_steps["preprocess"], FeaturePreprocessor
        )
