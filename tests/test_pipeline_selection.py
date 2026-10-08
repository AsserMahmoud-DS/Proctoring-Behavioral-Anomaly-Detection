"""Ensure validation locks the deployable winner before test evaluation."""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import cheatdetect.pipeline.report as report_mod
import cheatdetect.pipeline.search as search_mod
import cheatdetect.pipeline.train as tr
from cheatdetect.app.schemas import InferenceConfig
from cheatdetect.config import ExperimentConfig
from cheatdetect.models.base import AnomalyDetector


class ScoredDetector(AnomalyDetector):
    """Return fixed scores for synthetic samples without model training."""

    def __init__(self, name, validation_scores, test_scores):
        self.name = name
        self.scores = dict(zip(range(5), range(5), strict=True))
        self.scores.update(zip(range(10, 10 + len(validation_scores)), validation_scores, strict=True))
        self.scores.update(zip(range(20, 24), test_scores, strict=True))

    def fit(self, X):
        raise AssertionError("Selected detectors must not be retrained")

    def decision_function(self, X):
        return X["sample"].map(self.scores).to_numpy(dtype=float)


class FixedTestDetector(AnomalyDetector):
    """Return precomputed test scores; never refit."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def fit(self, X):
        raise AssertionError("Research detectors must not be refit")

    def decision_function(self, X):
        return self._scores


def test_search_module_never_references_test_data():
    """The validation-only search must not name test partitions or matrices."""
    tree = ast.parse(Path(search_mod.__file__).read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {"X_test", "y_test", "mixed_test", "TEST_MIXED_PATH", "TEST_DIR"}
    assert not (names & forbidden)


@pytest.mark.parametrize(
    "if_validation, svm_validation, validation_labels, expected_winner",
    [
        ([0, 4, 3, 5], [4, 0, 5, 3], [0, 0, 1, 1], "Ensemble"),
        ([0, 4, 3, 5], [0, 4, 3, 5], [0, 0, 1, 1], "IF"),
        ([1, 2, 3, 4, 0, 5], [0, 1, 2, 5, 3, 4], [0, 0, 0, 0, 1, 1], "OCSVM"),
    ],
)
@pytest.mark.parametrize("test_labels", [[0, 0, 1, 1], [1, 1, 0, 0]])
def test_winner_uses_validation_and_excludes_lstm(
    tmp_path, monkeypatch, if_validation, svm_validation, validation_labels,
    expected_winner, test_labels,
):
    data = {
        "X_train": pd.DataFrame({"sample": range(5)}),
        "X_val_normal": pd.DataFrame({"sample": range(5)}),
        "X_val": pd.DataFrame({"sample": range(10, 10 + len(validation_labels))}),
        "y_val": np.array(validation_labels),
        "X_test": pd.DataFrame({"sample": range(20, 24)}),
        "y_test": np.array(test_labels),
        "features_to_keep": ["sample"],
    }
    if_detector = ScoredDetector("IF", if_validation, [0, 1, 3, 4])
    ocsvm_detector = ScoredDetector("OCSVM", svm_validation, [10, 10, 0, 0])
    monkeypatch.setattr(tr, "prepare_data", lambda config: data)
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)

    def selected(detector):
        def search(X_train, X_val, y_val, *args, **kwargs):
            assert X_train is data["X_train"]
            assert X_val is data["X_val"]
            assert y_val is data["y_val"]
            return detector, pd.DataFrame()

        return search

    monkeypatch.setattr(
        search_mod.IsolationForestDetector, "grid_search", selected(if_detector)
    )
    monkeypatch.setattr(
        search_mod.OCSVMDetector, "grid_search", selected(ocsvm_detector)
    )
    selection_calls = []

    def validation_metric(y_true, scores):
        assert y_true is data["y_val"]
        selection_calls.append(scores.copy())
        return roc_auc_score(y_true, scores)

    monkeypatch.setattr(search_mod, "roc_auc_score", validation_metric)
    evaluate = report_mod.evaluate_model

    def evaluate_after_selection(name, scores, threshold, y_true):
        assert len(selection_calls) == 3
        assert y_true is data["y_test"]
        return evaluate(name, scores, threshold, y_true)

    monkeypatch.setattr(report_mod, "evaluate_model", evaluate_after_selection)

    def fake_lstm_data(config, prepared):
        assert prepared is data
        return {
            "X_test": data["X_test"],
            "y_val": data["y_val"],
            "y_test": data["y_test"],
        }

    monkeypatch.setattr(tr, "prepare_lstm_data", fake_lstm_data)

    def research_lstm(config, lstm_data):
        assert lstm_data["y_val"] is data["y_val"]
        saved = json.loads((tmp_path / "model_config.json").read_text())
        assert saved["model"] == expected_winner
        return {
            "detector": FixedTestDetector(data["y_test"].astype(float)),
            "val_scores": data["y_val"].astype(float),
            "threshold": 0.5,
            "grid_results": pd.DataFrame(),
        }

    monkeypatch.setattr(tr, "search_lstm", research_lstm)
    results = tr.train_pipeline(ExperimentConfig(lstm_enabled=True, ensemble_weights=(0.5,)))

    assert results["best_name"] == expected_winner
    assert results["selection_metric"] == "roc_auc"
    expected_score = roc_auc_score(
        data["y_val"], results["val_scores"][expected_winner]
    )
    assert results["selection_score_val"] == pytest.approx(expected_score)
    saved = json.loads((tmp_path / "model_config.json").read_text())
    assert saved["model"] == expected_winner
    assert saved["selection_metric"] == "roc_auc"
    assert saved["selection_score_val"] == pytest.approx(expected_score)
    assert InferenceConfig(**saved).model == expected_winner
    assert saved["pr_auc_test"] == pytest.approx(
        results["metrics_df"].loc[expected_winner, "pr_auc"]
    )
    assert "LSTM-AE" in results["metrics_df"].index
    assert "LSTM-AE" in results["val_scores"]
    if expected_winner == "IF":
        assert results["best_detector"] is if_detector
        assert results["val_scores"]["LSTM-AE"].tolist() == data["y_val"].tolist()
        assert expected_score < roc_auc_score(
            data["y_val"], results["val_scores"]["LSTM-AE"]
        )
    elif expected_winner == "OCSVM":
        assert results["best_detector"] is ocsvm_detector
        assert average_precision_score(data["y_val"], results["val_scores"]["IF"]) > average_precision_score(
            data["y_val"], results["val_scores"]["OCSVM"]
        )
    if expected_winner != "IF" and test_labels == [0, 0, 1, 1]:
        assert results["metrics_df"].loc["IF", "pr_auc"] > saved["pr_auc_test"]
    restored = joblib.load(tmp_path / "best_model.joblib")
    np.testing.assert_allclose(
        restored.decision_function(data["X_test"]),
        results["best_detector"].decision_function(data["X_test"]),
    )
