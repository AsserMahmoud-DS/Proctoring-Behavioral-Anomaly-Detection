"""Selectors must prefer ROC-AUC even when average precision prefers another model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from cheatdetect.models.isolation_forest import IsolationForestDetector
from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector
from cheatdetect.models.ocsvm import OCSVMDetector


LABELS = np.array([0, 0, 0, 0, 1, 1])
PR_WINNER = np.array([1, 2, 3, 4, 0, 5], dtype=float)
ROC_WINNER = np.array([0, 1, 2, 5, 3, 4], dtype=float)


def run_search(model_name, monkeypatch, labels=LABELS, tied=False):
    training = pd.DataFrame({"feature": range(8)})
    validation = pd.DataFrame({"feature": range(6)})
    if model_name == "IF":
        detector_class = IsolationForestDetector
        parameter, candidates = "n_estimators", [10, 20]
        grid = {parameter: candidates, "max_samples": ["auto"]}
    elif model_name == "OCSVM":
        detector_class = OCSVMDetector
        parameter, candidates = "nu", [0.1, 0.2]
        grid = {parameter: candidates, "kernel": ["rbf"]}
    else:
        detector_class = LSTMAutoencoderDetector
        parameter, candidates = "hidden_dim", [4, 8]
        grid = {parameter: candidates}
        training = training.to_numpy().reshape(-1, 1, 1)
        validation = validation.to_numpy().reshape(-1, 1, 1)

    def fit(detector, X, X_es=None):
        assert X is training
        return detector

    def scores(detector, X):
        assert X is validation
        if tied or getattr(detector, parameter) == candidates[0]:
            return PR_WINNER
        return ROC_WINNER

    monkeypatch.setattr(detector_class, "fit", fit)
    monkeypatch.setattr(detector_class, "decision_function", scores)
    if model_name == "LSTM-AE":
        best, results = detector_class.grid_search(
            training, validation, labels, None, ["feature"], grid,
            fixed_kwargs={"input_dim": 1, "seq_len": 1},
        )
    else:
        best, results = detector_class.grid_search(training, validation, labels, [], grid)
    return best, results, parameter, candidates


@pytest.mark.parametrize("model_name", ["IF", "OCSVM", "LSTM-AE"])
def test_grid_search_selects_roc_auc_over_pr_auc(model_name, monkeypatch):
    assert roc_auc_score(LABELS, ROC_WINNER) > roc_auc_score(LABELS, PR_WINNER)
    assert average_precision_score(LABELS, ROC_WINNER) < average_precision_score(LABELS, PR_WINNER)
    best, results, parameter, candidates = run_search(model_name, monkeypatch)
    assert getattr(best, parameter) == candidates[1]
    assert results["roc_auc"].is_monotonic_decreasing
    assert results.iloc[0]["roc_auc"] == pytest.approx(0.75)
    assert results.iloc[0]["pr_auc"] == pytest.approx(average_precision_score(LABELS, ROC_WINNER))
    assert results.iloc[0]["pr_auc"] < results.iloc[1]["pr_auc"]


@pytest.mark.parametrize("model_name", ["IF", "OCSVM", "LSTM-AE"])
def test_grid_search_roc_auc_ties_keep_first_candidate(model_name, monkeypatch):
    best, results, parameter, candidates = run_search(model_name, monkeypatch, tied=True)
    assert getattr(best, parameter) == results.iloc[0][parameter] == candidates[0]


@pytest.mark.parametrize("model_name", ["IF", "OCSVM", "LSTM-AE"])
@pytest.mark.parametrize("labels", [np.zeros(6), np.ones(6)])
def test_grid_search_rejects_single_class_validation(model_name, labels, monkeypatch):
    with pytest.raises(ValueError, match="both normal"):
        run_search(model_name, monkeypatch, labels=labels)
