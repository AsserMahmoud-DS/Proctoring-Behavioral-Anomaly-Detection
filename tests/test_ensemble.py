"""Regression tests for fixed ensemble score scaling and validation tuning."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from cheatdetect.app.schemas import Event, InferenceConfig, PredictionRequest
from cheatdetect.app.service import PredictionService
from cheatdetect.models.base import AnomalyDetector
from cheatdetect.models.ensemble import EnsembleDetector, grid_search


class ColumnDetector(AnomalyDetector):
    """Deterministic scores with configurable units and fit-call tracking."""

    def __init__(self, column, scale=1.0, offset=0.0):
        self.column = column
        self.scale = scale
        self.offset = offset
        self.fit_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        return self

    def decision_function(self, X):
        return self.scale * X[self.column].to_numpy(dtype=float) + self.offset


@pytest.fixture
def reference():
    return pd.DataFrame({"if_score": [0, 1, 2, 3, 4], "svm_score": [0, 2, 4, 6, 8]})


def make_ensemble(reference, weight=0.5):
    return EnsembleDetector(
        ColumnDetector("if_score"), ColumnDetector("svm_score"), if_weight=weight
    ).fit_score_normalizer(reference)


def test_normalization_requires_fitting(reference):
    detector = EnsembleDetector(ColumnDetector("if_score"), ColumnDetector("svm_score"))
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.decision_function(reference)


def test_reference_statistics_and_weighted_scores(reference):
    detector = make_ensemble(reference, weight=0.25)
    np.testing.assert_allclose(detector.score_scaler.center_, [2, 4])
    np.testing.assert_allclose(detector.score_scaler.scale_, [2, 4])
    query = pd.DataFrame({"if_score": [6], "svm_score": [8]})
    np.testing.assert_allclose(detector.decision_function(query), [1.25])
    assert detector.if_detector.fit_calls == detector.ocsvm_detector.fit_calls == 0


def test_scores_are_independent_of_batch_and_order(reference):
    detector = make_ensemble(reference)
    query = pd.DataFrame({"if_score": [6, -10, 1e9], "svm_score": [8, -20, 1e12]})
    expected = detector.decision_function(query)
    individual = np.concatenate(
        [detector.decision_function(query.iloc[[index]]) for index in range(len(query))]
    )
    np.testing.assert_allclose(individual, expected)
    np.testing.assert_allclose(detector.decision_function(query.iloc[::-1]), expected[::-1])
    np.testing.assert_allclose(detector.score_scaler.center_, [2, 4])
    np.testing.assert_allclose(detector.score_scaler.scale_, [2, 4])


def test_normalized_scores_are_invariant_to_detector_units(reference):
    original = make_ensemble(reference)
    rescaled = EnsembleDetector(
        ColumnDetector("if_score", scale=100, offset=50),
        ColumnDetector("svm_score", scale=0.01, offset=-20),
    ).fit_score_normalizer(reference)
    query = pd.DataFrame({"if_score": [6, -10], "svm_score": [8, -20]})
    np.testing.assert_allclose(rescaled.decision_function(query), original.decision_function(query))


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 10],
        1 + np.arange(5) * np.finfo(float).eps,
    ],
)
def test_degenerate_reference_iqr_is_rejected(reference, values):
    reference["svm_score"] = values
    with pytest.raises(ValueError, match="reference IQR"):
        make_ensemble(reference)


def test_empty_and_nonfinite_reference_are_rejected(reference):
    with pytest.raises(ValueError, match="nonempty"):
        make_ensemble(reference.iloc[:0])
    reference.loc[0, "svm_score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_ensemble(reference)


def test_nonfinite_prediction_scores_are_rejected(reference):
    detector = make_ensemble(reference)
    query = pd.DataFrame({"if_score": [np.inf], "svm_score": [8]})
    with pytest.raises(ValueError, match="finite"):
        detector.decision_function(query)


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan])
def test_invalid_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="if_weight"):
        EnsembleDetector(ColumnDetector("if_score"), ColumnDetector("svm_score"), weight)


def test_fit_trains_detectors_and_fits_score_scaler(reference):
    detector = EnsembleDetector(ColumnDetector("if_score"), ColumnDetector("svm_score"))
    assert detector.fit(reference) is detector
    assert detector.if_detector.fit_calls == detector.ocsvm_detector.fit_calls == 1
    np.testing.assert_allclose(detector.score_scaler.center_, [2, 4])


def test_grid_search_keeps_selected_models_and_tunes_on_validation(reference):
    if_detector = ColumnDetector("if_score")
    ocsvm_detector = ColumnDetector("svm_score")
    validation = pd.DataFrame({"if_score": [0, 1, 3, 4], "svm_score": [8, 6, 2, 0]})
    labels = np.array([0, 0, 1, 1])
    best, results = grid_search(
        if_detector, ocsvm_detector, validation, labels, (0.0, 0.5, 1.0), X_ref=reference
    )
    assert best.if_detector is if_detector
    assert best.ocsvm_detector is ocsvm_detector
    assert if_detector.fit_calls == ocsvm_detector.fit_calls == 0
    assert best.if_weight == 1.0
    assert results["roc_auc"].is_monotonic_decreasing
    assert results.iloc[0]["roc_auc"] == roc_auc_score(
        labels, best.decision_function(validation)
    )
    assert results.iloc[0]["pr_auc"] == average_precision_score(
        labels, best.decision_function(validation)
    )
    np.testing.assert_allclose(best.score_scaler.center_, [2, 4])
    reversed_best, _ = grid_search(
        if_detector, ocsvm_detector, validation, 1 - labels, (0.0, 0.5, 1.0),
        X_ref=reference,
    )
    assert reversed_best.if_weight == 0.0


def test_grid_search_prefers_roc_auc_when_pr_auc_disagrees(reference):
    validation = pd.DataFrame({
        "if_score": [1, 2, 3, 4, 0, 5],
        "svm_score": [0, 1, 2, 5, 3, 4],
    })
    labels = np.array([0, 0, 0, 0, 1, 1])
    best, results = grid_search(
        ColumnDetector("if_score"), ColumnDetector("svm_score"),
        validation, labels, (1.0, 0.0), X_ref=reference,
    )
    assert best.if_weight == 0.0
    assert results["roc_auc"].is_monotonic_decreasing
    assert results.iloc[0]["pr_auc"] < results.iloc[1]["pr_auc"]


def test_grid_search_roc_auc_ties_keep_first_weight(reference):
    validation = pd.DataFrame({"if_score": [0, 1, 3, 4], "svm_score": [0, 2, 6, 8]})
    best, results = grid_search(
        ColumnDetector("if_score"), ColumnDetector("svm_score"),
        validation, np.array([0, 0, 1, 1]), (0.7, 0.3), X_ref=reference,
    )
    assert best.if_weight == results.iloc[0]["if_weight"] == 0.7


@pytest.mark.parametrize("labels", [np.zeros(5), np.ones(5)])
def test_grid_search_rejects_single_class_validation(reference, labels):
    with pytest.raises(ValueError, match="both normal"):
        grid_search(
            ColumnDetector("if_score"), ColumnDetector("svm_score"),
            reference, labels, (0.5,), X_ref=reference,
        )


@pytest.mark.parametrize("weights", [(), (-0.1,), (np.nan,)])
def test_grid_search_rejects_invalid_weights(reference, weights):
    with pytest.raises(ValueError, match="weights"):
        grid_search(
            ColumnDetector("if_score"), ColumnDetector("svm_score"),
            reference, np.array([0, 0, 0, 1, 1]), weights, X_ref=reference,
        )


def test_serialized_ensemble_retains_scores_and_predictions(reference, tmp_path):
    detector = make_ensemble(reference)
    query = pd.DataFrame({"if_score": [6, -10], "svm_score": [8, -20]})
    artifact = tmp_path / "ensemble.joblib"
    joblib.dump(detector, artifact)
    restored = joblib.load(artifact)
    np.testing.assert_allclose(restored.decision_function(query), detector.decision_function(query))
    np.testing.assert_array_equal(restored.predict(query, 0.5), detector.predict(query, 0.5))


def test_prediction_service_preserves_single_window_score(tmp_path):
    reference = pd.DataFrame({"elapsed_time": [0, 1, 2, 3, 4]})
    detector = EnsembleDetector(
        ColumnDetector("elapsed_time"), ColumnDetector("elapsed_time", scale=100, offset=50)
    ).fit_score_normalizer(reference)
    artifact = tmp_path / "ensemble.joblib"
    joblib.dump(detector, artifact)
    config = InferenceConfig(
        model="Ensemble", chunk_size=4, step_size=2, cheating_threshold=0.5,
        threshold=0.25,
    )
    events = [
        Event(time=float(index), event_type="mousemove", x=float(index), y=float(index))
        for index in range(8)
    ]
    singleton = PredictionService(joblib.load(artifact), config).predict(
        PredictionRequest(session_id="single", events=events[:4])
    )
    batch = PredictionService(joblib.load(artifact), config).predict(
        PredictionRequest(session_id="batch", events=events)
    )
    assert singleton.chunks_pred[0].score == pytest.approx(0.5)
    assert singleton.chunks_pred[0].score == batch.chunks_pred[0].score
    assert singleton.verdict == batch.verdict == "anomalous"
