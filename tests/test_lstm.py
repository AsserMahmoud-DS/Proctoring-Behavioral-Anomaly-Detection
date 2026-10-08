"""Tests for the LSTM autoencoder sequence detector."""

import numpy as np

from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.models.base import SequenceAnomalyDetector
from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector

from synthetic import make_source_sequences

SEQ_LEN = 3
N_FEATURES = len(SOURCE_FEATURES)
FEATURES = list(SOURCE_FEATURES)


def _make_data(n: int, seed: int) -> np.ndarray:
    return make_source_sequences(n, SEQ_LEN, seed)


def _detector(**overrides) -> LSTMAutoencoderDetector:
    kwargs = dict(
        feature_names=FEATURES,
        input_dim=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=4,
        num_layers=1,
        latent_dropout=0.0,
        lr=1e-2,
        batch_size=8,
        epochs=2,
        patience=1,
        random_state=0,
    )
    kwargs.update(overrides)
    return LSTMAutoencoderDetector(**kwargs)


def test_detector_implements_sequence_interface():
    assert isinstance(_detector(), SequenceAnomalyDetector)


def test_default_recipe_is_base():
    detector = _detector()
    assert detector.recipe == "base"
    assert detector.preprocessor.recipe == "base"


def test_unfitted_detector_raises():
    detector = _detector()
    with np.testing.assert_raises(RuntimeError):
        detector.decision_function(_make_data(4, 0))


def test_fit_with_es_uses_early_stopping():
    X_train = _make_data(16, 0)
    X_es = _make_data(8, 1)

    detector = _detector(epochs=5, patience=2).fit(X_train, X_es=X_es)
    scores = detector.decision_function(X_es)

    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
    assert detector.model.best_epoch is not None
    assert detector.model.epochs_trained <= 5


def test_fit_without_es_runs_full_epochs():
    X_train = _make_data(12, 0)
    detector = _detector(epochs=3).fit(X_train)

    assert detector.decision_function(X_train).shape == (12,)
    assert detector.model.best_epoch is None
    assert detector.model.epochs_trained == 3


def test_grid_search_returns_sorted_results():
    X_train = _make_data(16, 0)
    X_val = _make_data(12, 1)
    X_es = _make_data(8, 2)
    y_val = np.array([0, 1] * 6)

    best, results = LSTMAutoencoderDetector.grid_search(
        X_train,
        X_val,
        y_val,
        X_es,
        feature_names=FEATURES,
        param_grid={
            "hidden_dim": [4, 8],
            "num_layers": [1],
            "latent_dropout": [0.0],
            "lr": [1e-2],
            "batch_size": [8],
        },
        fixed_kwargs={
            "input_dim": N_FEATURES,
            "seq_len": SEQ_LEN,
            "epochs": 2,
            "patience": 1,
        },
        random_state=0,
    )

    assert isinstance(best, LSTMAutoencoderDetector)
    assert len(results) == 2
    expected_columns = {
        "hidden_dim", "num_layers", "latent_dropout", "lr", "batch_size",
        "roc_auc", "pr_auc", "epochs_trained", "best_epoch",
        "optimizer_updates", "best_es_loss",
    }
    assert expected_columns.issubset(results.columns)
    assert results["roc_auc"].is_monotonic_decreasing
    # Best detector corresponds to the top row of the sorted results.
    assert best.hidden_dim == results.iloc[0]["hidden_dim"]
