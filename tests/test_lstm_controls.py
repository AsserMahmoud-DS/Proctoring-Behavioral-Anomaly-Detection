"""Frozen LSTM-AE controls: dropout split, optimizer safeguards, diagnostics."""

import numpy as np
import pytest
import torch

from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.models.lstm_ae import (
    LSTMAutoencoder,
    LSTMAutoencoderDetector,
    _lstm_anomaly_scores,
)

from synthetic import make_source_sequences

FEATURES = list(SOURCE_FEATURES)
SEQ_LEN = 3


def _detector(**overrides) -> LSTMAutoencoderDetector:
    kwargs = dict(
        feature_names=FEATURES,
        input_dim=len(SOURCE_FEATURES),
        seq_len=SEQ_LEN,
        hidden_dim=4,
        num_layers=1,
        latent_dropout=0.0,
        lr=1e-2,
        batch_size=8,
        epochs=5,
        patience=2,
        random_state=0,
    )
    kwargs.update(overrides)
    return LSTMAutoencoderDetector(**kwargs)


def test_dropout_layers_are_separate():
    stacked = LSTMAutoencoder(5, 4, num_layers=2, latent_dropout=0.3, lstm_dropout=0.25)
    assert stacked.encoder.dropout == pytest.approx(0.25)
    assert stacked.decoder.dropout == pytest.approx(0.25)
    assert stacked.latent_dropout.p == pytest.approx(0.3)

    single = LSTMAutoencoder(5, 4, num_layers=1, latent_dropout=0.3, lstm_dropout=0.25)
    assert single.encoder.dropout == 0.0
    assert single.decoder.dropout == 0.0
    assert single.latent_dropout.p == pytest.approx(0.3)


def test_optimizer_and_gradient_clipping_controls_are_applied(monkeypatch):
    captured = {}
    real_adam = torch.optim.Adam

    def spy_adam(params, **kwargs):
        captured.update(kwargs)
        return real_adam(params, **kwargs)

    clipped = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, **kwargs):
        clipped.append(max_norm)
        return real_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    detector = _detector(weight_decay=0.5, grad_clip=1.0, epochs=2, patience=1)
    detector.fit(
        make_source_sequences(8, SEQ_LEN, seed=0),
        X_es=make_source_sequences(4, SEQ_LEN, seed=1),
    )

    assert captured["weight_decay"] == pytest.approx(0.5)
    assert clipped and set(clipped) == {1.0}


def test_relative_improvement_tolerance_stops_early():
    detector = _detector(min_improvement=1.0, patience=1, epochs=20)
    detector.fit(
        make_source_sequences(8, SEQ_LEN, seed=0),
        X_es=make_source_sequences(4, SEQ_LEN, seed=1),
    )

    # A 100% relative improvement is impossible, so only epoch 0 can be best
    # and patience (1) stops training after the next epoch.
    assert detector.model.best_epoch == 0
    assert detector.model.epochs_trained == 2
    assert detector.model.best_es_loss is not None


def test_best_checkpoint_is_restored():
    X_es = make_source_sequences(8, SEQ_LEN, seed=1)
    detector = _detector(min_improvement=1.0, patience=1, epochs=10)
    detector.fit(make_source_sequences(16, SEQ_LEN, seed=0), X_es=X_es)

    scaled = detector._preprocess(X_es, fit=False)
    restored_loss = float(np.mean(_lstm_anomaly_scores(detector.model, scaled)))
    assert restored_loss == pytest.approx(detector.model.best_es_loss, rel=1e-5)


def test_full_budget_mode_records_updates_and_no_checkpoint():
    detector = _detector(epochs=3).fit(make_source_sequences(8, SEQ_LEN, seed=0))

    assert detector.model.epochs_trained == 3
    assert detector.model.optimizer_updates == 3  # one batch per epoch
    assert detector.model.best_epoch is None
    assert detector.model.best_es_loss is None


def test_grid_search_reports_training_diagnostics():
    best, results = LSTMAutoencoderDetector.grid_search(
        make_source_sequences(8, SEQ_LEN, seed=0),
        make_source_sequences(8, SEQ_LEN, seed=1),
        np.array([0, 1] * 4),
        make_source_sequences(4, SEQ_LEN, seed=2),
        feature_names=FEATURES,
        param_grid={
            "hidden_dim": [4],
            "latent_dropout": [0.0],
            "lr": [1e-2],
            "batch_size": [8],
        },
        fixed_kwargs={
            "input_dim": len(SOURCE_FEATURES),
            "seq_len": SEQ_LEN,
            "epochs": 3,
            "patience": 1,
        },
        random_state=0,
    )

    row = results.iloc[0]
    assert row["optimizer_updates"] == row["epochs_trained"]  # one batch per epoch
    assert row["best_es_loss"] is not None
    assert row["best_epoch"] == best.model.best_epoch
    assert best.model.optimizer_updates == row["optimizer_updates"]
