"""Pipeline-level regression tests for the LSTM sequence data path."""

import numpy as np
import pandas as pd

import cheatdetect.pipeline.train as tr
from cheatdetect.config import ExperimentConfig


def _write_session(path, seed, cheating_window=None):
    rng = np.random.default_rng(seed)
    rows = []
    x, y, t = 100.0, 100.0, 0.0
    for i in range(120):
        event_type = "mousemove" if rng.random() < 0.7 else "keydown"
        if event_type == "mousemove":
            x = float(np.clip(x + rng.normal(0, 8), 0, 1920))
            y = float(np.clip(y + rng.normal(0, 8), 0, 1080))
        t += float(rng.uniform(0.02, 0.4))
        cheating = cheating_window is not None and (
            cheating_window[0] <= i < cheating_window[1]
        )
        rows.append(
            {
                "Time (seconds)": t,
                "Event Type": event_type,
                "X Coordinate": x,
                "Y Coordinate": y,
                "Action": "",
                "Is Cheating": "TRUE" if cheating else "FALSE",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _patch_paths(monkeypatch, base):
    for name, value in {
        "PROCESSED_DIR": base / "processed",
        "MODELS_DIR": base / "models",
        "REPORTS_DIR": base / "reports",
        "EDA_DIR": base / "reports" / "eda",
        "VAL_DIR": base / "reports" / "val",
        "TEST_DIR": base / "reports" / "test",
        "SPLIT_INFO_PATH": base / "processed" / "split_info.json",
    }.items():
        monkeypatch.setattr(tr, name, value)


def _setup_dataset(tmp_path, monkeypatch):
    """Write 4 normal + 4 mixed sessions (colliding basenames) and patch paths."""
    normal_dir = tmp_path / "normal"
    mixed_dir = tmp_path / "mixed"
    normal_dir.mkdir()
    mixed_dir.mkdir()
    for i in range(4):
        _write_session(normal_dir / f"s{i}.csv", seed=i, cheating_window=None)
        _write_session(
            mixed_dir / f"s{i}.csv", seed=i + 50, cheating_window=(60, 100)
        )
    monkeypatch.setattr(tr, "NORMAL_DIR", normal_dir)
    monkeypatch.setattr(tr, "MIXED_DIR", mixed_dir)
    _patch_paths(monkeypatch, tmp_path)


def test_lstm_sequences_align_with_flat_pipeline(tmp_path, monkeypatch):
    """Normal and mixed dirs share basenames; keys must not collide."""
    _setup_dataset(tmp_path, monkeypatch)

    config = ExperimentConfig(
        chunk_size=20,
        step_size=10,
        aug_enabled=False,
        normal_val_size=0.25,
        mixed_val_size=0.5,
        lstm_sub_chunk=5,
        lstm_sub_step=5,
    )

    data = tr.prepare_data(config)
    lstm = tr.prepare_lstm_data(config, data)

    pd.testing.assert_frame_equal(
        data["X_val_normal"].reset_index(drop=True),
        data["X_val"].iloc[: len(data["X_val_normal"])].reset_index(drop=True),
    )

    assert lstm["X_train"].shape[1] == 4  # (20 - 5) // 5 + 1
    assert lstm["X_val"].shape[0] == len(data["y_val"])
    assert lstm["X_test"].shape[0] == len(data["y_test"])

    # Alignment of labels is the core invariant.
    assert np.array_equal(lstm["y_val"], data["y_val"])
    assert np.array_equal(lstm["y_test"], data["y_test"])
    assert data["y_val"].sum() > 0
    assert data["y_test"].sum() > 0

    # Early-stopping set = held-out pure-normal sequences (all-zero labels).
    assert lstm["X_es"].shape[0] > 0
    assert lstm["X_es"].shape[1] == 4
    assert lstm["y_es"].sum() == 0
    assert len(lstm["y_es"]) == lstm["X_es"].shape[0]
    # It is the normal_val portion baked into the combined val (built first).
    assert lstm["X_es"].shape[0] < lstm["X_val"].shape[0]
    assert np.array_equal(lstm["X_es"], lstm["X_val"][: lstm["X_es"].shape[0]])


def test_train_pipeline_with_lstm_end_to_end(tmp_path, monkeypatch):
    """Full pipeline runs with the LSTM enabled; LSTM never becomes best_model."""
    _setup_dataset(tmp_path, monkeypatch)
    ensemble_search = tr.ensemble_grid_search
    calibrated = []

    def capture_ensemble(if_detector, ocsvm_detector, X_val, y_val, weights, *, X_ref):
        assert len(X_ref) == 11
        assert len(X_ref) < len(X_val)
        ensemble, results = ensemble_search(
            if_detector, ocsvm_detector, X_val, y_val, weights, X_ref=X_ref
        )
        assert ensemble.if_detector is if_detector
        assert ensemble.ocsvm_detector is ocsvm_detector
        expected_center = np.median(
            np.column_stack(
                [if_detector.decision_function(X_ref), ocsvm_detector.decision_function(X_ref)]
            ),
            axis=0,
        )
        np.testing.assert_allclose(ensemble.score_scaler.center_, expected_center)
        calibrated.append(ensemble)
        return ensemble, results

    monkeypatch.setattr(tr, "ensemble_grid_search", capture_ensemble)

    config = ExperimentConfig(
        chunk_size=20,
        step_size=10,
        aug_enabled=False,
        normal_val_size=0.25,
        mixed_val_size=0.5,
        if_n_estimators=(20,),
        # "auto" keeps the IF best-params row object-typed and avoids tripping
        # an unrelated grid_search dtype bug (tracked as BUG-1, Phase 2).
        if_max_samples=("auto",),
        if_contamination=(0.1,),
        ocsvm_nu=(0.1,),
        ocsvm_gamma=("scale",),
        ocsvm_kernel=("rbf",),
        ensemble_weights=(0.5,),
        lstm_enabled=True,
        lstm_sub_chunk=5,
        lstm_sub_step=5,
        lstm_hidden_dims=(4,),
        lstm_num_layers=(1,),
        lstm_dropouts=(0.0,),
        lstm_lrs=(1e-2,),
        lstm_batch_sizes=(8,),
        lstm_epochs=2,
        lstm_patience=1,
    )

    results = tr.train_pipeline(config)

    assert "LSTM-AE" in results["metrics_df"].index
    assert results["best_name"] in {"IF", "OCSVM", "Ensemble"}
    assert results["lstm_grid_results"] is not None
    assert len(calibrated) == 1
    assert "LSTM-AE" in results["val_scores"]
    assert (tmp_path / "models" / "best_model.joblib").exists()
