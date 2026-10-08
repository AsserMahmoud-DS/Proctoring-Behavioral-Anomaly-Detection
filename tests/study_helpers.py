"""Tiny synthetic study environment shared by runner and reporting tests."""

import dataclasses

import numpy as np
import pandas as pd

from cheatdetect.data import dataset as data
from cheatdetect.experiments.experiment_1 import runner
from cheatdetect.experiments.experiment_1.config import StudyConfig

IF_CANDIDATES = [
    {"recipe": "IF-raw", "n_estimators": 5, "max_samples": 5, "max_features": 1.0,
     "contamination": "auto"},
]
OCSVM_CANDIDATES = [
    {"recipe": "SVM-base", "nu": 0.5, "gamma": 0.1, "kernel": "rbf",
     "tol": 1e-3, "max_iter": -1},
]
AE_CANDIDATES = [
    {"recipe": "AE-base", "hidden_dim": 4, "num_layers": 1, "batch_size": 4, "lr": 1e-2},
]


def raw_session(seed=42, cheating_window=None, n_events=75):
    """Raw session rows; ``cheating_window`` marks an event range as cheating."""
    rng = np.random.default_rng(seed)
    coordinates = 100 + np.cumsum(rng.normal(0, 2, (n_events, 2)), axis=0)
    cheating = np.zeros(n_events, dtype=bool)
    if cheating_window is not None:
        cheating[cheating_window[0]:cheating_window[1]] = True
    return pd.DataFrame({
        "Time (seconds)": np.arange(n_events) * 0.1,
        "Event Type": np.where(np.arange(n_events) % 2, "keydown", "mousemove"),
        "X Coordinate": coordinates[:, 0], "Y Coordinate": coordinates[:, 1],
        "Action": "", "Is Cheating": cheating,
    })


def tiny_config(**overrides) -> StudyConfig:
    base = dict(
        chunk_size=20,
        step_size=10,
        sub_chunk=5,
        sub_step=5,
        lstm_epochs=2,
        lstm_patience=1,
    )
    base.update(overrides)
    return dataclasses.replace(StudyConfig(), **base)


def build_tiny_study(tmp_path):
    """Write small raw sessions and build the frozen bundle in a tmp artifact root.

    Sets ``dataset.ARTIFACT_ROOT`` to the temporary root; callers are
    responsible for restoring it. Returns
    ``(study, config, split, normal_dir, mixed_dir, tmp_path)``.
    """
    data.ARTIFACT_ROOT = tmp_path / "artifacts"
    normal = tmp_path / "normal"
    mixed = tmp_path / "mixed"
    normal.mkdir()
    mixed.mkdir()
    split = {
        "normal_train": ["train.csv"],
        "normal_val": ["normal_val.csv"],
        "mixed_val": ["mixed_val.csv"],
        "mixed_test": ["test.csv"],
    }
    windows = {"mixed_val": (30, 60), "mixed_test": (20, 55)}
    for index, (key, names) in enumerate(split.items()):
        directory = normal if key.startswith("normal") else mixed
        raw_session(index, windows.get(key)).to_csv(directory / names[0], index=False)
    config = tiny_config()
    study = runner.build_study(
        config,
        split=split,
        normal_dir=normal,
        mixed_dir=mixed,
        output_dir=tmp_path / "artifacts" / "phase_03_study",
    )
    return study, config, split, normal, mixed, tmp_path
