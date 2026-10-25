"""Comparison tables join matched candidates without refitting."""

import pandas as pd
import pytest

from cheatdetect.experiments.experiment_1 import analysis


def _row(family, recipe, augmented, roc_auc, pr_auc=0.5, **params):
    return {
        "family": family,
        "recipe": recipe,
        "augmented": augmented,
        "status": "ok",
        "error": "",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_baseline": 0.5,
        "seconds": 1.0,
        **params,
    }


def test_matched_augmentation_pairs_and_flags_incomplete():
    candidates = pd.DataFrame(
        [
            _row("if", "IF-raw", False, 0.60, max_samples=128, max_features=1.0,
                 n_estimators=100, contamination="auto"),
            _row("if", "IF-raw", True, 0.66, max_samples=128, max_features=1.0,
                 n_estimators=100, contamination="auto"),
            _row("if", "IF-raw", False, 0.55, max_samples=256, max_features=1.0,
                 n_estimators=100, contamination="auto"),
            _row("ocsvm", "SVM-base", False, 0.5, nu=0.1, gamma=0.01, kernel="rbf",
                 tol=1e-3, max_iter=-1),
            _row("ocsvm", "SVM-base", True, 0.7, nu=0.1, gamma=0.01, kernel="rbf",
                 tol=1e-3, max_iter=-1),
        ]
    )

    table = analysis.matched_augmentation_table(candidates)

    assert len(table) == 3  # two IF pairs (one incomplete) + one OCSVM pair
    paired = table[table["paired"]].set_index(["family", "recipe"])
    assert paired.loc[("if", "IF-raw"), "delta_roc_auc"] == pytest.approx(0.06)
    assert paired.loc[("ocsvm", "SVM-base"), "delta_roc_auc"] == pytest.approx(0.2)
    assert paired.loc[("ocsvm", "SVM-base"), "delta_pr_auc"] == pytest.approx(0.0)
    incomplete = table[~table["paired"]]
    assert len(incomplete) == 1
    assert incomplete.iloc[0]["max_samples"] == 256
    assert pd.isna(incomplete.iloc[0]["roc_auc_on"])


def test_matched_augmentation_includes_ae_updates():
    candidates = pd.DataFrame(
        [
            _row("ae", "AE-base", False, 0.5, hidden_dim=16, num_layers=1,
                 batch_size=32, lr=1e-3, epochs_trained=5, optimizer_updates=100),
            _row("ae", "AE-base", True, 0.6, hidden_dim=16, num_layers=1,
                 batch_size=32, lr=1e-3, epochs_trained=7, optimizer_updates=140),
        ]
    )

    row = analysis.matched_augmentation_table(candidates).iloc[0]

    assert row["delta_roc_auc"] == pytest.approx(0.1)
    assert row["epochs_trained_off"] == 5
    assert row["optimizer_updates_on"] == 140


def test_depth_comparison_is_matched_on_batch_lr_and_hidden():
    candidates = pd.DataFrame(
        [
            _row("ae", "AE-base", False, 0.5, hidden_dim=16, num_layers=1,
                 batch_size=32, lr=1e-3, epochs_trained=5, optimizer_updates=100,
                 best_es_loss=0.2),
            _row("ae", "AE-base", False, 0.58, hidden_dim=16, num_layers=2,
                 batch_size=32, lr=1e-3, epochs_trained=4, optimizer_updates=80,
                 best_es_loss=0.18),
            # Excluded: different batch/lr or augmentation state.
            _row("ae", "AE-base", False, 0.9, hidden_dim=16, num_layers=2,
                 batch_size=16, lr=1e-3, epochs_trained=4, optimizer_updates=80,
                 best_es_loss=0.1),
            _row("ae", "AE-base", True, 0.9, hidden_dim=16, num_layers=1,
                 batch_size=32, lr=1e-3, epochs_trained=4, optimizer_updates=80,
                 best_es_loss=0.1),
        ]
    )

    table = analysis.depth_comparison_table(candidates)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["delta_roc_auc"] == pytest.approx(0.08)
    assert row["epochs_trained_1layer"] == 5
    assert row["epochs_trained_2layer"] == 4


def test_augmentation_winners_compare_locked_recipes_only():
    manifest = {
        "primary": [
            {"family": "if", "recipe": "IF-raw", "augmented": False,
             "roc_auc": 0.60, "pr_auc": 0.40, "threshold": 0.1,
             "threshold_info": {"fallback_used": False}},
            {"family": "if", "recipe": "IF-raw", "augmented": True,
             "roc_auc": 0.65, "pr_auc": 0.44, "threshold": 0.2,
             "threshold_info": {"fallback_used": True}},
            {"family": "ensemble", "recipe": "IF-raw+SVM-base", "augmented": False,
             "roc_auc": 0.9, "pr_auc": 0.9, "threshold": 0.3,
             "threshold_info": {"fallback_used": False}},
        ]
    }

    table = analysis.augmentation_winners_table(manifest)

    assert len(table) == 1  # ensembles have no matched recipe across conditions
    row = table.iloc[0]
    assert row["delta_roc_auc"] == pytest.approx(0.05)
    assert bool(row["fallback_used_off"]) is False
    assert bool(row["fallback_used_on"]) is True
