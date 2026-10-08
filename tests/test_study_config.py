"""Frozen study configuration matches the predeclared protocol budgets."""

import dataclasses

import pytest

from cheatdetect.experiments.experiment_1 import config as study_config


def test_budget_table_matches_protocol():
    counts = study_config.validate_budgets()
    assert counts == study_config.PROTOCOL_BUDGETS
    assert counts["if_per_condition"] == 24
    assert counts["ocsvm_per_condition"] == 36
    assert counts["ae_per_condition"] == 42
    assert counts["main_fits"] == 204
    assert counts["repeat_fits"] == 8
    assert counts["total_fits"] == 212


def test_recipe_axes_are_frozen():
    assert study_config.RECIPE_NAMES == (
        "IF-raw",
        "IF-log",
        "SVM-base",
        "SVM-log",
        "SVM-YJ",
        "SVM-quantile",
        "AE-base",
        "AE-log",
        "AE-YJ",
    )
    assert study_config.IF_RECIPES == ("IF-raw", "IF-log")
    assert study_config.OCSVM_RECIPES == (
        "SVM-base",
        "SVM-log",
        "SVM-YJ",
        "SVM-quantile",
    )
    assert study_config.AE_RECIPES == ("AE-base", "AE-log", "AE-YJ")
    assert study_config.AUGMENTATION_CONDITIONS == (False, True)


def test_if_candidates_are_the_frozen_grid():
    candidates = study_config.if_candidates()
    assert len(candidates) == 24
    assert {candidate["contamination"] for candidate in candidates} == {"auto"}
    assert {candidate["max_samples"] for candidate in candidates} == {128, 256}
    assert {candidate["max_features"] for candidate in candidates} == {0.75, 1.0}
    assert {candidate["n_estimators"] for candidate in candidates} == {100, 300, 500}


def test_ocsvm_candidates_use_numeric_gamma_scales():
    centers = {name: 2.0 for name in study_config.OCSVM_RECIPES}
    candidates = study_config.ocsvm_candidates(centers)
    assert len(candidates) == 36
    assert {candidate["kernel"] for candidate in candidates} == {"rbf"}
    assert {candidate["tol"] for candidate in candidates} == {1e-3}
    assert {candidate["max_iter"] for candidate in candidates} == {-1}
    assert {candidate["gamma"] for candidate in candidates} == {0.5, 2.0, 8.0}
    assert {candidate["nu"] for candidate in candidates} == {0.01, 0.05, 0.10}


def test_ae_candidates_include_the_depth_block():
    candidates = study_config.ae_candidates()
    assert len(candidates) == 42
    main = [c for c in candidates if c["num_layers"] == 1]
    depth = [c for c in candidates if c["num_layers"] == 2]
    assert len(main) == 36
    assert len(depth) == 6
    assert {c["hidden_dim"] for c in main} == {16, 32}
    assert {c["batch_size"] for c in main} == {16, 32, 64}
    assert {c["lr"] for c in main} == {3e-4, 1e-3}
    assert {c["hidden_dim"] for c in depth} == {16, 32}
    assert {c["batch_size"] for c in depth} == {32}
    assert {c["lr"] for c in depth} == {1e-3}


def test_family_candidates_require_gamma_centers_for_ocsvm():
    with pytest.raises(ValueError, match="gamma"):
        study_config.family_candidates("ocsvm")
    with pytest.raises(ValueError, match="Unknown model family"):
        study_config.family_candidates("mystery")


def test_study_config_is_frozen():
    config = study_config.StudyConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.main_seed = 7
    assert config.main_seed == 42
    assert config.repeat_seeds == (123, 2026)
