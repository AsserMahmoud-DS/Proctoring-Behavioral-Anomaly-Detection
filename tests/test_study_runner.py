"""Validation-only study runner: frozen preprocessing, lock, and isolation."""

import ast
import dataclasses
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import cheatdetect.experiments.experiment_1.runner as runner
from cheatdetect.data.feature_schema import SOURCE_FEATURES

from study_helpers import AE_CANDIDATES, IF_CANDIDATES, OCSVM_CANDIDATES


@pytest.fixture
def study_env(tiny_study):
    return tiny_study


def run_tiny(study_env, candidates, output_name="phase_04_study"):
    study, config, split, normal, mixed, tmp_path = study_env
    return runner.run_search(
        study,
        config,
        candidates=candidates,
        output_dir=tmp_path / "artifacts" / output_name,
    )


def test_run_search_locks_recipe_winners_and_ensembles(study_env):
    run = run_tiny(
        study_env, {"if": IF_CANDIDATES, "ocsvm": OCSVM_CANDIDATES, "ae": AE_CANDIDATES}
    )
    study, config, *_ = study_env
    assert set(run.winners) == {
        "if:IF-raw:aug_off", "if:IF-raw:aug_on",
        "ocsvm:SVM-base:aug_off", "ocsvm:SVM-base:aug_on",
        "ae:AE-base:aug_off", "ae:AE-base:aug_on",
    }
    assert set(run.ensembles) == {"ensemble:IF-raw+SVM-base:aug_off",
                                  "ensemble:IF-raw+SVM-base:aug_on"}
    assert len(run.manifest["primary"]) == 8
    assert run.manifest["frozen"] is True
    assert (run.directory / "candidates.csv").is_file()
    assert (run.directory / "reporting_manifest.json").is_file()
    assert {"n_iter", "converged", "seconds", "pr_auc_baseline"} <= set(
        run.candidates.columns
    )

    for winner in list(run.winners.values()) + list(run.ensembles.values()):
        assert winner.artifact.is_file()
        restored = joblib.load(winner.artifact)
        assert np.isfinite(winner.threshold)
        assert winner.threshold_info["threshold"] == pytest.approx(winner.threshold)
        np.testing.assert_allclose(
            restored.decision_function(
                np.zeros((3, 20, 25)) if winner.family == "ae" else pd.DataFrame(
                    np.zeros((3, len(SOURCE_FEATURES))), columns=SOURCE_FEATURES
                )
            ).shape,
            (3,),
        )

    off = run.winners["if:IF-raw:aug_off"]
    on = run.winners["if:IF-raw:aug_on"]
    assert off.params == on.params
    assert off.artifact != on.artifact
    assert run.manifest["study_config"]["chunk_size"] == 20
    assert run.manifest["split"] == study.manifest["split"]


def test_frozen_preprocessing_is_reused_for_augmented_training(study_env):
    run = run_tiny(study_env, {"if": IF_CANDIDATES})
    study, config, *_ = study_env
    winner = run.winners["if:IF-raw:aug_on"]
    views = runner.validation_views(study, augmented=True)

    assert joblib.hash(winner.detector.preprocessor) == joblib.hash(
        study.preprocessors["IF-raw"]
    )
    pd.testing.assert_frame_equal(
        winner.detector.preprocessor.transform(views.flat_train),
        study.training("IF-raw", True),
    )
    frozen_val = pd.concat(
        [
            study.transformed["IF-raw"]["normal_val"],
            study.transformed["IF-raw"]["mixed_val"],
        ],
        ignore_index=True,
    )
    model = winner.detector.pipeline.named_steps["model"]
    np.testing.assert_allclose(
        winner.detector.decision_function(views.flat_val),
        -model.decision_function(frozen_val),
    )


def test_preprocessing_state_is_unchanged_by_a_full_run(study_env):
    study, config, *_ = study_env
    before = joblib.hash(study.preprocessors)
    runner.run_search(
        study,
        config,
        candidates={"if": IF_CANDIDATES, "ocsvm": OCSVM_CANDIDATES},
        output_dir=study_env[-1] / "artifacts" / "repeat_run",
    )
    assert joblib.hash(study.preprocessors) == before


def test_failed_candidate_is_recorded_and_not_substituted(study_env):
    candidates = {
        "ocsvm": [
            {"recipe": "SVM-base", "nu": 2.0, "gamma": 0.1, "kernel": "rbf",
             "tol": 1e-3, "max_iter": -1},
            *OCSVM_CANDIDATES,
        ]
    }
    run = run_tiny(study_env, candidates)
    assert len(run.winners) == 2
    failed = run.candidates[run.candidates["status"] == "failed"]
    assert len(failed) == 2
    assert failed["error"].str.contains("nu").all()
    assert run.candidates["status"].eq("ok").sum() == 2


def test_ae_candidates_train_and_are_locked(study_env):
    run = run_tiny(study_env, {"ae": AE_CANDIDATES})
    assert len(run.winners) == 2
    assert len(run.ensembles) == 0
    assert {"epochs_trained", "best_epoch", "optimizer_updates"} <= set(
        run.candidates.columns
    )
    for winner in run.winners.values():
        assert winner.detector.model.epochs_trained <= 2
        assert winner.detector.latent_dropout == 0.1
        assert winner.detector.lstm_dropout == 0.0


def test_runner_rejects_outputs_outside_study_artifacts(study_env):
    study, config, *_ = study_env
    with pytest.raises(ValueError, match="study/artifacts"):
        runner.run_search(
            study,
            config,
            candidates={"if": IF_CANDIDATES},
            output_dir=study_env[-1] / "production_models",
        )


def test_load_study_rejects_a_stale_manifest(study_env):
    study, config, split, normal, mixed, tmp_path = study_env
    with pytest.raises(ValueError, match="manifest"):
        runner.load_study(
            dataclasses.replace(config, chunk_size=30),
            split=split,
            normal_dir=normal,
            mixed_dir=mixed,
            output_dir=tmp_path / "artifacts" / "phase_03_study",
        )


def test_runner_module_never_references_test_or_production_paths():
    tree = ast.parse(Path(runner.__file__).read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {
        "X_test", "y_test", "mixed_test", "TEST_DIR", "TEST_MIXED_PATH",
        "MODELS_DIR", "REPORTS_DIR",
    }
    assert not (names & forbidden)
    assert not (attrs & forbidden)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "cheatdetect.pipeline.report" not in imported
