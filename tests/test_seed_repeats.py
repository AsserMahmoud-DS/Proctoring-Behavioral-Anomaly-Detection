"""Phase 5 repeats the locked IF/AE finalists without selecting a best seed."""

import hashlib
import json

import joblib
import pytest

import cheatdetect.experiments.experiment_1.runner as runner

from study_helpers import AE_CANDIDATES, IF_CANDIDATES


@pytest.fixture
def study_env(tiny_study):
    return tiny_study


def _artifact_hashes(run) -> dict:
    return {
        key: hashlib.sha256(winner.artifact.read_bytes()).hexdigest()
        for key, winner in run.winners.items()
    }


def test_repeats_are_fixed_seed_and_never_overwrite_seed42(study_env):
    study, config, *_, tmp_path = study_env
    run = runner.run_search(
        study,
        config,
        candidates={"if": IF_CANDIDATES, "ae": AE_CANDIDATES},
        output_dir=tmp_path / "artifacts" / "phase_04_study",
    )
    primary = json.loads(
        (run.directory / "reporting_manifest.json").read_text()
    )["primary"]
    before = _artifact_hashes(run)

    repeats = runner.run_seed_repeats(
        study,
        config,
        run_dir=run.directory,
        output_dir=tmp_path / "artifacts" / "phase_05_seed_sensitivity",
    )

    assert len(repeats) == 8  # 2 families x 2 conditions x 2 seeds
    assert {winner.seed for winner in repeats.values()} == {123, 2026}
    assert len({winner.id for winner in repeats.values()}) == 8
    for winner in repeats.values():
        assert winner.artifact.is_file()
        assert winner.artifact.parent.name == "phase_05_seed_sensitivity"
        assert winner.detector.random_state == winner.seed
        assert winner.threshold_info["threshold"] == pytest.approx(winner.threshold)
        assert joblib.hash(winner.detector.preprocessor) == joblib.hash(
            study.preprocessors[winner.recipe]
        )

    # Seed-42 artifacts stay byte-identical, and the primary manifest is frozen.
    assert _artifact_hashes(run) == before
    updated = json.loads((run.directory / "reporting_manifest.json").read_text())
    assert updated["primary"] == primary
    assert updated["sensitivity_frozen"] is True
    assert len(updated["sensitivity"]) == 8
    assert all(entry["seed"] in config.repeat_seeds for entry in updated["sensitivity"])


def test_repeats_require_a_frozen_manifest(study_env):
    study, config, *_, tmp_path = study_env
    with pytest.raises(FileNotFoundError, match="manifest"):
        runner.run_seed_repeats(
            study,
            config,
            run_dir=tmp_path / "artifacts" / "missing_run",
        )
