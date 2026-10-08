"""Phase 6 reporting is gated by the frozen manifest and never selects."""

import ast
import json
from pathlib import Path

import pytest

from cheatdetect.experiments.experiment_1 import reporting, runner

from study_helpers import IF_CANDIDATES


@pytest.fixture
def study_env(tiny_study):
    return tiny_study


@pytest.fixture
def frozen_run(study_env):
    study, config, *_, tmp_path = study_env
    run = runner.run_search(
        study,
        config,
        candidates={"if": IF_CANDIDATES},
        output_dir=tmp_path / "artifacts" / "phase_04_study",
    )
    runner.run_seed_repeats(
        study,
        config,
        run_dir=run.directory,
        output_dir=tmp_path / "artifacts" / "phase_05_seed_sensitivity",
    )
    return study, config, run


def test_phase6_requires_a_frozen_manifest(study_env):
    study, config, *_, tmp_path = study_env
    with pytest.raises(FileNotFoundError, match="manifest"):
        reporting.run_phase6(study, run_dir=tmp_path / "artifacts" / "missing_run")


def test_phase6_requires_the_seed_repeats_first(study_env):
    study, config, *_, tmp_path = study_env
    run = runner.run_search(
        study,
        config,
        candidates={"if": IF_CANDIDATES},
        output_dir=tmp_path / "artifacts" / "phase_04_study",
    )
    with pytest.raises(ValueError, match="seed repeats"):
        reporting.run_phase6(study, run_dir=run.directory)


def test_phase6_reports_primary_and_sensitivity(frozen_run):
    study, config, run = frozen_run
    results, summary = reporting.run_phase6(study, run_dir=run.directory)

    assert len(results) == len(run.manifest["primary"]) + 4  # 2 IF cells x 2 seeds
    assert summary.index.is_unique
    assert {"roc_auc", "pr_auc", "pr_auc_baseline", "threshold"} <= set(summary.columns)
    output = run.directory / "phase_06_reporting"
    assert (output / "test_results.csv").is_file()
    payload = json.loads((output / "test_results.json").read_text())
    assert len(payload) == len(results)
    assert all("seed" in row for row in payload)


def test_phase6_rejects_a_mismatched_bundle(frozen_run):
    study, config, run = frozen_run
    manifest_path = run.directory / "reporting_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["study_manifest_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        reporting.run_phase6(study, run_dir=run.directory)


def test_reporting_module_never_selects_or_writes_production_paths():
    tree = ast.parse(Path(reporting.__file__).read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {"MODELS_DIR", "REPORTS_DIR", "TEST_DIR", "sweep_candidates"}
    assert not (names & forbidden)
    assert not (attrs & forbidden)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "cheatdetect.pipeline.search" not in imported
