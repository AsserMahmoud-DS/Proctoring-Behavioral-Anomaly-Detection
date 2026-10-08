"""Final test reporting (Study Protocol Phase 6), gated by the frozen manifest.

Reporting never selects models: it loads the artifacts named by the frozen
Phase 4/5 manifest, scores the held-out test split once, and writes metric
tables under ``study/artifacts/``. Running it before the manifest is frozen —
or before the Phase 5 seed repeats are appended — fails closed.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cheatdetect.data import dataset
from cheatdetect.pipeline.report import evaluate_scores, summarize

from .runner import DEFAULT_RUN_DIR, load_frozen_manifest

logger = logging.getLogger(__name__)

REPORT_DIR_NAME = "phase_06_reporting"


def _test_views(study) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Held-out flat parents, sequences, and labels for the final report."""
    raw = study.raw["mixed_test"]
    return raw.parent, raw.sequences, raw.labels


def _load_artifact(entry: dict):
    path = Path(entry["artifact"]).resolve()
    if not path.is_relative_to(dataset.ARTIFACT_ROOT.resolve()):
        raise ValueError("Reporting artifacts must stay inside study/artifacts")
    return joblib.load(path)


def _public(result: dict) -> dict:
    """Scalar-only view of an ``evaluate_model`` result for JSON output."""
    row = {key: value for key, value in result.items() if key not in {"cm", "scores"}}
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }


def run_phase6(
    study,
    run_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Score the locked approaches on the held-out test split.

    Args:
        study: Manifest-checked prepared bundle (source of frozen test views).
        run_dir: Phase 4 run directory holding ``reporting_manifest.json``.
        output_dir: Reporting output directory under ``study/artifacts/``.

    Returns:
        ``(results, summary)``: per-approach metric dicts (with score arrays
        for plotting) and the scalar summary table indexed by approach id.
    """
    run_dir = dataset.study_output_directory(run_dir or DEFAULT_RUN_DIR)
    output_dir = dataset.study_output_directory(
        output_dir or run_dir / REPORT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_frozen_manifest(run_dir)
    if manifest.get("study_manifest_version") != study.manifest.get("version"):
        raise ValueError("Reporting manifest does not match the prepared study bundle")
    if not manifest.get("sensitivity"):
        raise ValueError(
            "Frozen manifest has no Phase 5 sensitivity artifacts; "
            "run the seed repeats first"
        )

    entries = list(manifest["primary"]) + list(manifest["sensitivity"])
    flat_x, sequence_x, y_test = _test_views(study)

    results = []
    for entry in entries:
        detector = _load_artifact(entry)
        X = sequence_x if entry["family"] == "ae" else flat_x
        result = evaluate_scores(
            entry["id"],
            detector.decision_function(X),
            float(entry["threshold"]),
            y_test,
        )
        result.update(
            {
                "family": entry["family"],
                "recipe": entry["recipe"],
                "augmented": entry["augmented"],
                "seed": entry.get("seed"),
            }
        )
        results.append(result)

    summary = summarize(results)
    summary.to_csv(output_dir / "test_results.csv")
    (output_dir / "test_results.json").write_text(
        json.dumps([_public(result) for result in results], indent=2) + "\n"
    )
    logger.info("Reported %d approaches in %s", len(results), output_dir)
    return results, summary
