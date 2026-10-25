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
from cheatdetect.eval import (
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distributions,
)
from cheatdetect.pipeline.report import evaluate_scores, summarize

from .runner import DEFAULT_RUN_DIR, load_frozen_manifest

logger = logging.getLogger(__name__)

REPORT_DIR_NAME = "phase_06_reporting"
FIGURE_DIR_NAME = "figures"

# PreparedStudy partitions map onto the frozen split keys for session counts.
_PARTITION_SPLIT = {
    "train_original": "normal_train",
    "train_synthetic": "normal_train",
    "normal_val": "normal_val",
    "mixed_val": "mixed_val",
    "mixed_test": "mixed_test",
}


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


def _params_frame(entries: list[dict]) -> pd.DataFrame:
    """Flatten locked manifest params into CSV-friendly columns per approach."""
    rows = []
    for entry in entries:
        row = {"id": entry["id"]}
        row.update({f"param_{key}": value for key, value in entry["params"].items()})
        # ``seed`` already comes from the scored results; avoid a join overlap.
        for key in ("if_recipe", "ocsvm_recipe", "if_weight"):
            row[key] = entry.get(key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def _dataset_summary(study) -> pd.DataFrame:
    """Per-partition window/session counts and prevalence for the report."""
    rows = []
    for partition, view in study.raw.items():
        split_key = _PARTITION_SPLIT[partition]
        rows.append(
            {
                "partition": partition,
                "assigned_sessions": len(study.manifest["split"][split_key]),
                "contributing_sessions": view.identities["session_file"].nunique(),
                "parent_windows": len(view.parent),
                "anomalous_windows": int(view.labels.sum()),
                "prevalence": float(view.labels.mean()) if len(view.labels) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_figures(results: list[dict], y_test: np.ndarray, directory: Path) -> list[Path]:
    """Write grouped ROC/PR/score/confusion figures for the primary approaches.

    Panels are grouped by model family and augmentation condition; confusion
    matrices use the seed-42 primary rows only and are never averaged across
    seeds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    primary = [row for row in results if row.get("seed") is None]
    written: list[Path] = []
    for family in ("if", "ocsvm", "ae", "ensemble"):
        for augmented in (False, True):
            group = [
                row
                for row in primary
                if row["family"] == family and bool(row["augmented"]) == augmented
            ]
            if not group:
                continue
            condition = "aug_on" if augmented else "aug_off"
            scores = {row["model"]: row["scores"] for row in group}
            for plotter, prefix in (
                (plot_roc_curves, "roc"),
                (plot_pr_curves, "pr"),
                (plot_score_distributions, "scores"),
            ):
                path = directory / f"{prefix}_{family}_{condition}.png"
                plotter(scores, y_test, path)
                written.append(path)
            path = directory / f"confusion_{family}_{condition}.png"
            plot_confusion_matrices(group, path)
            written.append(path)
    return written


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
        for plotting) and the enriched summary table indexed by approach id
        (locked params, recipes/weights, and test prevalence included). Also
        persists ``test_results.csv``/``.json``, ``dataset_summary.csv``, and
        the grouped figures under ``figures/``.
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
    if manifest.get("missing_cells"):
        logger.warning(
            "Frozen manifest reports %d missing cell(s); the report is "
            "incomplete until they are resolved",
            len(manifest["missing_cells"]),
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

    summary = summarize(results).join(_params_frame(entries))
    summary["test_prevalence"] = float(np.mean(y_test))
    summary.to_csv(output_dir / "test_results.csv")
    _dataset_summary(study).to_csv(output_dir / "dataset_summary.csv", index=False)
    (output_dir / "test_results.json").write_text(
        json.dumps([_public(result) for result in results], indent=2) + "\n"
    )
    write_figures(results, y_test, output_dir / FIGURE_DIR_NAME)
    logger.info("Reported %d approaches in %s", len(results), output_dir)
    return results, summary
