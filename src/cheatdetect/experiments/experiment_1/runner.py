"""Validation-only study runner (Study Protocol Phase 3).

The runner builds the frozen paired study bundle, sweeps the predeclared
candidate grids on combined validation only, locks one recipe winner per
``(family, recipe, augmentation)`` cell, pairs the best IF/OCSVM winners into
one ensemble per augmentation condition, and freezes the reporting manifest
that Study Phase 6 consumes.

Two isolation rules are enforced by tests rather than convention:

- the runner never imports :mod:`cheatdetect.pipeline.report` and never names
  held-out test partitions, so test data cannot leak into selection;
- every write goes through
  :func:`cheatdetect.data.dataset.study_output_directory`, so experiment
  outputs can never touch ``best_models/`` or ``reports/``.

Frozen preprocessing (Option A): each candidate receives a deep copy of the
original-training-fitted :class:`~cheatdetect.data.FeaturePreprocessor` from the
prepared bundle, so augmentation changes only the rows the estimator sees.
"""

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cheatdetect.config import AUDIT_DIR, MIXED_DIR, NORMAL_DIR, STUDY_ARTIFACTS_DIR
from cheatdetect.data import (
    PreparedStudy,
    input_manifest,
    load_prepared,
    prepare_study,
    save_prepared,
)
from cheatdetect.data import dataset
from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.models import (
    IsolationForestDetector,
    OCSVMDetector,
    tune_threshold,
)
from cheatdetect.models.ensemble import grid_search as ensemble_grid_search
from cheatdetect.models.selection import sweep_candidates

from .config import (
    AE_RECIPES,
    AUGMENTATION_CONDITIONS,
    CONDITION_NAMES,
    IF_RECIPES,
    OCSVM_RECIPES,
    RECIPE_NAMES,
    StudyConfig,
    ae_candidates,
    config_as_dict,
    if_candidates,
    ocsvm_candidates,
)

logger = logging.getLogger(__name__)

BUNDLE_DIR_NAME = "phase_03_study"
RUN_DIR_NAME = "phase_04_study"
DEFAULT_BUNDLE_DIR = STUDY_ARTIFACTS_DIR / BUNDLE_DIR_NAME
DEFAULT_RUN_DIR = STUDY_ARTIFACTS_DIR / RUN_DIR_NAME

FAMILY_RECIPES = {
    "if": IF_RECIPES,
    "ocsvm": OCSVM_RECIPES,
    "ae": AE_RECIPES,
}


@dataclass(frozen=True)
class ValidationViews:
    """Raw training/validation views for one augmentation condition."""

    flat_train: pd.DataFrame
    flat_val: pd.DataFrame
    flat_val_normal: pd.DataFrame
    sequence_train: np.ndarray
    sequence_val: np.ndarray
    sequence_val_normal: np.ndarray
    y_val: np.ndarray


@dataclass
class Winner:
    """One locked recipe or ensemble winner with its fitted detector."""

    family: str
    recipe: str
    augmented: bool
    params: dict
    roc_auc: float
    pr_auc: float
    threshold: float
    threshold_info: dict
    detector: object = field(repr=False)
    artifact: Path | None = None
    if_recipe: str | None = None
    ocsvm_recipe: str | None = None
    if_weight: float | None = None

    @property
    def condition(self) -> str:
        return CONDITION_NAMES[self.augmented]

    @property
    def id(self) -> str:
        return f"{self.family}:{self.recipe}:{self.condition}"


@dataclass
class StudyRun:
    """In-memory result of a validation-only search run."""

    candidates: pd.DataFrame
    winners: dict[str, Winner]
    ensembles: dict[str, Winner]
    manifest: dict
    directory: Path


def frozen_split() -> dict:
    """Return the Phase 1 audit's frozen session assignment."""
    manifest = json.loads((AUDIT_DIR / "manifest.json").read_text())
    return {key: list(names) for key, names in manifest["split"].items()}


def build_study(
    config: StudyConfig = StudyConfig(),
    *,
    split: dict | None = None,
    normal_dir: Path = NORMAL_DIR,
    mixed_dir: Path = MIXED_DIR,
    output_dir: Path | None = None,
) -> PreparedStudy:
    """Build and persist the frozen paired bundle under ``study/artifacts/``."""
    directory = dataset.study_output_directory(output_dir or DEFAULT_BUNDLE_DIR)
    study = prepare_study(
        split=split,
        normal_dir=normal_dir,
        mixed_dir=mixed_dir,
        n_copies=config.n_copies,
        sigma_range=config.sigma_range,
        chunk_size=config.chunk_size,
        step_size=config.step_size,
        sub_chunk=config.sub_chunk,
        sub_step=config.sub_step,
        label_fraction=config.label_fraction,
        recipes=RECIPE_NAMES,
        random_state=config.main_seed,
    )
    save_prepared(study, directory)
    return study


def load_study(
    config: StudyConfig = StudyConfig(),
    *,
    split: dict | None = None,
    normal_dir: Path = NORMAL_DIR,
    mixed_dir: Path = MIXED_DIR,
    output_dir: Path | None = None,
) -> PreparedStudy:
    """Load the bundle only after a freshly computed manifest matches."""
    directory = dataset.study_output_directory(output_dir or DEFAULT_BUNDLE_DIR)
    split = frozen_split() if split is None else split
    expected = input_manifest(
        split,
        normal_dir,
        mixed_dir,
        config.n_copies,
        config.sigma_range,
        config.chunk_size,
        config.step_size,
        config.sub_chunk,
        config.sub_step,
        config.label_fraction,
        config.main_seed,
    )
    return load_prepared(directory, expected)


def validation_views(study: PreparedStudy, augmented: bool) -> ValidationViews:
    """Raw flat and sequence views for one augmentation condition."""
    raw = study.raw
    flat_val_normal = raw["normal_val"].parent
    return ValidationViews(
        flat_train=study.training_raw(augmented),
        flat_val=pd.concat(
            [flat_val_normal, raw["mixed_val"].parent], ignore_index=True
        ),
        flat_val_normal=flat_val_normal,
        sequence_train=study.training_sequences(augmented),
        sequence_val=np.concatenate(
            [raw["normal_val"].sequences, raw["mixed_val"].sequences], axis=0
        ),
        sequence_val_normal=raw["normal_val"].sequences,
        y_val=np.concatenate([raw["normal_val"].labels, raw["mixed_val"].labels]),
    )


def _candidate_pools(
    study: PreparedStudy,
    config: StudyConfig,
    candidates: Mapping[str, list[dict]] | None,
) -> dict[str, list[dict]]:
    if candidates is not None:
        return {family: list(pool) for family, pool in candidates.items()}
    return {
        "if": if_candidates(),
        "ocsvm": ocsvm_candidates(study.gamma_centers),
        "ae": ae_candidates(),
    }


def _params(candidate: Mapping) -> dict:
    return {key: value for key, value in candidate.items() if key != "recipe"}


def _evaluator(
    family: str,
    recipe: str,
    study: PreparedStudy,
    config: StudyConfig,
    views: ValidationViews,
):
    """Closure fitting one candidate on augmented raw rows with frozen preprocessing."""

    def evaluate(candidate: dict):
        params = _params(candidate)
        frozen = copy.deepcopy(study.preprocessors[recipe])
        # The detector's ``recipe`` names the preprocessing rule, while the
        # study recipe (e.g. "IF-raw") names the frozen processor to reuse.
        processor_recipe = frozen.recipe
        if family == "if":
            detector = IsolationForestDetector(
                recipe=processor_recipe,
                preprocessor=frozen,
                random_state=config.main_seed,
                **params,
            )
            detector.fit(views.flat_train)
            return detector, detector.decision_function(views.flat_val), {}
        if family == "ocsvm":
            detector = OCSVMDetector(
                recipe=processor_recipe,
                preprocessor=frozen,
                random_state=config.main_seed,
                **params,
            )
            detector.fit(views.flat_train)
            return detector, detector.decision_function(views.flat_val), {}
        if family == "ae":
            # Lazy import keeps torch out of phases that do not run the AE.
            from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector

            detector = LSTMAutoencoderDetector(
                feature_names=list(SOURCE_FEATURES),
                input_dim=len(SOURCE_FEATURES),
                seq_len=views.sequence_train.shape[1],
                recipe=processor_recipe,
                preprocessor=frozen,
                dropout=config.lstm_latent_dropout,
                epochs=config.lstm_epochs,
                patience=config.lstm_patience,
                random_state=config.main_seed,
                **params,
            )
            detector.fit(views.sequence_train, X_es=views.sequence_val_normal)
            return detector, detector.decision_function(views.sequence_val), {}
        raise ValueError(f"Unknown model family '{family}'")

    return evaluate


def _validation_scores(family: str, detector, views: ValidationViews) -> np.ndarray:
    if family == "ae":
        return detector.decision_function(views.sequence_val)
    return detector.decision_function(views.flat_val)


def _run_cell(
    family: str,
    recipe: str,
    augmented: bool,
    cell_candidates: list[dict],
    study: PreparedStudy,
    config: StudyConfig,
    views: ValidationViews,
    directory: Path,
) -> tuple[Winner | None, list[dict]]:
    evaluate = _evaluator(family, recipe, study, config, views)
    detector, results = sweep_candidates(cell_candidates, evaluate, views.y_val)
    rows = [
        {"family": family, "recipe": recipe, "augmented": augmented, **row}
        for row in results.to_dict("records")
    ]
    if detector is None:
        logger.warning(
            "No valid candidate for %s/%s (%s)", family, recipe, CONDITION_NAMES[augmented]
        )
        return None, rows

    scores = _validation_scores(family, detector, views)
    threshold_info = tune_threshold(scores, views.y_val, config.precision_floor)
    best_row = results.iloc[0]
    best_params = {
        key: best_row[key] for key in cell_candidates[0] if key in results.columns
    }
    winner = Winner(
        family=family,
        recipe=recipe,
        augmented=augmented,
        params=_params(best_params),
        roc_auc=float(best_row["roc_auc"]),
        pr_auc=float(best_row["pr_auc"]),
        threshold=float(threshold_info["threshold"]),
        threshold_info=threshold_info,
        detector=detector,
    )
    winner.artifact = directory / f"winner_{family}_{recipe}_{winner.condition}.joblib"
    joblib.dump(detector, winner.artifact)
    return winner, rows


def _best_overall(
    winners: Mapping[str, Winner], family: str, augmented: bool
) -> Winner | None:
    """Highest validation ROC-AUC winner for a family/condition (first tie)."""
    candidates = [
        winner
        for winner in winners.values()
        if winner.family == family and winner.augmented == augmented
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda winner: winner.roc_auc)


def _lock_ensemble(
    study: PreparedStudy,
    config: StudyConfig,
    winners: Mapping[str, Winner],
    augmented: bool,
    directory: Path,
) -> Winner | None:
    if_winner = _best_overall(winners, "if", augmented)
    ocsvm_winner = _best_overall(winners, "ocsvm", augmented)
    if if_winner is None or ocsvm_winner is None:
        return None

    views = validation_views(study, augmented)
    ensemble, results = ensemble_grid_search(
        if_winner.detector,
        ocsvm_winner.detector,
        views.flat_val,
        views.y_val,
        config.ensemble_weights,
        X_ref=views.flat_val_normal,
    )
    scores = ensemble.decision_function(views.flat_val)
    threshold_info = tune_threshold(scores, views.y_val, config.precision_floor)
    winner = Winner(
        family="ensemble",
        recipe=f"{if_winner.recipe}+{ocsvm_winner.recipe}",
        augmented=augmented,
        params={
            "if_weight": ensemble.if_weight,
            "ocsvm_weight": 1 - ensemble.if_weight,
        },
        roc_auc=float(results.iloc[0]["roc_auc"]),
        pr_auc=float(results.iloc[0]["pr_auc"]),
        threshold=float(threshold_info["threshold"]),
        threshold_info=threshold_info,
        detector=ensemble,
        if_recipe=if_winner.recipe,
        ocsvm_recipe=ocsvm_winner.recipe,
        if_weight=ensemble.if_weight,
    )
    winner.artifact = directory / f"winner_ensemble_{winner.condition}.joblib"
    joblib.dump(ensemble, winner.artifact)
    return winner


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _winner_entry(winner: Winner) -> dict:
    return _json_safe(
        {
            "id": winner.id,
            "family": winner.family,
            "recipe": winner.recipe,
            "augmented": winner.augmented,
            "params": winner.params,
            "roc_auc": winner.roc_auc,
            "pr_auc": winner.pr_auc,
            "threshold": winner.threshold,
            "threshold_info": winner.threshold_info,
            "artifact": str(winner.artifact),
            "if_recipe": winner.if_recipe,
            "ocsvm_recipe": winner.ocsvm_recipe,
            "if_weight": winner.if_weight,
        }
    )


def _freeze_manifest(
    study: PreparedStudy,
    config: StudyConfig,
    winners: Mapping[str, Winner],
    ensembles: Mapping[str, Winner],
) -> dict:
    return {
        "version": 1,
        "experiment": "experiment_1",
        "frozen": True,
        "study_config": config_as_dict(config),
        "study_manifest_version": study.manifest.get("version"),
        "split": study.manifest["split"],
        "primary": [_winner_entry(winner) for winner in winners.values()]
        + [_winner_entry(winner) for winner in ensembles.values()],
    }


def run_search(
    study: PreparedStudy,
    config: StudyConfig = StudyConfig(),
    *,
    candidates: Mapping[str, list[dict]] | None = None,
    output_dir: Path | None = None,
) -> StudyRun:
    """Sweep the frozen grids on validation only and lock the study winners.

    Args:
        study: Manifest-checked prepared bundle with original-fitted recipes.
        config: Frozen study configuration.
        candidates: Optional per-family candidate override (tests/smoke runs).
            Defaults to the full predeclared grids.
        output_dir: Study artifact directory (must stay under ``study/artifacts``).

    Returns:
        :class:`StudyRun` with the per-candidate table, locked winners,
        per-condition ensembles, and the frozen reporting manifest.
    """
    directory = dataset.study_output_directory(output_dir or DEFAULT_RUN_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    pools = _candidate_pools(study, config, candidates)
    rows: list[dict] = []
    winners: dict[str, Winner] = {}
    for augmented in AUGMENTATION_CONDITIONS:
        views = validation_views(study, augmented)
        for family, pool in pools.items():
            for recipe in FAMILY_RECIPES.get(family, ()):
                cell = [candidate for candidate in pool if candidate["recipe"] == recipe]
                if not cell:
                    continue
                winner, cell_rows = _run_cell(
                    family, recipe, augmented, cell, study, config, views, directory
                )
                rows.extend(cell_rows)
                if winner is not None:
                    winners[winner.id] = winner

    ensembles = {}
    for augmented in AUGMENTATION_CONDITIONS:
        ensemble = _lock_ensemble(study, config, winners, augmented, directory)
        if ensemble is not None:
            ensembles[ensemble.id] = ensemble

    candidates_df = pd.DataFrame(rows)
    manifest = _freeze_manifest(study, config, winners, ensembles)
    candidates_df.to_csv(directory / "candidates.csv", index=False)
    (directory / "reporting_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    logger.info(
        "Locked %d recipe winners and %d ensembles in %s",
        len(winners),
        len(ensembles),
        directory,
    )
    return StudyRun(candidates_df, winners, ensembles, manifest, directory)
