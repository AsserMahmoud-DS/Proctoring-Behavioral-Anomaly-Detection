"""Frozen Experiment 1 study configuration (Study Protocol Phase 3).

These values are declared in ``study/plans/experiment_plan.md`` and must not be
edited after validation results are seen. They are intentionally separate from
the production :class:`~cheatdetect.config.ExperimentConfig`: the study compares
nine fixed recipes under the frozen audit split, while production keeps its own
smaller deployment grid.

Candidate lists are generated in declaration order; ties are resolved by first
candidate. The study consumes the frozen audit split and the nine fitted
preprocessors from :func:`cheatdetect.data.prepare_study`; nothing here refits
preprocessing.
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Mapping

from cheatdetect.data.dataset import RECIPES

RECIPE_NAMES = tuple(name for name, _, _ in RECIPES)
IF_RECIPES = tuple(name for name in RECIPE_NAMES if name.startswith("IF-"))
OCSVM_RECIPES = tuple(name for name in RECIPE_NAMES if name.startswith("SVM-"))
AE_RECIPES = tuple(name for name in RECIPE_NAMES if name.startswith("AE-"))

# Augmentation off first, then on (protocol order).
AUGMENTATION_CONDITIONS = (False, True)
CONDITION_NAMES = {False: "aug_off", True: "aug_on"}

# Isolation Forest: 2 recipes x 2 max_samples x 2 max_features x 3 trees = 24.
IF_AXES = (
    ("max_samples", (128, 256)),
    ("max_features", (0.75, 1.0)),
    ("n_estimators", (100, 300, 500)),
)
# Contamination controls the estimator's default offset, not the score order
# used for selection and external threshold tuning.
IF_FIXED = {"contamination": "auto"}

# One-Class SVM: 4 recipes x 3 nu x 3 gamma scales = 36. Image gamma candidates
# are 0.25*centre, centre, 4*centre; the centre is the original-training
# variance-aware value frozen in StudyConfig.gamma_centers.
OCSVM_AXES = (("nu", (0.01, 0.05, 0.10)),)
OCSVM_GAMMA_SCALES = (0.25, 1.0, 4.0)
OCSVM_FIXED = {"kernel": "rbf", "tol": 1e-3, "max_iter": -1}

# LSTM-AE main search: 3 recipes x 2 hidden x 3 batch x 2 lr = 36 (1/1 layers).
AE_MAIN_AXES = (
    ("hidden_dim", (16, 32)),
    ("batch_size", (16, 32, 64)),
    ("lr", (3e-4, 1e-3)),
)
# Depth comparison: 3 recipes x 2 hidden = 6 (2/2 layers, batch 32, lr 1e-3).
AE_DEPTH_HIDDEN_DIMS = (16, 32)
AE_DEPTH_FIXED = {"batch_size": 32, "lr": 1e-3, "num_layers": 2}

PROTOCOL_BUDGETS = {
    "if_per_condition": 24,
    "ocsvm_per_condition": 36,
    "ae_per_condition": 42,
    "per_condition": 102,
    "main_fits": 204,
    "repeat_fits": 8,
    "total_fits": 212,
    "primary_approaches": 20,
}


@dataclass(frozen=True)
class StudyConfig:
    """Frozen geometry, grids, controls, and seeds for the paired study."""

    # Window geometry and labels.
    chunk_size: int = 50
    step_size: int = 25
    sub_chunk: int = 10
    sub_step: int = 5
    label_fraction: float = 0.5
    screen_bounds: tuple[int, int] = (1920, 1080)

    # Training-only augmentation.
    n_copies: int = 2
    sigma_range: tuple[float, float] = (2.0, 5.0)
    aug_seed: int = 42

    # Split reference (the study consumes the frozen audit assignments).
    split_seed: int = 42
    normal_val_size: float = 0.2
    mixed_val_size: float = 0.3

    # Model seeds.
    main_seed: int = 42
    repeat_seeds: tuple[int, ...] = (123, 2026)

    # Threshold policy and ensemble weights.
    precision_floor: float = 0.5
    ensemble_weights: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0)

    # Fixed AE controls (see the protocol table).
    lstm_epochs: int = 100
    lstm_patience: int = 10
    lstm_weight_decay: float = 1e-4
    lstm_latent_dropout: float = 0.1
    lstm_dropout: float = 0.0
    lstm_min_improvement: float = 1e-3
    lstm_grad_clip: float = 1.0


def _candidate_grid(axes: Iterable[tuple[str, Iterable]]) -> list[dict]:
    """Deterministic product in declaration order (ties keep first candidate)."""
    names = [name for name, _ in axes]
    values = [tuple(candidates) for _, candidates in axes]
    return [dict(zip(names, combination)) for combination in product(*values)]


def if_candidates() -> list[dict]:
    """All 24 Isolation Forest candidate parameter sets (recipe included)."""
    return [
        {"recipe": recipe, **params, **IF_FIXED}
        for recipe in IF_RECIPES
        for params in _candidate_grid(IF_AXES)
    ]


def ocsvm_candidates(gamma_centers: Mapping[str, float]) -> list[dict]:
    """All 36 One-Class SVM candidates, with numeric frozen gamma values."""
    candidates = []
    for recipe in OCSVM_RECIPES:
        center = gamma_centers[recipe]
        for scale in OCSVM_GAMMA_SCALES:
            for params in _candidate_grid(OCSVM_AXES):
                candidates.append(
                    {"recipe": recipe, **params, "gamma": scale * center, **OCSVM_FIXED}
                )
    return candidates


def ae_candidates() -> list[dict]:
    """All 42 LSTM-AE candidates: 36 main (1/1) then 6 depth (2/2)."""
    main = [
        {"recipe": recipe, "num_layers": 1, **params}
        for recipe in AE_RECIPES
        for params in _candidate_grid(AE_MAIN_AXES)
    ]
    depth = [
        {
            "recipe": recipe,
            "hidden_dim": hidden_dim,
            "num_layers": AE_DEPTH_FIXED["num_layers"],
            "batch_size": AE_DEPTH_FIXED["batch_size"],
            "lr": AE_DEPTH_FIXED["lr"],
        }
        for recipe in AE_RECIPES
        for hidden_dim in AE_DEPTH_HIDDEN_DIMS
    ]
    return main + depth


def family_candidates(
    family: str, gamma_centers: Mapping[str, float] | None = None
) -> list[dict]:
    """Candidates for ``if``, ``ocsvm``, or ``ae`` (the runner's family names)."""
    if family == "if":
        return if_candidates()
    if family == "ocsvm":
        if gamma_centers is None:
            raise ValueError("OCSVM candidates require frozen gamma centres")
        return ocsvm_candidates(gamma_centers)
    if family == "ae":
        return ae_candidates()
    raise ValueError(f"Unknown model family '{family}'; expected if, ocsvm, or ae")


def validate_budgets() -> dict:
    """Check the builder output against the frozen protocol budget table."""
    dummy_centers = {name: 1.0 for name in OCSVM_RECIPES}
    if_per_condition = len(if_candidates())
    ocsvm_per_condition = len(ocsvm_candidates(dummy_centers))
    ae_per_condition = len(ae_candidates())
    per_condition = if_per_condition + ocsvm_per_condition + ae_per_condition
    counts = {
        "if_per_condition": if_per_condition,
        "ocsvm_per_condition": ocsvm_per_condition,
        "ae_per_condition": ae_per_condition,
        "per_condition": per_condition,
        "main_fits": 2 * per_condition,
        "repeat_fits": 8,
        "total_fits": 2 * per_condition + 8,
        "primary_approaches": 20,
    }
    if counts != PROTOCOL_BUDGETS:
        raise ValueError(f"Study budgets drifted from the protocol: {counts}")
    return counts


def config_as_dict(config: StudyConfig) -> dict:
    """JSON-ready copy of the frozen config for run metadata."""
    return asdict(config)
