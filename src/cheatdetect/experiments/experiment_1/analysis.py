"""Validation-table comparisons for the Experiment 1 study (Phase 4 analysis).

These helpers are pure pandas transformations over the persisted candidate
table and the frozen reporting manifest: no detector is refit, no test data is
read. ``runner.run_search`` writes the three tables next to ``candidates.csv``
so both the CLI and any notebook-triggered run persist the same comparisons.

Comparisons (Experiment 1 protocol):

- **Matched configuration**: identical family/recipe/hyperparameters with
  augmentation off vs on. Only the training rows differ, so the deltas estimate
  the augmentation intervention. AE rows also report optimizer updates because
  augmentation adds update steps under a fixed epoch cap.
- **Independently tuned winners**: each condition's per-recipe winner from the
  frozen manifest (an approach-level comparison, not an isolated intervention).
- **AE depth**: the two-layer depth block against its matching one-layer
  configuration at batch 32 / learning rate 1e-3, holding recipe, hidden size,
  augmentation state, and seed fixed.
"""

from pathlib import Path

import pandas as pd

MATCHED_KEYS = {
    "if": ["max_samples", "max_features", "n_estimators", "contamination"],
    "ocsvm": ["nu", "gamma", "kernel", "tol", "max_iter"],
    "ae": ["hidden_dim", "num_layers", "batch_size", "lr"],
}
MATCHED_METRICS = ["roc_auc", "pr_auc", "seconds"]
AE_DIAGNOSTICS = ["epochs_trained", "optimizer_updates"]

COMPARISON_FILES = (
    "augmentation_matched.csv",
    "augmentation_winners.csv",
    "depth_comparison.csv",
)


def matched_augmentation_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Pair identical configurations across augmentation off/on."""
    ok = candidates[candidates["status"] == "ok"]
    frames = []
    for family, keys in MATCHED_KEYS.items():
        subset = ok[ok["family"] == family]
        if subset.empty or not set(keys).issubset(subset.columns):
            continue
        columns = ["recipe", *keys, *MATCHED_METRICS]
        if family == "ae":
            columns += AE_DIAGNOSTICS
        off = subset[~subset["augmented"].astype(bool)][columns]
        on = subset[subset["augmented"].astype(bool)][columns]
        merged = off.merge(
            on, on=["recipe", *keys], suffixes=("_off", "_on"), how="outer"
        )
        for metric in ("roc_auc", "pr_auc"):
            merged[f"delta_{metric}"] = merged[f"{metric}_on"] - merged[f"{metric}_off"]
        merged["paired"] = merged["roc_auc_off"].notna() & merged["roc_auc_on"].notna()
        merged.insert(0, "family", family)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        ["family", "recipe", "paired"], ascending=[True, True, False]
    ).reset_index(drop=True)


def augmentation_winners_table(manifest: dict) -> pd.DataFrame:
    """Per-recipe locked winners with augmentation off/on deltas."""
    if not manifest.get("primary"):
        return pd.DataFrame()
    rows = [
        {
            "family": entry["family"],
            "recipe": entry["recipe"],
            "augmented": bool(entry["augmented"]),
            "roc_auc": entry["roc_auc"],
            "pr_auc": entry["pr_auc"],
            "threshold": entry["threshold"],
            "fallback_used": entry["threshold_info"]["fallback_used"],
        }
        for entry in manifest["primary"]
        if entry["family"] != "ensemble"
    ]
    frame = pd.DataFrame(rows)
    off = frame[~frame["augmented"]]
    on = frame[frame["augmented"]]
    merged = off.merge(on, on=["family", "recipe"], suffixes=("_off", "_on"))
    for metric in ("roc_auc", "pr_auc"):
        merged[f"delta_{metric}"] = merged[f"{metric}_on"] - merged[f"{metric}_off"]
    return merged.sort_values(["family", "recipe"]).reset_index(drop=True)


def depth_comparison_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Two-layer vs matched one-layer AE configurations (batch 32, lr 1e-3)."""
    ae_columns = {"family", "status", "batch_size", "lr"}
    if candidates.empty or not ae_columns.issubset(candidates.columns):
        return pd.DataFrame()
    ae = candidates[
        (candidates["family"] == "ae")
        & (candidates["status"] == "ok")
        & (candidates["batch_size"] == 32)
        & (candidates["lr"] == 1e-3)
    ]
    columns = [
        "recipe", "hidden_dim", "augmented", "roc_auc", "pr_auc",
        *AE_DIAGNOSTICS, "best_es_loss",
    ]
    deep = ae[ae["num_layers"] == 2][columns]
    shallow = ae[ae["num_layers"] == 1][columns]
    merged = deep.merge(
        shallow, on=["recipe", "hidden_dim", "augmented"], suffixes=("_2layer", "_1layer")
    )
    for metric in ("roc_auc", "pr_auc"):
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_2layer"] - merged[f"{metric}_1layer"]
        )
    return merged.sort_values(["recipe", "hidden_dim", "augmented"]).reset_index(
        drop=True
    )


def write_comparisons(
    directory: Path, candidates: pd.DataFrame, manifest: dict
) -> dict[str, pd.DataFrame]:
    """Write the three comparison tables into the Phase 4 run directory."""
    directory = Path(directory)
    tables = {
        "augmentation_matched.csv": matched_augmentation_table(candidates),
        "augmentation_winners.csv": augmentation_winners_table(manifest),
        "depth_comparison.csv": depth_comparison_table(candidates),
    }
    for name, table in tables.items():
        table.to_csv(directory / name, index=False)
    return tables
