"""Original-training diagnostics; no detector fitting or held-out data loading."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from cheatdetect.config import AUDIT_DIR, MIXED_DIR, NORMAL_DIR, SPLIT_INFO_PATH
from cheatdetect.data.build import merge_window_switch_events
from cheatdetect.data.cleaning import clean_session_data
from cheatdetect.data.features.extract import extract_features_from_session
from cheatdetect.data.sequences import extract_sequences_from_sessions

from cheatdetect.data.feature_schema import (
    CONTINUOUS, COUNT_LOG, DIRECTION, SIGNED, SOURCE_FEATURES, UNSCALED,
)


def feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize before imputation so bad extraction cannot be hidden."""
    values = frame.to_numpy(dtype=float)
    finite = frame.where(np.isfinite(values))
    result = pd.DataFrame({
        "nonfinite_count": (~np.isfinite(values)).sum(axis=0),
        "negative_fraction": (values < 0).mean(axis=0),
        "zero_fraction": (values == 0).mean(axis=0),
        "unique_finite": finite.nunique(),
        "min": finite.min(), "median": finite.median(),
        "p99": finite.quantile(0.99), "max": finite.max(),
        "std": finite.std(ddof=0), "skew": finite.skew(),
    }, index=frame.columns)
    result["constant"] = result["unique_finite"] <= 1
    return result.rename_axis("feature").reset_index()


def audit_recipe(frame: pd.DataFrame, recipe: str, scale: bool) -> tuple[pd.DataFrame, dict]:
    """Fit a diagnostic recipe on this training representation only, not for reuse."""
    if recipe not in {"base", "log", "yj", "quantile"}:
        raise ValueError(f"Unknown audit recipe: {recipe}")
    numerical = frame.drop(columns=DIRECTION).copy().astype(float)
    if not np.isfinite(numerical.to_numpy()).all():
        raise ValueError("Audit recipes require finite original training features")
    direction = frame[DIRECTION].to_numpy()
    if not np.isin(direction, np.arange(9)).all():
        raise ValueError("Unexpected direction category")
    fitted = {}
    if recipe == "log":
        for feature in CONTINUOUS:
            values = numerical[feature]
            if feature in SIGNED:
                numerical[feature] = np.sign(values) * np.log1p(np.abs(values))
            else:
                if (values < 0).any():
                    raise ValueError(f"Negative values in log feature {feature}")
                numerical[feature] = np.log1p(values)
    elif recipe in {"yj", "quantile"}:
        for feature in CONTINUOUS:
            if numerical[feature].nunique() <= 1:
                fitted[feature] = {"constant_bypass": True}
                continue
            transformer = (
                PowerTransformer(method="yeo-johnson", standardize=False)
                if recipe == "yj" else QuantileTransformer(
                    n_quantiles=min(100, len(frame)), output_distribution="normal",
                    subsample=None, random_state=42,
                )
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                numerical[feature] = transformer.fit_transform(numerical[[feature]]).ravel()
            fitted[feature] = {"warnings": [str(item.message) for item in captured]}
            if recipe == "yj":
                fitted[feature]["lambda"] = float(transformer.lambdas_[0])
    if recipe != "base":
        numerical[list(COUNT_LOG)] = np.log1p(numerical[list(COUNT_LOG)])
    if scale:
        columns = [feature for feature in numerical if feature not in UNSCALED]
        scaler = StandardScaler()
        numerical[columns] = scaler.fit_transform(numerical[columns])
        fitted["scaler"] = {
            feature: {"mean": float(mean), "scale": float(std)}
            for feature, mean, std in zip(columns, scaler.mean_, scaler.scale_)
        }
    for category in range(9):
        numerical[f"{DIRECTION}_{category}"] = (direction == category).astype(float)
    if not np.isfinite(numerical.to_numpy()).all():
        raise ValueError(f"Nonfinite transformed values in {recipe}")
    return numerical, fitted


def kernel_geometry(frame: pd.DataFrame, random_state: int = 42) -> list[dict]:
    """Summarize off-diagonal kernels on a bounded reproducible training sample."""
    values = frame.to_numpy(dtype=float)
    variance = float(values.var(ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Cannot derive a finite positive gamma centre")
    gamma_center = 1 / (values.shape[1] * variance)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(values), size=min(512, len(values)), replace=False)
    distances = pairwise_distances(values[indices], metric="sqeuclidean")
    distances = distances[np.triu_indices(len(indices), k=1)]
    if not len(distances):
        raise ValueError("Kernel audit needs at least two training rows")
    rows = []
    for multiplier in (0.25, 1.0, 4.0):
        kernel = np.exp(-multiplier * gamma_center * distances)
        rows.append({
            "gamma_center": gamma_center, "gamma_multiplier": multiplier,
            "gamma": multiplier * gamma_center, "global_variance": variance,
            "sample_rows": len(indices), "sample_pairs": len(distances),
            "distance_p10": float(np.quantile(distances, 0.1)),
            "distance_median": float(np.median(distances)),
            "distance_p90": float(np.quantile(distances, 0.9)),
            "kernel_p10": float(np.quantile(kernel, 0.1)),
            "kernel_median": float(np.median(kernel)),
            "kernel_p90": float(np.quantile(kernel, 0.9)),
            "kernel_below_001_fraction": float((kernel < 0.01).mean()),
            "kernel_above_099_fraction": float((kernel > 0.99).mean()),
        })
    return rows


def run_original_training_audit(output_dir: Path = AUDIT_DIR) -> dict:
    """Read train raw files fresh; save diagnostics without touching legacy caches."""
    split = json.loads(SPLIT_INFO_PATH.read_text())
    normal_train, normal_val = train_test_split(
        sorted(NORMAL_DIR.glob("*.csv")), test_size=0.2, random_state=42,
    )
    mixed_val, mixed_test = train_test_split(
        sorted(MIXED_DIR.glob("*.csv")), test_size=0.7, random_state=42,
    )
    for key, paths in {
        "normal_train": normal_train, "normal_val": normal_val,
        "mixed_val": mixed_val, "mixed_test": mixed_test,
    }.items():
        if split[key] != [path.name for path in paths]:
            raise ValueError(f"Saved split disagrees with current seed-42 assignment: {key}")
    parent_frames, micro_frames, identity_frames, session_rows = [], [], [], []
    for path in normal_train:
        raw = pd.read_csv(path)
        cleaned = clean_session_data(raw)
        if cleaned["Is Cheating"].any():
            raise ValueError(f"Cheating events in normal training: {path.name}")
        parent = merge_window_switch_events(extract_features_from_session(
            cleaned, chunk_size=50, step_size=25, cheating_threshold=0.5,
            session_name=path.stem,
        ))
        sequences, labels, names = extract_sequences_from_sessions(
            [cleaned], 50, 25, 10, 5, cheating_threshold=0.5,
        )
        if len(parent) != len(sequences) or labels.any():
            raise ValueError(f"Parent/micro alignment failed: {path.name}")
        starts = np.arange(len(parent)) * 25
        identities = pd.DataFrame({
            "session_file": path.name, "parent_start_event": starts,
            "parent_stop_event_exclusive": starts + 50,
            "parent_start_time_seconds": cleaned.iloc[starts]["Time (seconds)"].to_numpy(),
        })
        identity_frames.append(identities)
        parent_frames.append(parent[list(SOURCE_FEATURES)])
        micro = merge_window_switch_events(pd.DataFrame(
            sequences.reshape(-1, len(names)), columns=names,
        ))
        micro_frames.append(micro[list(SOURCE_FEATURES)])
        mouse_times = cleaned.loc[cleaned["is_mouse_event"], "Time (seconds)"].sort_values().diff()
        positive_dt = mouse_times[mouse_times > 0]
        session_rows.append({
            "session_file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cleaned_events": len(cleaned), "parent_rows": len(parent),
            "time_reversals_in_event_order": int((cleaned["Time (seconds)"].diff() < 0).sum()),
            "duplicate_mouse_timestamps": int((mouse_times == 0).sum()),
            "minimum_positive_mouse_dt_seconds": float(positive_dt.min()),
            "keyboard_copy_paste_events": int(cleaned["Event Type"].isin(["copy", "paste"]).sum()),
            "mouse_x_max": float(cleaned["X Coordinate"].max()),
            "mouse_y_max": float(cleaned["Y Coordinate"].max()),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(output_dir / "sessions.csv", index=False)
    pd.concat(identity_frames, ignore_index=True).to_csv(output_dir / "parent_identities.csv", index=False)
    geometry_rows, transform_rows, parameters = [], [], {}
    representations = {
        "parent": pd.concat(parent_frames, ignore_index=True),
        "micro": pd.concat(micro_frames, ignore_index=True),
    }
    for representation, frame in representations.items():
        feature_statistics(frame).to_csv(output_dir / f"{representation}_features.csv", index=False)
        for recipe in ("base", "log", "yj", "quantile"):
            if representation == "micro" and recipe == "quantile":
                continue
            transformed, fitted = audit_recipe(frame, recipe, scale=True)
            parameters[f"{representation}_{recipe}"] = fitted
            summary = feature_statistics(transformed)
            summary.insert(0, "recipe", recipe)
            summary.insert(0, "representation", representation)
            transform_rows.append(summary)
            if representation == "parent":
                geometry_rows.extend({"recipe": recipe, **row} for row in kernel_geometry(transformed))
        if representation == "parent":
            for recipe in ("base", "log"):
                transformed, _ = audit_recipe(frame, recipe, scale=False)
                summary = feature_statistics(transformed)
                summary.insert(0, "recipe", f"IF-{recipe}")
                summary.insert(0, "representation", representation)
                transform_rows.append(summary)
    pd.concat(transform_rows, ignore_index=True).to_csv(output_dir / "transformed_features.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(output_dir / "kernel_geometry.csv", index=False)
    manifest = {
        "audit_version": 2, "split": split, "augmentation": False,
        "angle_difference_convention": "wrapped_minus_pi_to_pi",
        "read_held_out_contents": False, "legacy_caches_used": False,
        "source_features": list(SOURCE_FEATURES),
        "output_features": list(transformed.columns), "encoded_dimension": 33,
        "parent_geometry": {"events": 50, "stride": 25},
        "micro_geometry": {"events": 10, "stride": 5, "timesteps": 9},
        "parent_rows": len(representations["parent"]),
        "micro_rows_with_overlap": len(representations["micro"]),
        "original_training_sessions": len(normal_train),
        "if_effective_max_samples": {str(size): min(size, len(representations["parent"])) for size in (128, 256)},
        "main_model_fits_planned": 204, "total_model_fits_planned": 212,
        "signed_log_features": list(SIGNED), "diagnostic_fit_parameters": parameters,
        "specification_status": "frozen_phase_01",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    manifest = run_original_training_audit()
    print(json.dumps({key: manifest[key] for key in (
        "parent_rows", "micro_rows_with_overlap", "original_training_sessions",
        "encoded_dimension", "if_effective_max_samples", "specification_status",
    )}, indent=2))
