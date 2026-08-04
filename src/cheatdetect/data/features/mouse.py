"""Mouse movement feature extraction.

Computes kinematic and spatial features from mousemove/click events
within a single chunk: velocity, acceleration, jerk, curvature,
angular velocity, path geometry, idle time, and direction metrics.
"""

import numpy as np
import pandas as pd


def _safe_agg(series: pd.Series, func) -> float:
    """Apply an aggregation function to a Series, returning 0.0 on empty/NaN input.

    Args:
        series: Pandas Series to aggregate.
        func: Numpy reduction (e.g. ``np.mean``, ``np.std``).

    Returns:
        The scalar result, or 0.0 if the series is empty after
        dropping NaNs.
    """
    clean = series.dropna()
    return float(func(clean)) if len(clean) > 0 else 0.0


def extract_mouse_features(mouse_chunk: pd.DataFrame) -> dict:
    """Extract mouse kinematic and spatial features from a chunk.

    If the chunk has fewer than 2 mouse events, returns a dict of
    zeros matching the expected feature key schema.

    Feature groups produced:
        - **Path geometry**: path_length, straightness, direction_class,
          largest_deviation, sum_of_angles, sharp_angles,
          direction_changes
        - **Velocity**: mean, std, max
        - **Acceleration**: mean, std, max
        - **Jerk**: mean, std
        - **Angular velocity**: mean, std, min, max
        - **Curvature**: mean, std, min, max
        - **Other**: click_count, idle_time_ratio

    Args:
        mouse_chunk: DataFrame filtered to mouse events only (must
            contain ``X Coordinate``, ``Y Coordinate``,
            ``Time (seconds)``, ``Event Type``).

    Returns:
        Dictionary of 24 mouse feature values keyed with the
        ``mouse_`` prefix.
    """
    # Short-circuit: not enough points to compute derivatives
    if len(mouse_chunk) < 2:
        return {
            "mouse_path_length": 0.0,
            "mouse_straightness": 0.0,
            "mouse_velocity_mean": 0.0,
            "mouse_velocity_std": 0.0,
            "mouse_velocity_max": 0.0,
            "mouse_acceleration_mean": 0.0,
            "mouse_acceleration_std": 0.0,
            "mouse_acceleration_max": 0.0,
            "mouse_jerk_mean": 0.0,
            "mouse_jerk_std": 0.0,
            "mouse_direction_changes": 0,
            "mouse_click_count": 0,
            "mouse_idle_time_ratio": 0.0,
            "mouse_angular_velocity_mean": 0.0,
            "mouse_angular_velocity_std": 0.0,
            "mouse_angular_velocity_min": 0.0,
            "mouse_angular_velocity_max": 0.0,
            "mouse_curvature_mean": 0.0,
            "mouse_curvature_std": 0.0,
            "mouse_curvature_min": 0.0,
            "mouse_curvature_max": 0.0,
            "mouse_direction_class": 0,
            "mouse_sum_of_angles": 0.0,
            "mouse_largest_deviation": 0.0,
            "mouse_sharp_angles": 0,
        }

    chunk = mouse_chunk.sort_values("Time (seconds)").copy()

    # --- First-order derivatives (position → velocity) ---
    chunk["dx"] = chunk["X Coordinate"].diff()
    chunk["dy"] = chunk["Y Coordinate"].diff()
    chunk["dt"] = chunk["Time (seconds)"].diff()

    chunk["vx"] = chunk["dx"] / chunk["dt"]
    chunk["vy"] = chunk["dy"] / chunk["dt"]
    chunk["velocity"] = np.sqrt(chunk["vx"] ** 2 + chunk["vy"] ** 2)

    # --- Second-order derivatives (velocity → acceleration) ---
    chunk["ax"] = chunk["vx"].diff() / chunk["dt"]
    chunk["ay"] = chunk["vy"].diff() / chunk["dt"]
    chunk["acceleration"] = np.sqrt(chunk["ax"] ** 2 + chunk["ay"] ** 2)

    # --- Third-order derivative (acceleration → jerk) ---
    chunk["jerk"] = chunk["acceleration"].diff() / chunk["dt"]

    # --- Angular kinematics ---
    chunk["angle"] = np.arctan2(chunk["dy"], chunk["dx"])
    # Wrap angle differences into [−π, π] to prevent +2π spikes at ±π boundary
    angle_diff = (chunk["angle"].diff() + np.pi) % (2 * np.pi) - np.pi
    chunk["angular_velocity"] = angle_diff / chunk["dt"]

    # --- Curvature κ = |vx·ay − vy·ax| / |v|³ ---
    velocity_mag_cubed = chunk["velocity"] ** 3
    cross_product = chunk["vx"] * chunk["ay"] - chunk["vy"] * chunk["ax"]
    chunk["curvature"] = np.abs(cross_product) / velocity_mag_cubed

    # Replace ±inf with NaN so _safe_agg skips degenerate rows (dt=0 → inf,
    # velocity=0 in curvature → NaN) rather than corrupting aggregates.
    derivative_cols = [
        "vx", "vy", "velocity", "ax", "ay", "acceleration",
        "jerk", "angular_velocity", "curvature",
    ]
    chunk[derivative_cols] = chunk[derivative_cols].replace([np.inf, -np.inf], np.nan)

    # --- Path-level geometry ---
    distances = np.sqrt(chunk["dx"] ** 2 + chunk["dy"] ** 2)
    path_length = distances.sum()

    start_x, start_y = chunk.iloc[0]["X Coordinate"], chunk.iloc[0]["Y Coordinate"]
    end_x, end_y = chunk.iloc[-1]["X Coordinate"], chunk.iloc[-1]["Y Coordinate"]
    end_to_end_dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # Straightness: ratio of beeline distance to actual path length
    straightness = end_to_end_dist / path_length if path_length > 0 else 0.0
    direction_changes = int((np.abs(chunk["angle"].diff()) > np.pi / 4).sum())
    click_count = int((chunk["Event Type"] == "click").sum())

    # Idle ratio: fraction of time spent at the 10th-percentile velocity
    velocity_clean = chunk["velocity"].dropna()
    if len(velocity_clean) > 0:
        low_velocity_threshold = np.percentile(velocity_clean, 10)
        idle_time = chunk.loc[chunk["velocity"] <= low_velocity_threshold, "dt"].sum()
        total_time = chunk["dt"].sum()
        idle_ratio = idle_time / total_time if total_time > 0 else 0.0
    else:
        idle_ratio = 0.0

    # Direction class: 8-sector compass classification (1=N, 2=NE, …)
    if end_to_end_dist > 0:
        direction_angle = np.arctan2(end_y - start_y, end_x - start_x)
        direction_degrees = (np.degrees(direction_angle) + 360) % 360
        direction_class = min(int(direction_degrees // 45) + 1, 8)
    else:
        direction_class = 0

    # Largest perpendicular deviation from the start→end line
    if len(chunk) >= 3 and end_to_end_dist > 0:
        x1, y1, x2, y2 = start_x, start_y, end_x, end_y
        line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        deviations = np.abs(
            (y2 - y1) * chunk["X Coordinate"]
            - (x2 - x1) * chunk["Y Coordinate"]
            + x2 * y1
            - y2 * x1
        ) / line_length
        largest_deviation = float(deviations.max())
    else:
        largest_deviation = 0.0

    # Sharp angles: count of inter-event angle changes exceeding threshold
    angle_changes = np.abs(chunk["angle"].diff())
    sharp_angles = int((angle_changes > 0.0005).sum())
    sum_of_angles = _safe_agg(angle_changes, np.sum)

    return {
        "mouse_path_length": float(path_length),
        "mouse_straightness": float(straightness),
        "mouse_velocity_mean": _safe_agg(chunk["velocity"], np.mean),
        "mouse_velocity_std": _safe_agg(chunk["velocity"], np.std),
        "mouse_velocity_max": _safe_agg(chunk["velocity"], np.max),
        "mouse_acceleration_mean": _safe_agg(chunk["acceleration"], np.mean),
        "mouse_acceleration_std": _safe_agg(chunk["acceleration"], np.std),
        "mouse_acceleration_max": _safe_agg(chunk["acceleration"], np.max),
        "mouse_jerk_mean": _safe_agg(chunk["jerk"], np.mean),
        "mouse_jerk_std": _safe_agg(chunk["jerk"], np.std),
        "mouse_direction_changes": direction_changes,
        "mouse_click_count": click_count,
        "mouse_idle_time_ratio": float(idle_ratio),
        "mouse_angular_velocity_mean": _safe_agg(chunk["angular_velocity"], np.mean),
        "mouse_angular_velocity_std": _safe_agg(chunk["angular_velocity"], np.std),
        "mouse_angular_velocity_min": _safe_agg(chunk["angular_velocity"], np.min),
        "mouse_angular_velocity_max": _safe_agg(chunk["angular_velocity"], np.max),
        "mouse_curvature_mean": _safe_agg(chunk["curvature"], np.mean),
        "mouse_curvature_std": _safe_agg(chunk["curvature"], np.std),
        "mouse_curvature_min": _safe_agg(chunk["curvature"], np.min),
        "mouse_curvature_max": _safe_agg(chunk["curvature"], np.max),
        "mouse_direction_class": direction_class,
        "mouse_sum_of_angles": float(sum_of_angles),
        "mouse_largest_deviation": float(largest_deviation),
        "mouse_sharp_angles": sharp_angles,
    }