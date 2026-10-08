"""Ordered behavioral feature schema and fixed preprocessing blocks."""

SOURCE_FEATURES = (
    "elapsed_time", "copy_events", "paste_events", "mouse_path_length",
    "mouse_straightness", "mouse_velocity_mean", "mouse_acceleration_mean",
    "mouse_jerk_mean", "mouse_direction_changes", "mouse_click_count",
    "mouse_idle_time_ratio", "mouse_angular_velocity_mean",
    "mouse_angular_velocity_std", "mouse_angular_velocity_min",
    "mouse_curvature_mean", "mouse_curvature_std", "mouse_curvature_min",
    "mouse_direction_class", "mouse_sum_of_angles", "mouse_largest_deviation",
    "mouse_sharp_angles", "keyboard_typing_rate", "keyboard_burst_count",
    "keyboard_pause_count", "window_switch_events",
)
CONTINUOUS = (
    "elapsed_time", "mouse_path_length", "mouse_velocity_mean",
    "mouse_acceleration_mean", "mouse_jerk_mean", "mouse_angular_velocity_mean",
    "mouse_angular_velocity_std", "mouse_angular_velocity_min",
    "mouse_curvature_mean", "mouse_curvature_std", "mouse_curvature_min",
    "mouse_largest_deviation", "keyboard_typing_rate",
)
SIGNED = ("mouse_jerk_mean", "mouse_angular_velocity_mean", "mouse_angular_velocity_min")
COUNT_LOG = ("keyboard_burst_count", "keyboard_pause_count", "window_switch_events")
UNSCALED = ("copy_events", "paste_events", "window_switch_events")
DIRECTION = "mouse_direction_class"

NUMERICAL = tuple(feature for feature in SOURCE_FEATURES if feature != DIRECTION)
SCALED = tuple(feature for feature in NUMERICAL if feature not in UNSCALED)
COUNTS = (
    "copy_events", "paste_events", "mouse_direction_changes", "mouse_click_count",
    "mouse_sharp_angles", *COUNT_LOG,
)
OUTPUT_FEATURES = (*NUMERICAL, *(f"{DIRECTION}_{category}" for category in range(9)))
