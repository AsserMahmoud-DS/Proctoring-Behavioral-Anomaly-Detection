import pandas as pd

def merge_window_switch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Merge blur, focus, and tab_switch events into window_switch_events."""
    df = df.copy()
    cols_available = [c for c in ["blur_events", "focus_events", "tab_switch_events"]
                      if c in df.columns]
    if len(cols_available) >= 2:
        df["window_switch_events"] = df[cols_available].sum(axis=1)
        for col in cols_available:
            df.drop(columns=[col], inplace=True)
    elif len(cols_available) == 1:
        df["window_switch_events"] = df[cols_available[0]]
        df.drop(columns=cols_available, inplace=True)
    return df