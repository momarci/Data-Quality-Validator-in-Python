
"""
detect_range.py

Detect per-entity start and end dates,
and global panel date range.
"""

import pandas as pd


def detect_panel_date_ranges(df, entity_col, date_col):
    if entity_col not in df.columns:
        raise ValueError(f"Entity column '{entity_col}' not found.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[entity_col, date_col]).sort_values([entity_col, date_col])

    per_entity = {}
    for entity, grp in df.groupby(entity_col):
        dates = grp[date_col].dropna().sort_values()
        if len(dates) == 0:
            continue
        per_entity[entity] = {
            "start": dates.iloc[0],
            "end": dates.iloc[-1],
            "n_obs": len(dates)
        }

    # Global range
    all_dates = df[date_col].dropna().sort_values()
    global_start = all_dates.iloc[0] if len(all_dates) else None
    global_end = all_dates.iloc[-1] if len(all_dates) else None

    return {
        "global_start": global_start,
        "global_end": global_end,
        "entity_stats": per_entity
    }
