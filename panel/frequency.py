
"""
frequency.py

Panel frequency analysis:
- Per-entity frequency
- Suspicious gaps
- Panel dominant frequency
"""

import pandas as pd
import numpy as np


def _entity_frequency_info(dates: pd.Series):
    """Compute frequency info for one entity."""
    dates = dates.dropna().sort_values().reset_index(drop=True)

    if len(dates) < 2:
        return {
            "unique_deltas": [],
            "most_common_delta": None,
            "delta_counts": {},
            "is_regular": False,
            "suspicious_gaps": pd.DataFrame(columns=["start", "end", "delta"])
        }

    deltas = dates.diff().dropna().reset_index(drop=True)
    deltas_rounded = deltas.dt.round("D")

    delta_counts = deltas_rounded.value_counts()
    unique_deltas = list(delta_counts.index)
    most_common = unique_deltas[0] if unique_deltas else None

    is_regular = len(unique_deltas) == 1

    median_delta = deltas_rounded.median()
    suspicious_mask = (deltas_rounded > 2 * median_delta).to_numpy()

    start_arr = dates.iloc[:-1].to_numpy()
    end_arr = dates.iloc[1:].to_numpy()
    delta_arr = deltas_rounded.to_numpy()

    suspicious = pd.DataFrame({
        "start": start_arr[suspicious_mask],
        "end": end_arr[suspicious_mask],
        "delta": delta_arr[suspicious_mask]
    })

    return {
        "unique_deltas": unique_deltas,
        "most_common_delta": most_common,
        "delta_counts": delta_counts.to_dict(),
        "is_regular": is_regular,
        "suspicious_gaps": suspicious
    }


def panel_frequency_analysis(df, entity_col, date_col):
    """Compute frequency per entity and overall panel frequency consistency."""

    if entity_col not in df.columns:
        raise ValueError(f"Entity column '{entity_col}' not found.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[entity_col, date_col]).sort_values([entity_col, date_col])

    entity_results = {}
    common_deltas = []

    for entity, grp in df.groupby(entity_col):
        info = _entity_frequency_info(grp[date_col])
        entity_results[entity] = info

        if info["most_common_delta"] is not None:
            common_deltas.append(info["most_common_delta"])

    # Determine panel dominant frequency
    if common_deltas:
        dominant_freq = pd.Series(common_deltas).mode().iloc[0]
    else:
        dominant_freq = None

    # Compare each entity’s frequency with dominant
    matching = []
    mismatching = []

    for entity, info in entity_results.items():
        if info["most_common_delta"] == dominant_freq:
            matching.append(entity)
        else:
            mismatching.append(entity)

    return {
        "entity_results": entity_results,
        "panel_dominant_frequency": dominant_freq,
        "consistency": {
            "consistent": len(mismatching) == 0,
            "matching": matching,
            "mismatching": mismatching
        }
    }
