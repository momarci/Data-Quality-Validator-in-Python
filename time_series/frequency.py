
"""
frequency.py

Advanced frequency diagnostics:
- unique deltas
- most common delta
- suspicious gaps
- expected vs observed frequency check
"""

import pandas as pd
import numpy as np


def analyze_frequency(df: pd.DataFrame, date_col: str, expected_freq=None):
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().reset_index(drop=True)

    if len(dates) < 2:
        raise ValueError("Too few valid date values for frequency analysis.")

    deltas = dates.diff().dropna().reset_index(drop=True)
    deltas_rounded = deltas.dt.round("D")

    # Unique deltas
    delta_counts = deltas_rounded.value_counts()
    unique_deltas = list(delta_counts.index)

    most_common = unique_deltas[0] if unique_deltas else None

    # Regularity
    is_regular = len(unique_deltas) == 1

    # Suspicious gaps
    median_delta = deltas_rounded.median()
    suspicious_mask = (deltas_rounded > 2 * median_delta).to_numpy()

    start_arr = dates.iloc[:-1].to_numpy()
    end_arr = dates.iloc[1:].to_numpy()
    del_arr = deltas_rounded.to_numpy()

    suspicious_gaps = pd.DataFrame({
        "start": start_arr[suspicious_mask],
        "end": end_arr[suspicious_mask],
        "delta": del_arr[suspicious_mask]
    })

    # Expected vs observed freq
    expected_match = None
    if expected_freq and most_common:
        try:
            expected_ns = pd.tseries.frequencies.to_offset(expected_freq).nanos
            observed_ns = most_common.to_timedelta64().astype("timedelta64[ns]").astype(int)
            expected_match = (expected_ns == observed_ns)
        except Exception:
            expected_match = None

    return {
        "unique_deltas": unique_deltas,
        "most_common_delta": most_common,
        "delta_counts": delta_counts.to_dict(),
        "is_regular": is_regular,
        "suspicious_gaps": suspicious_gaps,
        "expected_vs_observed_match": expected_match,
        "n_observations": len(dates)
    }
