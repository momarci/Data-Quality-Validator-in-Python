
"""
detect_range.py

Detects:
- starting date
- ending date
- inferred frequency (using pandas + basic fallback)
"""

import pandas as pd
import numpy as np


def _fallback_freq(deltas):
    """
    Basic heuristic fallback frequency detector.
    """
    if len(deltas) == 0:
        return None

    # Convert deltas to integer days
    days = deltas.dt.days

    # Mode of day intervals
    mode = days.mode()

    if mode.empty:
        return None

    d = mode.iloc[0]

    if d == 1:
        return "D"
    if d == 7:
        return "W"
    if 28 <= d <= 31:
        return "M"
    if 89 <= d <= 92:
        return "Q"
    if 360 <= d <= 370:
        return "A"

    return None


def detect_date_range_and_frequency(df, date_col):
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()

    if len(dates) < 2:
        raise ValueError("Not enough valid date values to detect frequency.")

    start = dates.iloc[0]
    end = dates.iloc[-1]

    # Try pandas frequency inference
    freq_pd = pd.infer_freq(dates)
    deltas = dates.diff().dropna()

    if freq_pd:
        final_freq = freq_pd
    else:
        final_freq = _fallback_freq(deltas)

    return {
        "start": start,
        "end": end,
        "frequency_pandas": freq_pd,
        "frequency_fallback": final_freq,
        "final_frequency": final_freq,
        "n_observations": len(dates)
    }
