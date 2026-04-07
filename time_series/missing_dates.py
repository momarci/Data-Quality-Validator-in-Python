
"""
missing_dates.py

Detect missing timestamps based on frequency.
"""

import pandas as pd


def detect_missing_dates(df, date_col, inferred_freq=None):
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()

    if inferred_freq is None:
        raise ValueError("Missing date detection requires known frequency.")

    start = dates.iloc[0]
    end = dates.iloc[-1]

    # Generate expected dates
    try:
        expected = pd.date_range(start=start, end=end, freq=inferred_freq)
    except Exception:
        raise ValueError(f"Frequency '{inferred_freq}' not valid for date_range.")

    expected = pd.DatetimeIndex(expected)
    actual = pd.DatetimeIndex(dates.unique())

    missing = expected.difference(actual)

    return {
        "start": start,
        "end": end,
        "frequency": inferred_freq,
        "missing_dates": list(missing),
        "missing_count": len(missing),
        "expected_length": len(expected),
        "actual_length": len(actual)
    }
