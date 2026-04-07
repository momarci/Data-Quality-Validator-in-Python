
"""
stl_decomposition.py

STL decomposition with GUI override support.
"""

import numpy as np
from statsmodels.tsa.seasonal import STL
import pandas as pd


def _seasonal_strength(seasonal, residual) -> float:
    """Hyndman & Athanasopoulos (2021) seasonal strength:
    F_S = max(0,  1 - Var(R) / Var(S + R))
    Range [0, 1] — 1 means variance is entirely explained by the seasonal component."""
    var_r  = float(np.var(residual))
    var_sr = float(np.var(seasonal + residual))
    if var_sr == 0:
        return 0.0
    return round(max(0.0, 1.0 - var_r / var_sr), 4)


def infer_seasonal_period(freq):
    if freq is None:
        return None

    freq = freq.upper()

    defaults = {
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "A": 1
    }
    return defaults.get(freq)


def run_stl(df, date_col, value_col, period=None, inferred_freq=None, gui_override_callback=None):
    series = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")

    ts = pd.Series(values.values, index=series).dropna().sort_index()

    if len(ts) < 8:
        raise ValueError("Not enough observations for STL decomposition.")

    # Determine period
    if period is None:
        period = infer_seasonal_period(inferred_freq)

    if period is None:
        if gui_override_callback is None:
            raise ValueError("Cannot infer STL seasonal period and no override callback provided.")

        override = gui_override_callback({
            "title": "STL Seasonal Period",
            "fields": {
                "seasonal_period": {
                    "type": "text",
                    "label": "Enter seasonal period (>=2):"
                }
            }
        })
        period = int(override["seasonal_period"])

    if period < 2:
        raise ValueError("Seasonal period must be >= 2 for STL.")

    stl = STL(ts, period=period, robust=True)
    result = stl.fit()

    return {
        "period":            period,
        "seasonal_strength": _seasonal_strength(result.seasonal, result.resid),
        "trend":             result.trend,
        "seasonal":          result.seasonal,
        "residual":          result.resid,
    }
