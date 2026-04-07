
"""
min_max.py

Computes min/max for each numeric or datetime column.
"""

import pandas as pd


def compute_min_max(df: pd.DataFrame):
    results = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            results[col] = {
                "min": df[col].min(),
                "max": df[col].max()
            }

    return results
