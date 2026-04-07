
"""
median.py

Computes the median of numeric columns.
"""

import pandas as pd


def compute_median(df: pd.DataFrame):
    results = {}

    for col in df.select_dtypes(include=["number"]).columns:
        results[col] = {"median": df[col].median()}

    return results
