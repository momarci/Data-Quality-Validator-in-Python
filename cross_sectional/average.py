
"""
average.py

Computes descriptive location / dispersion / shape statistics per numeric column:
  mean, std, skewness (Fisher), excess kurtosis (normal = 0).
"""

import pandas as pd


def compute_average(df: pd.DataFrame):
    results = {}

    for col in df.select_dtypes(include=["number"]).columns:
        s = df[col].dropna()
        results[col] = {
            "mean":     float(s.mean()),
            "std":      float(s.std()),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),  # excess kurtosis (normal = 0)
        }

    return results
