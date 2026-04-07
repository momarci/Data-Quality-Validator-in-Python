
"""
outliers.py

Outlier detection using IQR (default). Supports:

- IQR method
- Z-score (optional)
"""

import numpy as np
import pandas as pd


def detect_outliers(df: pd.DataFrame, method="iqr", iqr_multiplier=1.5, z_threshold=3.0):
    out = {}

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        series = df[col].dropna()

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            mask = (series < lower) | (series > upper)
            sample = series[mask]
            out[col] = {
                "method": "IQR",
                "lower_fence": lower,
                "upper_fence": upper,
                "outliers": sample.head(500).tolist(),
                "sample_capped": len(sample) > 500,
                "count": int(mask.sum()),
            }

        else:  # z-score
            z = (series - series.mean()) / series.std(ddof=0)
            mask = z.abs() > z_threshold
            sample = series[mask]
            out[col] = {
                "method": "Z-score",
                "threshold": z_threshold,
                "outliers": sample.head(500).tolist(),
                "sample_capped": len(sample) > 500,
                "count": int(mask.sum()),
            }

    return out
