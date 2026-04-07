"""
normality.py

Per-column normality tests:
  - Jarque-Bera  (valid for any n; tests skewness + excess kurtosis jointly)
  - Shapiro-Wilk (exact for small n; subsampled when n > max_shapiro_n)
"""

import pandas as pd
from scipy import stats


def test_normality(df: pd.DataFrame, max_shapiro_n: int = 5_000) -> dict:
    out = {}
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        n = len(series)
        if n < 3:
            out[col] = {"error": "too few observations"}
            continue

        col_result = {}

        # Jarque-Bera — works for any sample size
        try:
            jb_stat, jb_p = stats.jarque_bera(series)
            col_result["jarque_bera"] = {
                "statistic": float(jb_stat),
                "pvalue":    float(jb_p),
                "is_normal": bool(jb_p >= 0.05),
            }
        except Exception as e:
            col_result["jarque_bera"] = {"error": str(e)}

        # Shapiro-Wilk — subsample when n is large
        try:
            subsampled = n > max_shapiro_n
            sw_series  = series.sample(max_shapiro_n, random_state=42) if subsampled else series
            sw_stat, sw_p = stats.shapiro(sw_series)
            col_result["shapiro_wilk"] = {
                "statistic":  float(sw_stat),
                "pvalue":     float(sw_p),
                "is_normal":  bool(sw_p >= 0.05),
                "subsampled": subsampled,
                "n_used":     int(len(sw_series)),
            }
        except Exception as e:
            col_result["shapiro_wilk"] = {"error": str(e)}

        out[col] = col_result
    return out
