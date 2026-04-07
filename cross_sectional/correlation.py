"""
correlation.py

Pearson and Spearman correlation matrices for numeric columns.
Both are computed on the pairwise-complete observations (dropna on the full
numeric subset so every cell uses the same n).
"""

import pandas as pd


def compute_correlation(df: pd.DataFrame) -> dict:
    num_df = df.select_dtypes(include=["number"])
    cols   = num_df.columns.tolist()

    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}

    clean  = num_df.dropna()
    n_used = len(clean)

    if n_used < 3:
        return {"error": "Not enough complete rows for correlation analysis"}

    pearson  = clean.corr(method="pearson").round(4)
    spearman = clean.corr(method="spearman").round(4)

    return {
        "columns":  cols,
        "n_used":   n_used,
        "pearson":  pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }
