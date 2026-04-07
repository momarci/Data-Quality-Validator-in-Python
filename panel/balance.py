"""
balance.py

Two diagnostics for panel datasets:

1. Panel balance check
   Counts distinct (entity, date) pairs and compares against the
   theoretical full-grid count (n_entities × n_time_periods).
   An unbalanced panel has missing combinations.

2. Within / between variance decomposition
   For each numeric column decomposes overall variance into:
     - Between-entity variance  (how much means differ across entities)
     - Within-entity variance   (how much values fluctuate within an entity)
   Expressed both in absolute terms and as % of overall variance.
"""

import pandas as pd
import numpy as np


def check_panel_balance(df: pd.DataFrame, entity_col: str, date_col: str) -> dict:
    n_entities = int(df[entity_col].nunique())
    n_periods  = int(df[date_col].nunique())
    expected   = n_entities * n_periods
    actual     = int(df.groupby([entity_col, date_col]).ngroups)
    missing    = expected - actual

    return {
        "n_entities":           n_entities,
        "n_time_periods":       n_periods,
        "expected_obs":         expected,
        "actual_obs":           actual,
        "is_balanced":          bool(actual == expected),
        "missing_combinations": missing,
        "balance_ratio":        round(actual / expected, 4) if expected > 0 else 0.0,
    }


def compute_variance_decomposition(df: pd.DataFrame,
                                   entity_col: str,
                                   numeric_cols: list) -> dict:
    out = {}
    for col in numeric_cols:
        sub = df[[entity_col, col]].dropna()
        if len(sub) < 2 or sub[entity_col].nunique() < 2:
            continue
        overall_var = float(sub[col].var())
        if overall_var == 0:
            continue
        entity_means = sub.groupby(entity_col)[col].mean()
        between_var  = float(entity_means.var())
        within_var   = float(sub.groupby(entity_col)[col].var().mean())
        out[col] = {
            "overall_variance": round(overall_var, 6),
            "between_variance": round(between_var, 6),
            "within_variance":  round(within_var, 6),
            "between_pct":      round(between_var / overall_var * 100, 2),
            "within_pct":       round(within_var  / overall_var * 100, 2),
        }
    return out
