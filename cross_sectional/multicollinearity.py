
"""
multicollinearity.py

Computes:
- Variance Inflation Factor (VIF)
- Condition number of design matrix
"""

import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, columns: list):
    out = {}
    X = df[columns].dropna().copy()

    # Remove constant columns
    zero_cols = [c for c in columns if X[c].std(ddof=0) == 0]
    for c in zero_cols:
        columns.remove(c)

    X = X[columns]
    Xc = add_constant(X)

    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        try:
            out[col] = float(variance_inflation_factor(Xc.values, i))
        except (ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
            out[col] = None

    return out


def compute_condition_number(df: pd.DataFrame, columns: list):
    X = df[columns].dropna().copy()
    Xc = add_constant(X)

    svals = np.linalg.svd(Xc, compute_uv=False)
    cond_num = float(svals[0] / max(svals[-1], 1e-12))

    return cond_num


def run_multicollinearity_checks(df, columns):
    return {
        "vif": compute_vif(df, columns.copy()),
        "condition_number": compute_condition_number(df, columns.copy())
    }
