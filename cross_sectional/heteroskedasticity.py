
"""
heteroskedasticity.py

Runs:
- Breusch-Pagan test
- White test
"""

import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan, het_white


def run_heteroskedasticity_tests(df: pd.DataFrame, y_col: str, x_cols: list):
    if y_col not in df.columns:
        raise ValueError(f"Y column '{y_col}' not in dataframe.")
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"X column '{c}' not found.")

    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").dropna()

    # Align indexes
    combined = pd.concat([y, X], axis=1).dropna()
    y_clean = combined[y_col]
    X_clean = combined[x_cols]

    Xc = add_constant(X_clean)
    model = OLS(y_clean, Xc).fit()

    bp_lm, bp_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)
    white_lm, white_pval, white_f, white_f_pval = het_white(model.resid, model.model.exog)

    return {
        "breusch_pagan": {
            "lm": bp_lm,
            "lm_pvalue": bp_pval,
            "f_stat": f_stat,
            "f_pvalue": f_pval
        },
        "white_test": {
            "lm": white_lm,
            "lm_pvalue": white_pval,
            "f_stat": white_f,
            "f_pvalue": white_f_pval
        }
    }
