"""
volatility.py

ARCH-LM test for conditional heteroskedasticity (volatility clustering).

A significant result means squared residuals are serially correlated —
i.e., the series exhibits ARCH effects / time-varying variance.  This is
common in financial return series and should be flagged for analysts who
might otherwise assume homoskedastic errors.
"""

import pandas as pd
from statsmodels.stats.diagnostic import het_arch


def arch_lm_test(series: pd.Series, lags: int = 5) -> dict:
    """
    Run the Engle ARCH-LM test on *series*.

    Parameters
    ----------
    series : pd.Series  — numeric values (NaNs are dropped)
    lags   : int        — number of lags for the auxiliary regression

    Returns
    -------
    dict with keys:
        lm_statistic, lm_pvalue, f_statistic, f_pvalue,
        lags, has_arch_effects
    """
    clean = series.dropna()
    n     = len(clean)
    if n < lags * 2 + 5:
        return {"error": f"Need at least {lags * 2 + 5} observations (have {n})"}
    try:
        lm, lm_p, f_stat, f_p = het_arch(clean, nlags=lags)
        return {
            "lm_statistic":    float(lm),
            "lm_pvalue":       float(lm_p),
            "f_statistic":     float(f_stat),
            "f_pvalue":        float(f_p),
            "lags":            lags,
            "has_arch_effects": bool(lm_p < 0.05),
        }
    except Exception as e:
        return {"error": str(e)}
