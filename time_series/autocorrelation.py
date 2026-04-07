
"""
autocorrelation.py

ACF/PACF and Ljung-Box diagnostics.
"""

import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def compute_acf_pacf(df, date_col, value_col, nlags=12):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")

    ts = pd.Series(values.values, index=dates).dropna().sort_index()

    if len(ts) < 10:
        raise ValueError("Too few observations for ACF/PACF.")

    return {
        "acf": acf(ts, nlags=min(nlags, len(ts)//2), fft=True).tolist(),
        "pacf": pacf(ts, nlags=min(nlags, len(ts)//2)).tolist()
    }


def ljung_box_test(df, date_col, value_col, lags=12):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")

    ts = pd.Series(values.values, index=dates).dropna().sort_index()

    if len(ts) < 10:
        raise ValueError("Too few observations for Ljung-Box test.")

    res = acorr_ljungbox(ts, lags=[lags], return_df=True)
    return {
        "lag": lags,
        "lb_stat": res["lb_stat"].iloc[0],
        "pvalue": res["lb_pvalue"].iloc[0]
    }
