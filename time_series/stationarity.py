
"""
stationarity.py

Unit-root / stationarity tests:
  - ADF  (Augmented Dickey-Fuller)
  - KPSS (level and trend)
  - PP   (Phillips-Perron) — requires statsmodels >= 0.14; gracefully skipped otherwise
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf(series):
    stat = adfuller(series, autolag="AIC")
    return {
        "statistic":      stat[0],
        "pvalue":         stat[1],
        "lags":           stat[2],
        "n_obs":          stat[3],
        "critical_values": stat[4],
        "stationary":     stat[1] < 0.05,
    }


def run_kpss_test(series, regression="c"):
    stat = kpss(series, regression=regression, nlags="legacy")
    return {
        "statistic":      stat[0],
        "pvalue":         stat[1],
        "lags":           stat[3],
        "critical_values": stat[2],
        "stationary":     stat[1] > 0.05,
    }


def run_pp_test(series):
    """Phillips-Perron test — robust to serial correlation without needing to
    specify lag length explicitly (uses Newey-West long-run variance)."""
    try:
        from statsmodels.tsa.stattools import phillips_perron
        stat = phillips_perron(series)
        return {
            "statistic":      float(stat[0]),
            "pvalue":         float(stat[1]),
            "lags":           int(stat[2]),
            "n_obs":          int(stat[3]),
            "critical_values": {k: float(v) for k, v in stat[4].items()},
            "stationary":     bool(stat[1] < 0.05),
        }
    except ImportError:
        return {"error": "Phillips-Perron requires statsmodels >= 0.14"}
    except Exception as e:
        return {"error": str(e)}


def stationarity_analysis(df, date_col, value_col):
    dates  = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")

    ts = pd.Series(values.values, index=dates).dropna().sort_index()

    if len(ts) < 10:
        raise ValueError("Not enough observations for stationarity tests.")

    adf        = run_adf(ts)
    kpss_level = run_kpss_test(ts, "c")
    kpss_trend = run_kpss_test(ts, "ct")
    pp         = run_pp_test(ts)

    return {
        "adf":        adf,
        "kpss_level": kpss_level,
        "kpss_trend": kpss_trend,
        "pp":         pp,
    }
