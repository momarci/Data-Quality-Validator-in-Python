"""
time_series/structural_breaks.py

Structural break detection for time-series data:
  - Chow test (split-sample F-test at candidate breakpoints)
  - CUSUM test (cumulative sum of recursive residuals)
  - Zivot-Andrews unit root test with structural break
  - Bai-Perron optimal multiple breakpoint test (DP + BIC)
  - Shift in means scan (Welch t-test across candidate splits)
  - Breakpoint plots saved to disk

All operations are vectorized and memory-safe for large series.
"""

import matplotlib
matplotlib.use("Agg")

import os
import gc
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chow Test
# ---------------------------------------------------------------------------

def chow_test(series: pd.Series, break_idx: int) -> dict:
    """
    Chow breakpoint test at a given index split.

    Fits y = a + b*t  on the full sample and on each sub-sample,
    then computes the F-statistic comparing restricted vs unrestricted RSS.
    """
    y = series.values.astype(float)
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    X = np.column_stack([np.ones(n), t])

    k = X.shape[1]

    # Full-sample OLS
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_full = np.sum((y - X @ beta_full) ** 2)

    # Sub-sample 1
    X1, y1 = X[:break_idx], y[:break_idx]
    if len(y1) <= k:
        return {"error": "Sub-sample 1 too small for Chow test"}
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    rss1 = np.sum((y1 - X1 @ beta1) ** 2)

    # Sub-sample 2
    X2, y2 = X[break_idx:], y[break_idx:]
    if len(y2) <= k:
        return {"error": "Sub-sample 2 too small for Chow test"}
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    rss2 = np.sum((y2 - X2 @ beta2) ** 2)

    rss_unrestricted = rss1 + rss2
    denom_df = n - 2 * k

    if denom_df <= 0 or rss_unrestricted == 0:
        return {"error": "Degenerate regression — cannot compute Chow F-stat"}

    f_stat = ((rss_full - rss_unrestricted) / k) / (rss_unrestricted / denom_df)
    p_value = 1.0 - sp_stats.f.cdf(f_stat, k, denom_df)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "break_index": int(break_idx),
        "n_obs": n,
        "significant": bool(p_value < 0.05),
    }


def chow_scan(series: pd.Series, n_candidates: int = 20) -> dict:
    """
    Scan across evenly-spaced candidate breakpoints and return the one
    with the highest Chow F-statistic (most likely structural break).
    """
    n = len(series)
    margin = max(10, int(n * 0.1))
    candidates = np.linspace(margin, n - margin, n_candidates, dtype=int)

    best = None
    results = []
    for bp in candidates:
        res = chow_test(series, bp)
        if "error" in res:
            continue
        results.append(res)
        if best is None or res["f_statistic"] > best["f_statistic"]:
            best = res

    return {
        "best_break": best,
        "all_candidates": results,
    }


# ---------------------------------------------------------------------------
# CUSUM Test
# ---------------------------------------------------------------------------

def cusum_test(series: pd.Series) -> dict:
    """
    OLS-CUSUM test: cumulative sum of recursive residuals normalised
    by the standard deviation.  If the path crosses the 5% significance
    bands (±0.948), the null of parameter stability is rejected.
    """
    y = series.values.astype(float)
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    X = np.column_stack([np.ones(n), t])
    k = X.shape[1]

    if n <= k + 2:
        return {"error": "Not enough observations for CUSUM test"}

    # Cap to 5 000 obs before the O(n²) expanding-OLS loop to prevent hangs
    MAX_CUSUM_OBS = 5_000
    cusum_subsampled_to = None
    if n > MAX_CUSUM_OBS:
        step = n // MAX_CUSUM_OBS
        y = y[::step]
        X = X[::step]
        n = len(y)
        cusum_subsampled_to = n

    # Recursive residuals (simplified: use expanding OLS)
    rec_resid = []
    for i in range(k + 1, n):
        Xi, yi = X[:i], y[:i]
        beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        pred = X[i] @ beta
        rec_resid.append(y[i] - pred)

    rec_resid = np.array(rec_resid)
    sigma = np.std(rec_resid, ddof=1) if len(rec_resid) > 1 else 1.0
    if sigma == 0:
        sigma = 1.0

    cusum = np.cumsum(rec_resid) / sigma
    cusum_norm = cusum / np.sqrt(len(rec_resid))

    # 5% significance boundary (Brownian bridge ≈ ±0.948)
    boundary_5pct = 0.948
    max_abs_cusum = float(np.max(np.abs(cusum_norm)))
    rejects = bool(max_abs_cusum > boundary_5pct)

    # Find index of maximum deviation
    max_dev_pos = int(np.argmax(np.abs(cusum_norm)))
    break_index = k + 1 + max_dev_pos  # map back to original index

    # Downsample stored path to max 500 points to keep results.json small
    cusum_stored = cusum_norm
    if len(cusum_norm) > 500:
        ds = len(cusum_norm) // 500
        cusum_stored = cusum_norm[::ds]

    return {
        "max_cusum": max_abs_cusum,
        "boundary_5pct": boundary_5pct,
        "rejects_stability": rejects,
        "cusum_values": cusum_stored.tolist(),
        "break_index": break_index,
        "n_recursive_residuals": len(rec_resid),
        "subsampled_to": cusum_subsampled_to,
    }


# ---------------------------------------------------------------------------
# Zivot-Andrews Test
# ---------------------------------------------------------------------------

def zivot_andrews_test(series: pd.Series, model: str = "c", trim: float = 0.15) -> dict:
    """
    Zivot-Andrews unit-root test with one structural break.

    model : 'c' (intercept break), 't' (trend break), 'ct' (both)
    trim  : fraction of endpoints to exclude

    Returns the breakpoint that minimises the ADF t-statistic.
    """
    try:
        from statsmodels.tsa.stattools import zivot_andrews as za
        res = za(series.values, maxlag=None, regression=model, autolag="AIC")
        return {
            "za_statistic": float(res[0]),
            "p_value": float(res[1]),
            "break_index": int(res[4]),
            "lags_used": int(res[2]),
            "critical_values": {k: float(v) for k, v in res[3].items()},
            "stationary_with_break": bool(res[1] < 0.05),
        }
    except ImportError:
        return {"error": "statsmodels >=0.14 required for Zivot-Andrews test"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Bai-Perron Test
# ---------------------------------------------------------------------------

def bai_perron_test(series: pd.Series, max_breaks: int = 5,
                    min_segment_frac: float = 0.10) -> dict:
    """
    Bai-Perron (1998/2003) optimal multiple breakpoint test — pure mean model.

    Uses dynamic programming with O(1) segment RSS (prefix-sum trick) to find
    the globally optimal partition into up to max_breaks+1 segments.
    The number of breaks is selected by minimising the BIC.

    Series is subsampled to ≤ 3 000 obs for speed on long series; break indices
    are scaled back to the original index space.

    Returns
    -------
    optimal_breaks  : int          number of breaks selected by BIC
    break_indices   : list[int]    positions in the *original* series
    break_dates     : list[str]    index labels at break positions (if DatetimeIndex)
    segments        : list[dict]   per-segment stats (start, end, n_obs, mean, std)
    bic_scores      : dict         BIC value for each candidate number of breaks
    """
    y_full = series.values.astype(float)
    n_full = len(y_full)
    idx = series.index  # keep for date mapping

    min_size_full = max(5, int(n_full * min_segment_frac))
    max_possible = n_full // min_size_full - 1
    max_k = max(1, min(max_breaks, max_possible))

    if n_full < 3 * min_size_full:
        return {"error": f"Series too short for Bai-Perron "
                         f"(need ≥ {3 * min_size_full} obs, have {n_full})"}

    # Subsample for speed
    MAX_OBS = 3000
    if n_full > MAX_OBS:
        step = n_full // MAX_OBS
        y = y_full[::step]
        idx_sub = idx[::step]
        scale = step          # break indices will be multiplied by this
    else:
        y = y_full
        idx_sub = idx
        scale = 1

    n = len(y)
    min_size = max(3, int(n * min_segment_frac))
    max_k = max(1, min(max_k, n // min_size - 1))

    # --- Prefix sums for O(1) segment RSS ---
    ps  = np.zeros(n + 1)   # prefix sum of y
    ps2 = np.zeros(n + 1)   # prefix sum of y²
    for i in range(n):
        ps[i + 1]  = ps[i]  + y[i]
        ps2[i + 1] = ps2[i] + y[i] ** 2

    def rss(i: int, j: int) -> float:
        """RSS for y[i:j] under the constant (mean) model."""
        length = j - i
        if length < 2:
            return 0.0
        s  = ps[j]  - ps[i]
        sq = ps2[j] - ps2[i]
        return sq - s * s / length

    INF = float("inf")

    # F[k][j] = min RSS with exactly k breaks in y[0:j]  (k+1 segments)
    # bp[k][j] = last break index achieving that minimum
    F  = np.full((max_k + 1, n + 1), INF)
    bp = np.zeros((max_k + 1, n + 1), dtype=int)

    # k = 0: no break, single segment
    for j in range(min_size, n + 1):
        F[0, j] = rss(0, j)

    # k ≥ 1: vectorised inner loop over candidate last-break t
    for k in range(1, max_k + 1):
        for j in range((k + 1) * min_size, n + 1):
            t_lo = k * min_size
            t_hi = j - min_size          # inclusive
            if t_lo > t_hi:
                continue

            t_arr = np.arange(t_lo, t_hi + 1)
            prev  = F[k - 1, t_arr]     # shape (m,)
            valid = prev < INF

            if not valid.any():
                continue

            # Vectorised RSS for last segment [t:j]
            s_seg  = ps[j]  - ps[t_arr]
            sq_seg = ps2[j] - ps2[t_arr]
            len_seg = j - t_arr
            rss_seg = sq_seg - s_seg ** 2 / len_seg

            total = prev + rss_seg
            total[~valid] = INF

            best_local = int(np.argmin(total))
            F[k, j]  = total[best_local]
            bp[k, j] = t_arr[best_local]

    # --- BIC selection ---
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic_scores: dict = {}
    for k in range(0, max_k + 1):
        r = F[k, n]
        if r < INF and r > 0:
            bic_scores[k] = float(n * np.log(r / n) + k * np.log(n))

    if not bic_scores:
        return {"error": "BIC could not be computed — all DP states are infeasible"}

    optimal_k = int(min(bic_scores, key=bic_scores.get))

    # --- Backtrack break indices ---
    def backtrack(k: int, j: int) -> list:
        if k == 0:
            return []
        t = int(bp[k, j])
        return backtrack(k - 1, t) + [t]

    raw_bps = backtrack(optimal_k, n)          # indices in subsampled space
    orig_bps = [int(t * scale) for t in raw_bps]   # map back to original space

    # --- Segment stats (in original space) ---
    boundaries = [0] + orig_bps + [n_full]
    segments = []
    for i in range(len(boundaries) - 1):
        seg = y_full[boundaries[i]: boundaries[i + 1]]
        date_start = str(idx[boundaries[i]])
        date_end   = str(idx[min(boundaries[i + 1] - 1, n_full - 1)])
        segments.append({
            "start_index": int(boundaries[i]),
            "end_index":   int(boundaries[i + 1] - 1),
            "date_start":  date_start,
            "date_end":    date_end,
            "n_obs":       int(len(seg)),
            "mean":        float(seg.mean()),
            "std":         float(seg.std(ddof=1)) if len(seg) > 1 else 0.0,
        })

    # Break dates
    break_dates = [str(idx[min(b, n_full - 1)]) for b in orig_bps]

    return {
        "optimal_breaks": optimal_k,
        "break_indices":  orig_bps,
        "break_dates":    break_dates,
        "segments":       segments,
        "bic_scores":     {str(k): round(v, 4) for k, v in bic_scores.items()},
        "total_rss":      float(F[optimal_k, n]),
        "subsampled":     n_full > MAX_OBS,
    }


# ---------------------------------------------------------------------------
# Shift-in-Means Test
# ---------------------------------------------------------------------------

def shift_in_means_test(series: pd.Series, n_candidates: int = 50) -> dict:
    """
    Scan for the single most significant shift in mean using Welch's t-test.

    For each candidate split point the series is divided into two sub-samples
    and an independent two-sample t-test (unequal variances) is run.  The split
    with the largest |t-statistic| is reported as the most likely mean-shift
    breakpoint.

    Returns
    -------
    best_break   : dict  — break_index, t_statistic, p_value,
                          mean_before, mean_after, shift_magnitude, significant
    n_candidates : int
    """
    y = series.values.astype(float)
    n = len(y)

    margin = max(5, int(n * 0.10))
    candidates = np.linspace(margin, n - margin, n_candidates, dtype=int)
    candidates = np.unique(candidates)

    best: dict = {}
    best_abs_t = -1.0

    for bp in candidates:
        s1 = y[:bp]
        s2 = y[bp:]
        if len(s1) < 3 or len(s2) < 3:
            continue

        t_stat, p_val = sp_stats.ttest_ind(s1, s2, equal_var=False)
        if not np.isfinite(t_stat):
            continue

        entry = {
            "break_index":      int(bp),
            "t_statistic":      float(t_stat),
            "p_value":          float(p_val),
            "mean_before":      float(s1.mean()),
            "mean_after":       float(s2.mean()),
            "shift_magnitude":  float(s2.mean() - s1.mean()),
            "significant":      bool(p_val < 0.05),
        }

        if abs(t_stat) > best_abs_t:
            best_abs_t = abs(t_stat)
            best = entry

    if not best:
        return {"error": "Could not compute shift-in-means test"}

    # Attach break date if index is datetime-like
    try:
        best["break_date"] = str(series.index[best["break_index"]])
    except Exception:
        pass

    return {
        "best_break":    best,
        "n_candidates":  len(candidates),
    }


# ---------------------------------------------------------------------------
# All-in-one runner
# ---------------------------------------------------------------------------

def run_structural_break_tests(df, date_col, value_col) -> dict:
    """
    Run all structural break tests on a time series.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    ts = pd.Series(values.values, index=dates).dropna().sort_index()

    if len(ts) < 30:
        return {"error": "Need at least 30 observations for structural break tests"}

    # Subsample for speed on very large series (>50k)
    if len(ts) > 50000:
        step = len(ts) // 50000
        ts = ts.iloc[::step]
        logger.info(f"Subsampled to {len(ts)} obs for structural break tests")

    results = {}

    # Chow scan
    try:
        results["chow"] = chow_scan(ts)
    except Exception as e:
        results["chow"] = {"error": str(e)}

    # CUSUM
    try:
        results["cusum"] = cusum_test(ts)
    except Exception as e:
        results["cusum"] = {"error": str(e)}

    # Zivot-Andrews
    try:
        results["zivot_andrews"] = zivot_andrews_test(ts, model="c")
    except Exception as e:
        results["zivot_andrews"] = {"error": str(e)}

    # Bai-Perron
    try:
        results["bai_perron"] = bai_perron_test(ts)
        logger.info(f"  Bai-Perron: {results['bai_perron'].get('optimal_breaks', '?')} break(s)")
    except Exception as e:
        results["bai_perron"] = {"error": str(e)}

    # Shift in means
    try:
        results["shift_in_means"] = shift_in_means_test(ts)
    except Exception as e:
        results["shift_in_means"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_structural_breaks(df, date_col, value_col, break_results, save_dir):
    """
    Generate and save structural break visualisation plots.

    1. Time series with detected breakpoints marked.
    2. CUSUM path with significance bands.
    3. Chow F-statistic scan across candidate breakpoints.
    """
    os.makedirs(save_dir, exist_ok=True)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    ts = pd.Series(values.values, index=dates).dropna().sort_index()

    if len(ts) > 50000:
        step = len(ts) // 50000
        ts = ts.iloc[::step]

    # Collect break indices
    break_indices = []
    labels = []

    chow = break_results.get("chow", {})
    if isinstance(chow, dict) and "best_break" in chow and chow["best_break"]:
        bb = chow["best_break"]
        if bb.get("significant") and "break_index" in bb:
            break_indices.append(bb["break_index"])
            labels.append("Chow")

    cusum = break_results.get("cusum", {})
    if isinstance(cusum, dict) and cusum.get("rejects_stability") and "break_index" in cusum:
        break_indices.append(cusum["break_index"])
        labels.append("CUSUM")

    za = break_results.get("zivot_andrews", {})
    if isinstance(za, dict) and za.get("stationary_with_break") and "break_index" in za:
        break_indices.append(za["break_index"])
        labels.append("ZA")

    sim = break_results.get("shift_in_means", {})
    sim_bb = sim.get("best_break", {}) if isinstance(sim, dict) else {}
    if sim_bb.get("significant") and "break_index" in sim_bb:
        break_indices.append(sim_bb["break_index"])
        labels.append("Shift")

    # --- Plot 1: Series with breakpoints ---
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    ax.plot(ts.index, ts.values, linewidth=0.8, color="#2c7bb6")
    ax.set_title(f"{value_col} — Structural Breakpoints", fontsize=13, fontweight="bold")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.3)

    colors = ["#d7191c", "#fdae61", "#1a9641"]
    for i, (bi, lbl) in enumerate(zip(break_indices, labels)):
        if 0 <= bi < len(ts):
            ax.axvline(ts.index[bi], color=colors[i % len(colors)], linestyle="--",
                       linewidth=1.5, label=f"{lbl} break")
    if labels:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "structural_breaks.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: CUSUM path ---
    if isinstance(cusum, dict) and "cusum_values" in cusum:
        cv = np.array(cusum["cusum_values"])
        bnd = cusum.get("boundary_5pct", 0.948)

        fig2, ax2 = plt.subplots(figsize=(12, 4), dpi=100)
        ax2.plot(cv, linewidth=0.9, color="#2c7bb6")
        ax2.axhline(bnd, color="#d7191c", linestyle="--", linewidth=1, label="+5% boundary")
        ax2.axhline(-bnd, color="#d7191c", linestyle="--", linewidth=1, label="-5% boundary")
        ax2.axhline(0, color="grey", linewidth=0.5)
        ax2.fill_between(range(len(cv)), -bnd, bnd, alpha=0.08, color="#d7191c")
        ax2.set_title("CUSUM Test — Recursive Residuals", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Recursive observation index")
        ax2.set_ylabel("Normalised CUSUM")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, "cusum_path.png"), dpi=100, bbox_inches="tight")
        plt.close(fig2)

    # --- Plot 3: Chow F-scan ---
    if isinstance(chow, dict) and "all_candidates" in chow:
        cands = chow["all_candidates"]
        if cands:
            bp_x = [c["break_index"] for c in cands]
            f_y = [c["f_statistic"] for c in cands]
            p_y = [c["p_value"] for c in cands]

            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 7), dpi=100, sharex=True)

            ax3a.bar(bp_x, f_y, width=max(1, (bp_x[-1]-bp_x[0])/(len(bp_x)*1.5)),
                     color="#2c7bb6", alpha=0.8)
            ax3a.set_ylabel("F-statistic")
            ax3a.set_title("Chow Test — F-statistic Scan", fontsize=13, fontweight="bold")
            ax3a.grid(True, alpha=0.3)

            ax3b.plot(bp_x, p_y, marker="o", markersize=4, color="#d7191c")
            ax3b.axhline(0.05, color="green", linestyle="--", linewidth=1, label="p = 0.05")
            ax3b.set_ylabel("p-value")
            ax3b.set_xlabel("Candidate break index")
            ax3b.legend(loc="best", fontsize=9)
            ax3b.grid(True, alpha=0.3)

            fig3.tight_layout()
            fig3.savefig(os.path.join(save_dir, "chow_scan.png"), dpi=100, bbox_inches="tight")
            plt.close(fig3)

    # --- Plot 4: Bai-Perron segments ---
    bp_res = break_results.get("bai_perron", {})
    if isinstance(bp_res, dict) and "segments" in bp_res and bp_res.get("optimal_breaks", 0) > 0:
        segments = bp_res["segments"]
        orig_bps = bp_res.get("break_indices", [])

        fig4, ax4 = plt.subplots(figsize=(12, 5), dpi=100)
        ax4.plot(ts.index, ts.values, linewidth=0.8, color="#aab4c8", zorder=1)

        palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        boundaries = [0] + orig_bps + [len(ts)]
        for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            seg_idx = ts.index[lo: hi]
            seg_val = ts.values[lo: hi]
            col = palette[i % len(palette)]
            ax4.plot(seg_idx, seg_val, linewidth=1.4, color=col)
            ax4.axhline(np.mean(seg_val), xmin=lo / len(ts), xmax=hi / len(ts),
                        color=col, linestyle="--", linewidth=1.5, alpha=0.9)

        for bi in orig_bps:
            if 0 <= bi < len(ts):
                ax4.axvline(ts.index[bi], color="black", linestyle=":", linewidth=1.2, alpha=0.7)

        n_breaks = bp_res["optimal_breaks"]
        ax4.set_title(f"Bai-Perron — {n_breaks} Optimal Break(s), Segment Means",
                      fontsize=13, fontweight="bold")
        ax4.set_xlabel(date_col)
        ax4.set_ylabel(value_col)
        ax4.grid(True, alpha=0.25)
        fig4.tight_layout()
        fig4.savefig(os.path.join(save_dir, "bai_perron_segments.png"),
                     dpi=100, bbox_inches="tight")
        plt.close(fig4)

        # BIC scores bar chart
        bic_scores = bp_res.get("bic_scores", {})
        if len(bic_scores) > 1:
            ks    = [int(k) for k in bic_scores]
            bics  = [bic_scores[str(k)] for k in ks]
            opt_k = bp_res["optimal_breaks"]

            fig5, ax5 = plt.subplots(figsize=(7, 4), dpi=100)
            bar_colors = ["#2c7bb6" if k != opt_k else "#d7191c" for k in ks]
            ax5.bar(ks, bics, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax5.set_xlabel("Number of breaks (k)")
            ax5.set_ylabel("BIC")
            ax5.set_title("Bai-Perron — BIC Model Selection\n(red = selected)",
                          fontsize=12, fontweight="bold")
            ax5.set_xticks(ks)
            ax5.grid(True, axis="y", alpha=0.3)
            fig5.tight_layout()
            fig5.savefig(os.path.join(save_dir, "bai_perron_bic.png"),
                         dpi=100, bbox_inches="tight")
            plt.close(fig5)

    # --- Plot 5: Shift in means ---
    if isinstance(sim, dict) and "best_break" in sim:
        bb = sim["best_break"]
        bi = bb.get("break_index")
        if bi is not None and 0 < bi < len(ts):
            fig6, ax6 = plt.subplots(figsize=(12, 5), dpi=100)
            ax6.plot(ts.index, ts.values, linewidth=0.9, color="#aab4c8", zorder=1)

            # Before segment
            ax6.plot(ts.index[:bi], ts.values[:bi], linewidth=1.5, color="#2c7bb6", label="Before")
            ax6.axhline(bb["mean_before"], color="#2c7bb6", linestyle="--",
                        linewidth=1.4, alpha=0.85,
                        label=f"Mean before = {bb['mean_before']:.4g}")

            # After segment
            ax6.plot(ts.index[bi:], ts.values[bi:], linewidth=1.5, color="#d7191c", label="After")
            ax6.axhline(bb["mean_after"], color="#d7191c", linestyle="--",
                        linewidth=1.4, alpha=0.85,
                        label=f"Mean after = {bb['mean_after']:.4g}")

            ax6.axvline(ts.index[bi], color="black", linestyle=":", linewidth=1.5,
                        label=f"Break (idx={bi})")

            sig_str = f"  p = {bb['p_value']:.4f}{'  **' if bb['p_value'] < 0.05 else ''}"
            ax6.set_title(f"Shift in Means — t = {bb['t_statistic']:.3f}{sig_str}",
                          fontsize=13, fontweight="bold")
            ax6.set_xlabel(date_col)
            ax6.set_ylabel(value_col)
            ax6.legend(fontsize=9, loc="best")
            ax6.grid(True, alpha=0.25)
            fig6.tight_layout()
            fig6.savefig(os.path.join(save_dir, "shift_in_means.png"),
                         dpi=100, bbox_inches="tight")
            plt.close(fig6)

    gc.collect()
