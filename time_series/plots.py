
"""
plots.py

Time-series plots:
- Line plot
- STL components (provided series)

Memory-optimized for large datasets.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc


def time_series_plot(df, date_col, value_col, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    sns.lineplot(data=df, x=date_col, y=value_col, ax=ax)
    ax.set_title(f"{value_col} over time")
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        return None
    return fig


def stl_plot(stl_output, save_path=None):
    trend = stl_output["trend"]
    seasonal = stl_output["seasonal"]
    resid = stl_output["residual"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(trend.index, trend.values)
    axes[0].set_title("Trend")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(seasonal.index, seasonal.values)
    axes[1].set_title("Seasonal")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(resid.index, resid.values)
    axes[2].set_title("Residual")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_visible(False)  # Hide unused subplot

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        return None

    return fig

def acf_pacf_plots(series, save_dir, max_lags=12):
    """Generate ACF/PACF plots with memory optimization."""
    import os
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    os.makedirs(save_dir, exist_ok=True)

    # Subsample if series too long to prevent memory issues
    clean_series = series.dropna()
    if len(clean_series) > 2000:
        clean_series = clean_series.iloc[::max(1, len(clean_series)//2000)]

    fig1 = plot_acf(clean_series, lags=min(max_lags, len(clean_series)//2), ax=None)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "acf.png"), dpi=100, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_pacf(clean_series, lags=min(max_lags, len(clean_series)//2), method="ywm", ax=None)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "pacf.png"), dpi=100, bbox_inches="tight")
    plt.close(fig2)
    
    gc.collect()
