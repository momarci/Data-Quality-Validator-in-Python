
"""
plots.py

Cross-sectional plotting:
- scatter plots
- grouped bar plots
- boxplots

All plots save to a directory if provided.
Memory-optimized for large datasets.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import logging

logger = logging.getLogger(__name__)


def _save_or_return(figs, save_dir):
    if save_dir is None:
        return figs

    os.makedirs(save_dir, exist_ok=True)

    for i, fig in enumerate(figs, start=1):
        fig.savefig(os.path.join(save_dir, f"plot_{i}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    gc.collect()  # Force garbage collection after saving figures
    return []


def scatter_plots(df, cols=None, save_dir=None, max_plots=20):
    """Generate scatter plots with limit to prevent memory overload."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns

    figs = []
    cols = list(cols)
    
    # Limit number of scatter plot combinations
    plot_count = 0
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if plot_count >= max_plots:
                logger.warning(f"Scatter plots limited to {max_plots} due to dataset size")
                break
                
            fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
            
            # Downsample if too many points
            if len(df) > 10000:
                sample_df = df.sample(n=10000, random_state=42)
                ax.text(0.02, 0.98, f"(showing {len(sample_df)} of {len(df)} points)",
                       transform=ax.transAxes, fontsize=8, verticalalignment="top")
            else:
                sample_df = df
                
            sns.scatterplot(data=sample_df, x=cols[i], y=cols[j], ax=ax, alpha=0.5, s=20)
            ax.set_title(f"{cols[i]} vs {cols[j]}")
            figs.append(fig)
            plot_count += 1
        
        if plot_count >= max_plots:
            break

    return _save_or_return(figs, save_dir)


def grouped_bar_plots(df, cat_col, num_cols=None, save_dir=None):
    """Generate grouped bar plots."""
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns

    if not cat_col:
        return []

    figs = []

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
        sns.barplot(
            data=df,
            x=cat_col,
            y=col,
            errorbar="sd",
            ax=ax
        )
        ax.set_title(f"{col} by {cat_col}")
        ax.tick_params(axis="x", rotation=45)
        figs.append(fig)

    return _save_or_return(figs, save_dir)


def boxplots(df, num_cols=None, cat_col=None, save_dir=None):
    """Generate boxplots."""
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns

    figs = []

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y=col, ax=ax)
        else:
            sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(f"Boxplot: {col}")
        figs.append(fig)

    return _save_or_return(figs, save_dir)

