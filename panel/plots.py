
"""
plots.py

Panel plotting utilities:
- per-entity line plots
- faceted time-series
- panel average
- cross-sectional snapshot at specific date

Memory-optimized for large panel datasets.
"""
import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import logging

logger = logging.getLogger(__name__)


def _save_or_return(figs, save_dir):
    if save_dir is None:
        return figs

    os.makedirs(save_dir, exist_ok=True)
    for i, fig in enumerate(figs, start=1):
        fig.savefig(os.path.join(save_dir, f"panel_plot_{i}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    
    gc.collect()  # Force garbage collection
    return []


def per_entity_series(df, entity_col, date_col, value_col, save_dir=None, max_entities=20):
    """Per-entity plots with limit to prevent memory overload."""
    figs = []
    
    unique_entities = df[entity_col].nunique()
    if unique_entities > max_entities:
        logger.warning(f"Too many entities ({unique_entities}). Plotting top {max_entities} only.")
        # Get top entities by data frequency
        top_entities = df[entity_col].value_counts().head(max_entities).index
        df = df[df[entity_col].isin(top_entities)]
    
    for entity, grp in df.groupby(entity_col):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
        sns.lineplot(data=grp, x=date_col, y=value_col, ax=ax)
        ax.set_title(f"{entity_col} = {entity}")
        ax.grid(True, alpha=0.3)
        figs.append(fig)

    return _save_or_return(figs, save_dir)


def faceted_panel_plot(df, entity_col, date_col, value_col, wrap=4, save_dir=None, max_entities=16):
    """
    Faceted panel time-series plot with entity limit.
    Ensures the output is ALWAYS a Matplotlib figure so savefig() works.
    """
    unique_entities = df[entity_col].nunique()
    if unique_entities > max_entities:
        logger.warning(f"Too many entities ({unique_entities}). Plotting top {max_entities} only.")
        top_entities = df[entity_col].value_counts().head(max_entities).index
        df = df[df[entity_col].isin(top_entities)]

    figs = []

    # TRY MODERN seaborn.objects
    try:
        import seaborn.objects as so

        p = (
            so.Plot(df, x=date_col, y=value_col)
            .facet(col=entity_col, wrap=wrap)
            .add(so.Line())
        )

        # p.plot() returns a "Plotter", not a Figure
        # Extract the Matplotlib figure manually:
        rendered = p.plot()
        fig = rendered.figure  # <- This is the REAL Matplotlib figure

    except Exception:
        # FALLBACK: classic seaborn FacetGrid
        g = sns.FacetGrid(df, col=entity_col, col_wrap=wrap, sharex=True, sharey=True, height=3)
        g.map_dataframe(sns.lineplot, date_col, value_col)
        fig = g.fig  # FacetGrid.fig is a proper Matplotlib figure

    figs.append(fig)
    return _save_or_return(figs, save_dir)

def panel_average(df, date_col, value_col, save_dir=None):
    """Panel average over time."""
    avg = (
        df.groupby(date_col)[value_col]
        .mean()
        .reset_index()
        .sort_values(date_col)
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    sns.lineplot(data=avg, x=date_col, y=value_col, ax=ax)
    ax.set_title("Panel Average Over Time")
    ax.grid(True, alpha=0.3)

    return _save_or_return([fig], save_dir)


def snapshot_across_entities(df, entity_col, date_col, value_col, date_value, save_dir=None, max_entities=30):
    """Cross-sectional snapshot at specific date."""
    date_value = pd.to_datetime(date_value)
    subset = df[df[date_col] == date_value]

    if len(subset) == 0:
        logger.warning(f"No data found for date {date_value}")
        return []
    
    if len(subset) > max_entities:
        logger.warning(f"Snapshot has {len(subset)} entities. Showing top {max_entities} by value.")
        subset = subset.nlargest(max_entities, value_col)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    sns.barplot(data=subset, x=entity_col, y=value_col, ax=ax)
    ax.set_title(f"Snapshot at {date_value.date()}")
    ax.tick_params(axis="x", rotation=45)

    return _save_or_return([fig], save_dir)

