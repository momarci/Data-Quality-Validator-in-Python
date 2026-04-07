
"""
ask_data_type.py

Auto-detect dataset type using:
- Date columns (0, 1, >1)
- Uniqueness of dates
- Presence of categorical ID columns

Returns:
{
    "detected_type": ...,
    "reasoning": ...
}
"""

import pandas as pd


def detect_data_type(df: pd.DataFrame, date_columns):
    n_dates = len(date_columns)

    # Case 1 — No date column → Cross-sectional
    if n_dates == 0:
        return {
            "detected_type": "cross_sectional",
            "reasoning": "No date columns detected."
        }

    # Case 2 — More than one date → ambiguous, let GUI choose
    if n_dates > 1:
        return {
            "detected_type": "unknown",
            "reasoning": "Multiple date-like columns detected."
        }

    # Exactly one date column
    date_col = date_columns[0]
    uniques = df[date_col].nunique()
    rows = len(df)

    # Perfect one-to-one → time series
    if uniques == rows:
        return {
            "detected_type": "time_series",
            "reasoning": "Single date column with unique timestamps."
        }

    # Many rows per date & categorical column present → panel
    cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    if cat_cols:
        return {
            "detected_type": "panel",
            "reasoning": "Date column + entity-like categorical column found."
        }

    # Fallback
    return {
        "detected_type": "unknown",
        "reasoning": "Data ambiguous; user input needed."
    }
