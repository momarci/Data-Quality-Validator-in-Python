
"""
convert_str_to_category.py

Detects string/object dtype columns and converts them into pandas 'category'.
Supports optional encoding:
- none        → keep as category
- integer     → category codes
- onehot      → dummy encoding

Returns:
{
    "df": converted dataframe,
    "converted_columns": [...],
    "levels": { col: [...] }
}
"""

import pandas as pd


def detect_string_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"
    ]


def convert_to_category(df, columns=None, encode="none"):
    if columns is None:
        columns = detect_string_columns(df)

    df_out = df.copy()
    levels = {}

    for col in columns:
        df_out[col] = df_out[col].astype("category")
        levels[col] = list(df_out[col].cat.categories)

    if encode == "integer":
        for col in columns:
            df_out[col] = df_out[col].cat.codes

    elif encode == "onehot":
        df_out = pd.get_dummies(df_out, columns=columns, dtype=float)

    return {
        "df": df_out,
        "converted_columns": columns,
        "levels": levels
    }
