
"""
detect_date_column.py

Detects which columns in a DataFrame are date-like by:
- Attempting datetime conversion
- Scoring based on column name keywords
- Ignoring columns with known non-date patterns (e.g., %)

This module returns:
{
    "detected": [...],
    "details": { column: {score...}}
}
"""

import pandas as pd


DATE_KEYWORDS = ["date", "time", "year", "month", "day", "period", "timestamp"]
NON_DATE_PATTERNS = ["%", "ratio", "change", "diff"]


def name_score(colname: str) -> int:
    name = colname.lower()
    return sum(k in name for k in DATE_KEYWORDS)


def looks_non_date(series: pd.Series) -> bool:
    """Exclude columns containing non-date-style chars (% etc.)"""
    if series.dtype == "object":
        sample = series.astype(str).head(50)
        if sample.str.contains("%").any():
            return True
    return False


def detect_date_columns(df: pd.DataFrame, threshold=0.8):
    detected = []
    details = {}

    for col in df.columns:
        series = df[col]

        if looks_non_date(series):
            details[col] = {
                "parse_success_ratio": 0.0,
                "name_score": name_score(col),
                "is_date": False,
                "reason": "Non-date pattern detected"
            }
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        ratio = parsed.notna().mean()

        score = name_score(col)

        is_date = (ratio >= threshold) or (score > 0 and ratio > 0.3)

        if is_date:
            detected.append(col)

        details[col] = {
            "parse_success_ratio": float(ratio),
            "name_score": score,
            "is_date": is_date
        }

    return {
        "detected": detected,
        "details": details
    }
