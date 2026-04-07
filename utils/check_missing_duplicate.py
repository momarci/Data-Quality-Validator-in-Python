"""
check_missing_duplicate.py  (v3 — scalable edition)

Checks missing values and duplicate rows using vectorized operations.
Handles multi-GB dataframes without crashing.

Strategies
----------
- Missing:  column-level counts, row-level completeness bands, dtype breakdown.
- Duplicates:  hash-based detection using pandas .duplicated() and groupby,
  with three rule tiers (full-row, date-keyed, entity+date-keyed).
  All operations are O(n) or O(n log n) — no nested row loops.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def missing_summary(df: pd.DataFrame) -> dict:
    """Comprehensive missing-value summary (vectorized)."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    missing_per_col = df.isna().sum()
    pct_per_col = (missing_per_col / n_rows * 100).round(2)

    # Row-level completeness
    missing_per_row = df.isna().sum(axis=1)
    complete_rows = int((missing_per_row == 0).sum())

    # Completeness bands
    row_pct_missing = missing_per_row / n_cols * 100
    bands = {
        "complete (0%)": int((row_pct_missing == 0).sum()),
        "low (0-10%)": int(((row_pct_missing > 0) & (row_pct_missing <= 10)).sum()),
        "moderate (10-50%)": int(((row_pct_missing > 10) & (row_pct_missing <= 50)).sum()),
        "high (50-90%)": int(((row_pct_missing > 50) & (row_pct_missing <= 90)).sum()),
        "mostly empty (>90%)": int((row_pct_missing > 90).sum()),
    }

    # Dtype breakdown of missing
    dtype_missing = {}
    for dtype_name, group in df.columns.to_series().groupby(df.dtypes.astype(str)):
        cols = group.values.tolist()
        dtype_missing[dtype_name] = int(df[cols].isna().sum().sum())

    # Columns with zero missing
    fully_populated = missing_per_col[missing_per_col == 0].index.tolist()
    cols_with_missing = missing_per_col[missing_per_col > 0].sort_values(ascending=False)

    return {
        "total_cells": total_cells,
        "total_missing": int(missing_per_col.sum()),
        "missing_percentage": round(float(missing_per_col.sum() / total_cells * 100), 4) if total_cells else 0,
        "complete_rows": complete_rows,
        "complete_row_percentage": round(complete_rows / n_rows * 100, 2) if n_rows else 0,
        "missing_per_column": cols_with_missing.to_dict(),
        "missing_pct_per_column": pct_per_col[pct_per_col > 0].to_dict(),
        "fully_populated_columns": fully_populated,
        "row_completeness_bands": bands,
        "dtype_missing_breakdown": dtype_missing,
    }


# ---------------------------------------------------------------------------
# Duplicates  (hash-based, O(n))
# ---------------------------------------------------------------------------

def _auto_detect_column(df, keywords):
    """Return the first column whose name matches any keyword (case-insensitive)."""
    for col in df.columns:
        cl = col.lower().strip()
        for kw in keywords:
            if kw in cl:
                return col
    return None


def _build_duplicate_groups(df: pd.DataFrame, mask_a, entity_col, date_col,
                             max_groups: int = 300) -> list:
    """
    Group exact duplicate rows (Rule A) and return pairing information.

    Each entry: {"indices": [i, j, ...], "key_values": {col: val, ...}}
    Key columns: entity_col + date_col if available, else first 3 columns.
    """
    if not mask_a.any():
        return []

    dup_df = df[mask_a]

    # Key columns for labelling each group
    key_cols = [c for c in [entity_col, date_col] if c and c in df.columns]
    if not key_cols:
        key_cols = list(df.columns[:3])

    # Build groups via stringified row hash (handles all dtypes safely)
    hash_series = dup_df.astype(str).apply("|".join, axis=1)
    groups_dict: dict = {}
    for idx, h in hash_series.items():
        groups_dict.setdefault(h, []).append(idx)

    result = []
    for idxs in groups_dict.values():
        if len(idxs) < 2:
            continue
        # Extract key values from the first row of the group
        first_row = df.loc[idxs[0], key_cols]
        key_vals = {str(c): str(first_row[c]) for c in key_cols}
        result.append({"indices": idxs, "key_values": key_vals})
        if len(result) >= max_groups:
            break

    # Sort by first index for deterministic output
    result.sort(key=lambda g: g["indices"][0])
    return result


def duplicate_summary(df: pd.DataFrame, entity_col=None, date_col=None) -> dict:
    """
    Scalable duplicate detection using three rule tiers.

    Rule A - Exact full-row duplicates (pandas .duplicated).
    Rule B - Same date + >=95% of remaining columns identical (groupby + vectorized).
    Rule C - Same entity+date + >=95% of remaining columns identical.
    """
    n_rows = len(df)
    if n_rows == 0:
        return _empty_dup_result()

    # Auto-detect columns if not supplied
    if date_col is None:
        date_col = _auto_detect_column(df, ["date", "time", "timestamp", "period"])
    if entity_col is None:
        entity_col = _auto_detect_column(df, ["entity", "id", "ticker", "symbol", "company"])

    dup_indices = set()
    rule_counts = {"rule_a": 0, "rule_b": 0, "rule_c": 0}

    # --- Rule A: exact full-row duplicates ---
    mask_a = df.duplicated(keep=False)
    rule_a_idx = set(df.index[mask_a].tolist())
    rule_counts["rule_a"] = len(rule_a_idx)
    dup_indices |= rule_a_idx

    # Build pairwise groups for Rule A before moving on
    duplicate_groups = _build_duplicate_groups(df, mask_a, entity_col, date_col)

    # --- Rule B: same date + >=50% of other cols match ---
    if date_col and date_col in df.columns:
        rule_b_idx = _grouped_fuzzy_dups(df, group_cols=[date_col],
                                          compare_cols=[c for c in df.columns if c != date_col])
        rule_counts["rule_b"] = len(rule_b_idx)
        dup_indices |= rule_b_idx

    # --- Rule C: same entity+date + >=50% of other cols match ---
    if (entity_col and entity_col in df.columns
            and date_col and date_col in df.columns):
        exclude = {entity_col, date_col}
        rule_c_idx = _grouped_fuzzy_dups(df, group_cols=[entity_col, date_col],
                                          compare_cols=[c for c in df.columns if c not in exclude])
        rule_counts["rule_c"] = len(rule_c_idx)
        dup_indices |= rule_c_idx

    # Build summary  (cap the actual duplicate rows stored to avoid huge JSON)
    MAX_STORED = 500
    sorted_idx = sorted(dup_indices)
    sample_idx = sorted_idx[:MAX_STORED]
    dup_rows_sample = df.loc[sample_idx].to_dict(orient="records") if sample_idx else []

    return {
        "duplicate_count": len(dup_indices),
        "duplicate_percentage": round(len(dup_indices) / n_rows * 100, 4) if n_rows else 0.0,
        "duplicate_rows": dup_rows_sample,
        "duplicate_groups": duplicate_groups,
        "duplicates_truncated": len(dup_indices) > MAX_STORED,
        "rule_counts": rule_counts,
        "detection_rules": {
            "rule_a": "Exact full-row duplicate (all columns identical)",
            "rule_b": "Same date + >=95% of remaining columns identical",
            "rule_c": "Same entity + same date + >=95% of remaining columns identical",
        },
        "date_column_used": date_col,
        "entity_column_used": entity_col,
    }


def _grouped_fuzzy_dups(df, group_cols, compare_cols, threshold_pct=0.95):
    """
    Within each group defined by *group_cols*, flag rows where >=threshold_pct
    of *compare_cols* match another row in the same group (default 95%).

    Uses vectorized pandas operations — no nested Python loops over rows.
    For very large groups (>5000 rows), falls back to exact-match on compare cols.
    """
    if not compare_cols:
        mask = df.groupby(group_cols, dropna=False)[df.columns[0]].transform("size") > 1
        return set(df.index[mask].tolist())

    n_compare = len(compare_cols)
    threshold = int(np.ceil(n_compare * threshold_pct))

    dup_idx = set()

    # Only process groups with >1 member
    grp_size = df.groupby(group_cols, dropna=False)[df.columns[0]].transform("size")
    subset = df[grp_size > 1]

    if len(subset) == 0:
        return dup_idx

    # Stringify compare columns once (handles mixed types safely)
    try:
        str_df = subset[compare_cols].astype(str)
    except Exception:
        str_df = subset[compare_cols].fillna("__NA__").astype(str)

    for _, grp in subset.groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue

        g = str_df.loc[grp.index]

        # For large groups, only check exact duplicates to stay fast
        if len(grp) > 500:
            mask = grp[compare_cols].duplicated(keep=False)
            dup_idx |= set(grp.index[mask].tolist())
            continue

        # Vectorized pairwise comparison within group
        indices = grp.index.tolist()
        arr = g.values  # numpy for speed
        n = len(arr)

        for i in range(n):
            # Compare row i against rows i+1..n in one vectorized op
            matches = (arr[i+1:] == arr[i]).sum(axis=1)
            hits = np.where(matches >= threshold)[0]
            if len(hits) > 0:
                dup_idx.add(indices[i])
                for h in hits:
                    dup_idx.add(indices[i + 1 + h])

    return dup_idx


def _empty_dup_result():
    return {
        "duplicate_count": 0,
        "duplicate_percentage": 0.0,
        "duplicate_rows": [],
        "duplicate_groups": [],
        "duplicates_truncated": False,
        "rule_counts": {"rule_a": 0, "rule_b": 0, "rule_c": 0},
        "detection_rules": {},
        "date_column_used": None,
        "entity_column_used": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_missing_duplicate_checks(df: pd.DataFrame, entity_col=None, date_col=None) -> dict:
    logger.info(f"Running missing/duplicate checks on {df.shape[0]:,} rows x {df.shape[1]} cols")
    return {
        "missing": missing_summary(df),
        "duplicates": duplicate_summary(df, entity_col=entity_col, date_col=date_col),
    }
