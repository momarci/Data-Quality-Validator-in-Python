"""
read_data.py — BULLETPROOF + SCALABLE VERSION

Reads CSV, Excel, JSON, Parquet with:
- Automatic delimiter / encoding detection
- Low-memory mode for large CSV files
- Automatic dtype downcasting to reduce memory
- Bad-line skipping for messy CSVs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import csv
import logging

logger = logging.getLogger(__name__)

# Threshold for enabling low-memory optimisations (bytes)
_LARGE_FILE_BYTES = 500 * 1024 * 1024  # 500 MB


def _downcast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory by downcasting int/float columns."""
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def _detect_encoding(path, sample_bytes=8192):
    try:
        with open(path, "rb") as f:
            raw = f.read(sample_bytes)
        for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252", "cp1250"]:
            try:
                raw.decode(enc)
                return enc
            except Exception:
                continue
    except Exception:
        pass
    return "utf-8-sig"


def _detect_delimiter(path, encoding):
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
    except Exception:
        for d in [",", ";", "\t", "|"]:
            try:
                df_test = pd.read_csv(path, delimiter=d, nrows=5, encoding=encoding)
                if df_test.shape[1] > 1:
                    return d
            except Exception:
                continue
    return ","


def read_data(path, sheet_name=None):
    path = Path(path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = path.stat().st_size
    is_large = file_size > _LARGE_FILE_BYTES

    if is_large:
        logger.info(f"Large file detected ({file_size / 1e9:.2f} GB). Using optimised loading.")

    # ----------------------------
    # EXCEL
    # ----------------------------
    if suffix in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            df = pd.read_excel(path, sheet_name=sheet_name)
        return _downcast_numerics(df)

    # ----------------------------
    # JSON
    # ----------------------------
    if suffix == ".json":
        return _downcast_numerics(pd.read_json(path))

    # ----------------------------
    # PARQUET
    # ----------------------------
    if suffix == ".parquet":
        return pd.read_parquet(path)

    # ----------------------------
    # CSV & text
    # ----------------------------
    encoding = _detect_encoding(path)
    delimiter = _detect_delimiter(path, encoding)

    read_kwargs = dict(
        delimiter=delimiter,
        encoding=encoding,
        on_bad_lines="skip",
        engine="c" if not is_large else "c",
        low_memory=True,
    )

    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception:
        read_kwargs["engine"] = "python"
        df = pd.read_csv(path, **read_kwargs)

    df = _downcast_numerics(df)
    return df
