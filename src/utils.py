\
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def normalise_filename(value: object) -> str:
    name = str(value).strip()
    return name if name.lower().endswith(".csv") else f"{name}.csv"


def parse_start_time(value: object):
    """Parse NASA MATLAB date-vector strings and ordinary timestamps."""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value

    text = str(value).strip()

    # Try ordinary pandas parsing first.
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed

    # NASA metadata may contain values such as:
    # [2008. 4. 2. 13. 8. 17.]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 6:
        parts = [int(float(x)) for x in nums[:6]]
        try:
            return pd.Timestamp(
                year=parts[0], month=parts[1], day=parts[2],
                hour=parts[3], minute=parts[4], second=parts[5]
            )
        except Exception:
            return pd.NaT

    return pd.NaT


def numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first matching numeric column, case-insensitively."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        actual = lower_map.get(name.lower())
        if actual is not None:
            return pd.to_numeric(df[actual], errors="coerce")
    return None


def safe_stat(series: pd.Series | None, op: str) -> float:
    if series is None:
        return np.nan
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    if op == "mean":
        return float(clean.mean())
    if op == "min":
        return float(clean.min())
    if op == "max":
        return float(clean.max())
    if op == "std":
        return float(clean.std(ddof=0))
    if op == "first":
        return float(clean.iloc[0])
    if op == "last":
        return float(clean.iloc[-1])
    raise ValueError(f"Unsupported operation: {op}")
