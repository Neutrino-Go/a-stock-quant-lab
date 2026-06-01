from __future__ import annotations

from typing import Any

import pandas as pd


def create_train_test_split(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_ratio: float = 0.7,
) -> dict[str, str]:
    start = _to_timestamp(start_date, "start_date")
    end = _to_timestamp(end_date, "end_date")
    _validate_date_range(start, end)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    total_days = max((end - start).days, 1)
    train_days = max(int(total_days * train_ratio), 1)
    train_end = start + pd.Timedelta(days=train_days)
    test_start = train_end + pd.Timedelta(days=1)
    if test_start > end:
        raise ValueError("train_ratio leaves no out-of-sample test window")

    return {
        "train_start": _date_str(start),
        "train_end": _date_str(train_end),
        "test_start": _date_str(test_start),
        "test_end": _date_str(end),
    }


def create_walk_forward_splits(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict[str, str | int]]:
    start = _to_timestamp(start_date, "start_date")
    end = _to_timestamp(end_date, "end_date")
    _validate_date_range(start, end)
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months, test_months, and step_months must be positive months")

    splits: list[dict[str, str | int]] = []
    train_start = start
    fold = 1
    while True:
        test_start = train_start + pd.DateOffset(months=train_months)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end:
            break
        if test_start > train_end and test_start <= test_end:
            splits.append(
                {
                    "fold": fold,
                    "train_start": _date_str(train_start),
                    "train_end": _date_str(train_end),
                    "test_start": _date_str(test_start),
                    "test_end": _date_str(test_end),
                }
            )
            fold += 1
        train_start = train_start + pd.DateOffset(months=step_months)
        if train_start >= end:
            break
    return splits


def _to_timestamp(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value)
    except Exception as exc:
        raise ValueError(f"{label} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid date")
    return pd.Timestamp(timestamp).normalize()


def _validate_date_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")


def _date_str(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")
