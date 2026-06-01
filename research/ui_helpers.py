from __future__ import annotations

import re
from typing import Any

import pandas as pd


RATING_OPTIONS = ["unrated", "A", "B", "C", "D"]
RATING_LABELS = {
    "A": "A：值得继续研究",
    "B": "B：可观察",
    "C": "C：普通",
    "D": "D：淘汰",
    "unrated": "unrated：未评级",
}


def parse_tags(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        tag = str(item).strip()
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def rating_display_name(rating: str | None) -> str:
    return RATING_LABELS.get(str(rating), RATING_LABELS["unrated"])


def experiment_option_label(row: dict[str, Any] | pd.Series) -> str:
    data = dict(row)
    created_at = str(data.get("created_at", ""))[:10]
    experiment_id = str(data.get("id", ""))[:8]
    return f"{data.get('name', '')} | {data.get('experiment_type', '')} | {created_at} | {experiment_id}"


def filter_experiment_table(
    experiments: pd.DataFrame,
    experiment_type: str | None = None,
    strategy_name: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    tag: str | None = None,
    rating: str | None = None,
) -> pd.DataFrame:
    if experiments.empty:
        return experiments.copy()
    result = experiments.copy()
    if experiment_type:
        result = result[result["experiment_type"].eq(experiment_type)]
    if strategy_name:
        result = result[result["strategy_name"].fillna("").eq(strategy_name)]
    if rating:
        result = result[result["rating"].fillna("unrated").eq(rating)]
    if start_date is not None and "created_at" in result.columns:
        result = result[pd.to_datetime(result["created_at"]).ge(pd.to_datetime(start_date))]
    if end_date is not None and "created_at" in result.columns:
        result = result[pd.to_datetime(result["created_at"]).le(pd.to_datetime(end_date))]
    if tag:
        result = result[result["tags"].map(lambda tags: tag in (tags or []))]
    return result.reset_index(drop=True)
