from __future__ import annotations

import re
from typing import Any

import pandas as pd


RATING_OPTIONS = ["unrated", "A", "B", "C", "D"]
RATING_LABELS = {
    "A": "A: 值得继续研究",
    "B": "B: 可观察",
    "C": "C: 普通",
    "D": "D: 淘汰",
    "unrated": "unrated: 未评级",
}

EXPERIMENT_TYPE_OPTIONS = [
    "all",
    "single_strategy_backtest",
    "portfolio_backtest",
    "parameter_optimization",
    "strategy_leaderboard",
    "market_regime_analysis",
]

SORTABLE_EXPERIMENT_COLUMNS = [
    "created_at",
    "net_CAGR",
    "max_drawdown",
    "sharpe_ratio",
    "final_equity",
    "cost_impact",
    "win_rate",
]

LOWER_IS_BETTER_METRICS = {"cost_impact", "total_cost", "cash_drag"}
DATE_SORT_COLUMNS = {"created_at", "start_date", "end_date"}


def parse_tags(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [item.strip() for item in re.split(r"[,，\s]+", value) if item.strip()]
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
    keyword: str | None = None,
) -> pd.DataFrame:
    if experiments.empty:
        return experiments.copy()
    result = experiments.copy()
    if experiment_type and experiment_type != "all":
        result = result[result["experiment_type"].eq(experiment_type)]
    if strategy_name and strategy_name != "all":
        result = result[result["strategy_name"].fillna("").eq(strategy_name)]
    if rating and rating != "all":
        result = result[result["rating"].fillna("unrated").eq(rating)]
    if start_date is not None and "created_at" in result.columns:
        created_at = pd.to_datetime(result["created_at"], errors="coerce").dt.date
        result = result[created_at.ge(pd.to_datetime(start_date).date())]
    if end_date is not None and "created_at" in result.columns:
        created_at = pd.to_datetime(result["created_at"], errors="coerce").dt.date
        result = result[created_at.le(pd.to_datetime(end_date).date())]
    if tag:
        needle = tag.strip().lower()
        result = result[result["tags"].map(lambda tags: _tags_match(tags, needle))]
    if keyword:
        result = result[_keyword_mask(result, keyword)]
    return result.reset_index(drop=True)


def sort_metric_values(metric_name: str, values: list[Any] | pd.Series, ascending: bool = False) -> list[Any]:
    frame = pd.DataFrame({"value": list(values)})
    sorted_frame = _sort_frame_by_metric(frame, "value", metric_name, ascending)
    return [None if pd.isna(value) else value for value in sorted_frame["value"].tolist()]


def sort_experiment_table(experiments: pd.DataFrame, sort_by: str, ascending: bool = False) -> pd.DataFrame:
    if experiments.empty or sort_by not in experiments.columns:
        return experiments.copy()
    return _sort_frame_by_metric(experiments.copy(), sort_by, sort_by, ascending).reset_index(drop=True)


def delete_confirmation_matches(experiment_name: str | None, confirmation: str | None) -> bool:
    return bool(experiment_name) and str(confirmation or "").strip() == str(experiment_name)


def _sort_frame_by_metric(frame: pd.DataFrame, column: str, metric_name: str, ascending: bool) -> pd.DataFrame:
    if metric_name in DATE_SORT_COLUMNS:
        key = pd.to_datetime(frame[column], errors="coerce")
    else:
        key = pd.to_numeric(frame[column], errors="coerce")

    sortable = frame.assign(_sort_key=key)
    present = sortable[sortable["_sort_key"].notna()]
    missing = sortable[sortable["_sort_key"].isna()]

    if metric_name in LOWER_IS_BETTER_METRICS:
        numeric_ascending = not ascending
    else:
        numeric_ascending = ascending

    present = present.sort_values("_sort_key", ascending=numeric_ascending)
    return pd.concat([present, missing], ignore_index=True).drop(columns=["_sort_key"])


def _tags_match(tags: Any, needle: str) -> bool:
    if not needle:
        return True
    if not isinstance(tags, list):
        tags = parse_tags(tags)
    return any(needle in str(tag).lower() for tag in tags)


def _keyword_mask(frame: pd.DataFrame, keyword: str) -> pd.Series:
    needle = keyword.strip().lower()
    if not needle:
        return pd.Series(True, index=frame.index)
    columns = [column for column in ["name", "description", "notes", "conclusion"] if column in frame.columns]
    if not columns:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        mask = mask | frame[column].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
    return mask
