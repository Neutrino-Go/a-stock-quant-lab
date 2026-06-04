from __future__ import annotations

from typing import Any

import pandas as pd

from data_quality.adjustment import attach_adjust_type, validate_adjust_type_consistency
from data_quality.models import DataQualityIssue, date_iso


REQUIRED_DAILY_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]


class DataValidator:
    def __init__(self, abnormal_return_threshold: float = 0.35, stale_days: int = 3) -> None:
        self.abnormal_return_threshold = abnormal_return_threshold
        self.stale_days = stale_days

    def validate_daily_prices(self, df: pd.DataFrame, code: str | None = None, adjust_type: str | None = None) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        try:
            if df.empty:
                return issues
            frame = df.copy()
            if adjust_type and "adjust_type" not in frame.columns:
                frame = attach_adjust_type(frame, adjust_type)
            missing = [column for column in REQUIRED_DAILY_COLUMNS if column not in frame.columns]
            for column in missing:
                issues.append(DataQualityIssue("ERROR", "missing_required_field", code=code, field=column, message=f"Missing required field: {column}"))
            if missing:
                return issues
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            issues.extend(self._date_issues(frame, code))
            issues.extend(self._duplicate_issues(frame))
            issues.extend(self._numeric_issues(frame))
            issues.extend(self._sorting_issues(frame))
            issues.extend(self._return_issues(frame))
            issues.extend(self._stale_issues(frame))
            issues.extend(validate_adjust_type_consistency(frame, raise_on_error=False))
        except Exception as exc:
            issues.append(DataQualityIssue("CRITICAL", "validator_exception", code=code, message=str(exc)))
        return issues

    def _date_issues(self, frame: pd.DataFrame, code: str | None) -> list[DataQualityIssue]:
        return [
            DataQualityIssue("ERROR", "missing_trading_day", code=code or _value(row, "code"), date=None, field="date", message="date is missing")
            for _, row in frame.loc[frame["date"].isna()].iterrows()
        ]

    def _duplicate_issues(self, frame: pd.DataFrame) -> list[DataQualityIssue]:
        subset = ["code", "date"] + (["adjust_type"] if "adjust_type" in frame.columns else [])
        duplicates = frame.loc[frame.duplicated(subset=subset, keep=False)]
        return [
            DataQualityIssue("ERROR", "duplicate_price_row", code=_value(row, "code"), date=date_iso(row.get("date")), message="Duplicate price row", details={"subset": subset})
            for _, row in duplicates.iterrows()
        ]

    def _numeric_issues(self, frame: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ["open", "high", "low", "close"]:
            for _, row in frame.loc[frame[column].le(0)].iterrows():
                issues.append(DataQualityIssue("ERROR", "negative_price", code=_value(row, "code"), date=date_iso(row.get("date")), field=column, message=f"{column} must be positive"))
        for column in ["volume", "amount", "turnover"]:
            issue_type = f"negative_{column}"
            for _, row in frame.loc[frame[column].lt(0)].iterrows():
                issues.append(DataQualityIssue("ERROR", issue_type, code=_value(row, "code"), date=date_iso(row.get("date")), field=column, message=f"{column} must be non-negative"))
        invalid = frame.loc[(frame["high"] < frame[["open", "close"]].max(axis=1)) | (frame["low"] > frame[["open", "close"]].min(axis=1))]
        for _, row in invalid.iterrows():
            issues.append(DataQualityIssue("ERROR", "invalid_ohlc", code=_value(row, "code"), date=date_iso(row.get("date")), message="OHLC relationship is invalid"))
        for _, row in frame.loc[frame["volume"].eq(0)].iterrows():
            issues.append(DataQualityIssue("WARNING", "zero_volume", code=_value(row, "code"), date=date_iso(row.get("date")), field="volume", message="volume is zero"))
        return issues

    def _sorting_issues(self, frame: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for code, group in frame.groupby("code", dropna=False):
            if not group["date"].is_monotonic_increasing:
                issues.append(DataQualityIssue("WARNING", "data_not_sorted", code=str(code), field="date", message="price data is not sorted by date"))
        return issues

    def _return_issues(self, frame: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for code, group in frame.sort_values(["code", "date"]).groupby("code", dropna=False):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change()
            abnormal = group.loc[returns.abs().gt(self.abnormal_return_threshold)]
            for _, row in abnormal.iterrows():
                issues.append(DataQualityIssue("WARNING", "abnormal_return", code=str(code), date=date_iso(row.get("date")), field="close", message="abnormal daily return"))
        return issues

    def _stale_issues(self, frame: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for code, group in frame.sort_values(["code", "date"]).groupby("code", dropna=False):
            stale_mask = group[["open", "high", "low", "close"]].diff().abs().sum(axis=1).eq(0) & pd.to_numeric(group["volume"], errors="coerce").gt(0)
            if stale_mask.rolling(self.stale_days).sum().ge(self.stale_days).any():
                row = group.loc[stale_mask].iloc[-1]
                issues.append(DataQualityIssue("WARNING", "stale_data", code=str(code), date=date_iso(row.get("date")), message="price unchanged for multiple rows with volume"))
        return issues


def _value(row: pd.Series, key: str) -> Any:
    return row.get(key) if key in row else None
