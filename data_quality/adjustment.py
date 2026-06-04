from __future__ import annotations

import pandas as pd

from data_quality.models import AdjustmentConfig, DataQualityIssue


ADJUST_ALIASES = {
    "": "unknown",
    "none": "raw",
    "no": "raw",
    "raw": "raw",
    "qfq": "qfq",
    "hfq": "hfq",
    "前复权": "qfq",
    "后复权": "hfq",
    "不复权": "raw",
    "unknown": "unknown",
}


def normalize_adjust_type(adjust_type: object) -> str:
    key = str(adjust_type or "").strip().lower()
    if key not in ADJUST_ALIASES:
        raise ValueError(f"Unknown adjust_type: {adjust_type}")
    return ADJUST_ALIASES[key]


def attach_adjust_type(df: pd.DataFrame, adjust_type: str = "qfq") -> pd.DataFrame:
    result = df.copy()
    result["adjust_type"] = normalize_adjust_type(adjust_type)
    return result


def validate_adjust_type_consistency(
    df: pd.DataFrame,
    config: AdjustmentConfig | None = None,
    raise_on_error: bool = True,
) -> list[DataQualityIssue]:
    config = config or AdjustmentConfig()
    if df.empty:
        return []
    if "adjust_type" not in df.columns:
        issue = DataQualityIssue(
            "WARNING",
            "adjust_type_missing_legacy",
            field="adjust_type",
            message="price data does not contain adjust_type; selected adjust_type is treated as assumption only",
            details={
                "requested_adjust_type": config.adjust_type,
                "actual_adjust_type": "unknown",
                "assumed_adjust_type": config.adjust_type,
            },
        )
        if raise_on_error and not config.allow_mixed_adjust_type:
            raise ValueError(issue.message)
        return [issue]
    values = sorted({str(value or "unknown").strip().lower() for value in df["adjust_type"].dropna().unique()})
    issues: list[DataQualityIssue] = []
    if not values:
        issues.append(DataQualityIssue("WARNING", "adjust_type_mismatch", field="adjust_type", message="adjust_type is empty"))
    normalized = []
    for value in values:
        try:
            normalized.append(normalize_adjust_type(value))
        except ValueError:
            issues.append(DataQualityIssue("WARNING", "adjust_type_mismatch", field="adjust_type", message=f"Unknown adjust_type: {value}"))
    if len(set(normalized)) > 1 and not config.allow_mixed_adjust_type:
        issue = DataQualityIssue("ERROR", "adjust_type_mismatch", field="adjust_type", message=f"Mixed adjust_type values: {values}")
        issues.append(issue)
        if raise_on_error:
            raise ValueError(issue.message)
    if "unknown" in normalized:
        issues.append(DataQualityIssue("WARNING", "adjust_type_mismatch", field="adjust_type", message="unknown adjust_type is present"))
    return issues


def get_price_table_name_or_filter(adjust_type: str) -> dict[str, str]:
    return {"table": "daily_price", "adjust_type": normalize_adjust_type(adjust_type)}
