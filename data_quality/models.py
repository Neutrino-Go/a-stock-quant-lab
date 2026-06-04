from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATABASE_PATH


SEVERITIES = {"INFO", "WARNING", "ERROR", "CRITICAL"}
ADJUST_TYPES = {"raw", "qfq", "hfq", "unknown"}


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str
    issue_type: str
    code: str | None = None
    name: str | None = None
    date: str | None = None
    field: str | None = None
    message: str = ""
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_record(self, run_id: str | None = None) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "severity": self.severity,
            "issue_type": self.issue_type,
            "code": self.code,
            "name": self.name,
            "date": _date_or_none(self.date),
            "field": self.field,
            "message": self.message,
            "details_json": json_dumps(self.details),
            "created_at": now_iso(),
        }


@dataclass(frozen=True)
class DataQualityRunResult:
    run_id: str
    status: str
    summary: dict[str, Any]
    issues: list[DataQualityIssue] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class StockUniverseResult:
    as_of_date: str
    universe_name: str
    codes: list[str]
    stock_count: int
    config: dict[str, Any]
    warnings: list[str] = dataclass_field(default_factory=list)
    bias_flags: list[str] = dataclass_field(default_factory=list)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdjustmentConfig:
    adjust_type: str = "qfq"
    price_fields: tuple[str, ...] = ("open", "high", "low", "close")
    allow_mixed_adjust_type: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DATA_QUALITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trading_calendar (
    date TEXT NOT NULL,
    is_trading_day INTEGER NOT NULL,
    market TEXT NOT NULL DEFAULT 'CN',
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (date, market)
);

CREATE TABLE IF NOT EXISTS data_quality_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    scope TEXT NOT NULL,
    adjust_type TEXT,
    universe_config_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    issue_count INTEGER NOT NULL DEFAULT 0,
    critical_issue_count INTEGER NOT NULL DEFAULT 0,
    warning_issue_count INTEGER NOT NULL DEFAULT 0,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    severity TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    code TEXT,
    name TEXT,
    date TEXT,
    field TEXT,
    message TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_universe_snapshots (
    id TEXT PRIMARY KEY,
    snapshot_date TEXT NOT NULL,
    universe_name TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    config_json TEXT NOT NULL,
    stock_count INTEGER NOT NULL,
    codes_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_status_history (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    name TEXT,
    date TEXT,
    is_st INTEGER,
    is_suspended INTEGER,
    is_delisted INTEGER,
    list_date TEXT,
    delist_date TEXT,
    source TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_fetch_logs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT,
    target TEXT,
    code TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT,
    row_count INTEGER,
    error_message TEXT,
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS adjustment_metadata (
    id TEXT PRIMARY KEY,
    code TEXT,
    adjust_type TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    row_count INTEGER,
    created_at TEXT NOT NULL,
    source TEXT,
    checksum TEXT
);

CREATE INDEX IF NOT EXISTS idx_data_quality_issues_run ON data_quality_issues(run_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_type ON data_quality_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_runs_created ON data_quality_runs(created_at);
"""


def ensure_data_quality_schema(db_path: str | Path = DATABASE_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(DATA_QUALITY_SCHEMA_SQL)


def connect(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_data_quality_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def json_dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)


def json_loads(value: Any, default: Any | None = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, dict | list):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_code(value: Any) -> str:
    return str(value).strip().zfill(6)


def date_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _date_or_none(value: Any) -> str | None:
    return date_iso(value)
