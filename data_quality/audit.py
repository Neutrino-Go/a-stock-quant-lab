from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATABASE_PATH
from data.database import get_engine, get_stock_daily
from data_quality.adjustment import attach_adjust_type, validate_adjust_type_consistency
from data_quality.calendar import TradingCalendarService
from data_quality.models import DataQualityIssue, DataQualityRunResult, connect, date_iso, json_dumps, json_loads, now_iso
from data_quality.universe import StockUniverseBuilder
from data_quality.validators import DataValidator


class DataQualityAuditor:
    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path).close()

    def run_audit(
        self,
        scope: str,
        start_date: str,
        end_date: str,
        codes: list[str] | None = None,
        adjust_type: str = "qfq",
        universe_config: dict[str, Any] | None = None,
        price_df: pd.DataFrame | None = None,
        check_calendar_gaps: bool = True,
        check_survivorship_bias: bool = True,
        check_adjustment: bool = True,
        max_codes: int = 50,
    ) -> DataQualityRunResult:
        run_id = self._create_run(scope, start_date, end_date, adjust_type, universe_config or {})
        try:
            prices = price_df.copy() if isinstance(price_df, pd.DataFrame) else self._load_price_data(codes, start_date, end_date)
            if not prices.empty and "adjust_type" not in prices.columns:
                prices = attach_adjust_type(prices, adjust_type)
            if codes and not prices.empty:
                normalized_codes = {str(code).zfill(6) for code in codes}
                prices = prices.loc[prices["code"].astype(str).str.zfill(6).isin(normalized_codes)]
            if max_codes and not prices.empty:
                allowed = prices["code"].astype(str).str.zfill(6).drop_duplicates().head(int(max_codes)).tolist()
                prices = prices.loc[prices["code"].astype(str).str.zfill(6).isin(allowed)]
            issues = DataValidator().validate_daily_prices(prices, adjust_type=adjust_type)
            if check_calendar_gaps and not prices.empty:
                issues.extend(self._calendar_gap_issues(prices, start_date, end_date))
            if check_adjustment and not prices.empty:
                issues.extend(validate_adjust_type_consistency(prices, raise_on_error=False))
            if check_survivorship_bias:
                result = StockUniverseBuilder(self.db_path).build_universe(end_date, **(universe_config or {}))
                issues.extend(
                    DataQualityIssue("WARNING", "survivorship_bias_warning", message=warning, details={"bias_flags": result.bias_flags})
                    for warning in result.warnings
                    if "bias" in warning.lower()
                )
            summary = self._summary(prices, issues, adjust_type, universe_config or {})
            self._replace_issues(run_id, issues)
            self._finish_run(run_id, "completed", summary, len(issues))
            return DataQualityRunResult(run_id, "completed", summary, issues)
        except Exception as exc:
            summary = {"error": str(exc), "total_issues": 1, "critical_issues": 1, "warning_issues": 0, "error_issues": 0, "info_issues": 0}
            issue = DataQualityIssue("CRITICAL", "audit_failed", message=str(exc))
            self._replace_issues(run_id, [issue])
            self._finish_run(run_id, "failed", summary, 1)
            return DataQualityRunResult(run_id, "failed", summary, [issue])

    def list_runs(self) -> pd.DataFrame:
        with connect(self.db_path) as conn:
            frame = pd.read_sql_query("SELECT * FROM data_quality_runs ORDER BY created_at DESC", conn)
        if not frame.empty:
            frame["summary_json"] = frame["summary_json"].map(lambda value: json_loads(value, {}))
            frame["universe_config_json"] = frame["universe_config_json"].map(lambda value: json_loads(value, {}))
        return frame

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM data_quality_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["summary_json"] = json_loads(result.get("summary_json"), {})
        result["universe_config_json"] = json_loads(result.get("universe_config_json"), {})
        return result

    def list_issues(
        self,
        run_id: str | None = None,
        severity: str | None = None,
        issue_type: str | None = None,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        keyword: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if run_id:
            conditions.append("run_id = :run_id")
            params["run_id"] = run_id
        if severity and severity != "all":
            conditions.append("severity = :severity")
            params["severity"] = severity
        if issue_type and issue_type != "all":
            conditions.append("issue_type = :issue_type")
            params["issue_type"] = issue_type
        if code:
            conditions.append("code = :code")
            params["code"] = str(code).zfill(6)
        if start_date:
            conditions.append("date >= :start_date")
            params["start_date"] = date_iso(start_date)
        if end_date:
            conditions.append("date <= :end_date")
            params["end_date"] = date_iso(end_date)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with connect(self.db_path) as conn:
            frame = pd.read_sql_query(f"SELECT * FROM data_quality_issues {where} ORDER BY created_at DESC", conn, params=params)
        if not frame.empty:
            frame["details_json"] = frame["details_json"].map(lambda value: json_loads(value, {}))
            if keyword:
                keyword_lower = keyword.lower()
                mask = frame.apply(lambda row: keyword_lower in " ".join(str(item).lower() for item in row.values), axis=1)
                frame = frame.loc[mask]
        return frame.reset_index(drop=True)

    def _load_price_data(self, codes: list[str] | None, start_date: str, end_date: str) -> pd.DataFrame:
        engine = get_engine(self.db_path)
        if codes:
            frames = [get_stock_daily(code=code, start_date=start_date, end_date=end_date, engine=engine) for code in codes]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return get_stock_daily(start_date=start_date, end_date=end_date, engine=engine)

    def _calendar_gap_issues(self, prices: pd.DataFrame, start_date: str, end_date: str) -> list[DataQualityIssue]:
        service = TradingCalendarService(self.db_path)
        issues: list[DataQualityIssue] = []
        for code in prices["code"].astype(str).str.zfill(6).drop_duplicates():
            for missing in service.find_missing_trading_days(prices, code, start_date, end_date):
                issues.append(DataQualityIssue("WARNING", "missing_trading_day", code=code, date=missing, field="date", message="missing trading day in price data"))
        return issues

    def _create_run(self, scope: str, start_date: str, end_date: str, adjust_type: str, universe_config: dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_quality_runs(
                    id, created_at, status, start_date, end_date, scope, adjust_type,
                    universe_config_json, summary_json, issue_count, critical_issue_count,
                    warning_issue_count, git_commit
                )
                VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, 0, 0, 0, ?)
                """,
                (run_id, now_iso(), date_iso(start_date), date_iso(end_date), scope, adjust_type, json_dumps(universe_config), json_dumps({}), _git_commit()),
            )
        return run_id

    def _replace_issues(self, run_id: str, issues: list[DataQualityIssue]) -> None:
        records = [issue.to_record(run_id) for issue in issues]
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM data_quality_issues WHERE run_id = ?", (run_id,))
            if records:
                conn.executemany(
                    """
                    INSERT INTO data_quality_issues(
                        id, run_id, severity, issue_type, code, name, date, field,
                        message, details_json, created_at
                    )
                    VALUES (
                        :id, :run_id, :severity, :issue_type, :code, :name, :date,
                        :field, :message, :details_json, :created_at
                    )
                    """,
                    records,
                )

    def _finish_run(self, run_id: str, status: str, summary: dict[str, Any], issue_count: int) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE data_quality_runs
                SET status = ?, summary_json = ?, issue_count = ?,
                    critical_issue_count = ?, warning_issue_count = ?
                WHERE id = ?
                """,
                (
                    status,
                    json_dumps(summary),
                    int(issue_count),
                    int(summary.get("critical_issues", 0)),
                    int(summary.get("warning_issues", 0)),
                    run_id,
                ),
            )

    def _summary(self, prices: pd.DataFrame, issues: list[DataQualityIssue], adjust_type: str, universe_config: dict[str, Any]) -> dict[str, Any]:
        counts = {severity: sum(issue.severity == severity for issue in issues) for severity in ["CRITICAL", "ERROR", "WARNING", "INFO"]}
        issue_types = pd.Series([issue.issue_type for issue in issues]).value_counts().to_dict() if issues else {}
        return {
            "total_issues": len(issues),
            "critical_issues": counts["CRITICAL"],
            "error_issues": counts["ERROR"],
            "warning_issues": counts["WARNING"],
            "info_issues": counts["INFO"],
            "codes_checked": int(prices["code"].nunique()) if not prices.empty and "code" in prices else 0,
            "records_checked": int(len(prices)),
            "missing_day_count": int(issue_types.get("missing_trading_day", 0)),
            "duplicate_count": int(issue_types.get("duplicate_price_row", 0)),
            "invalid_ohlc_count": int(issue_types.get("invalid_ohlc", 0)),
            "stale_code_count": int(issue_types.get("stale_data", 0)),
            "adjust_type": adjust_type,
            "universe_bias_flags": universe_config.get("bias_flags", []),
        }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        return ""
