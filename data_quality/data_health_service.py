from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import DATABASE_PATH
from data.database import get_database_status, get_engine, get_stock_daily, get_stock_list
from data_quality.audit import DataQualityAuditor
from data_quality.models import connect, date_iso


class DataHealthService:
    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path).close()

    def get_database_health_summary(self, price_df: pd.DataFrame | None = None, stock_list: pd.DataFrame | None = None) -> dict[str, Any]:
        prices = self._prices(price_df)
        stocks = self._stocks(stock_list)
        latest_run = self.get_latest_data_quality_run()
        return {
            "stock_count": int(len(stocks)),
            "daily_price_count": int(len(prices)),
            "start_date": date_iso(prices["date"].min()) if not prices.empty and "date" in prices else None,
            "end_date": date_iso(prices["date"].max()) if not prices.empty and "date" in prices else None,
            "latest_trading_day": date_iso(prices["date"].max()) if not prices.empty and "date" in prices else None,
            "latest_quality_run_id": latest_run.get("id") if latest_run else None,
            "critical_issue_count": latest_run.get("critical_issue_count", 0) if latest_run else 0,
            "warning_issue_count": latest_run.get("warning_issue_count", 0) if latest_run else 0,
            "adjust_type": self._dominant_adjust_type(prices),
        }

    def get_latest_data_quality_run(self) -> dict[str, Any] | None:
        runs = DataQualityAuditor(self.db_path).list_runs()
        if runs.empty:
            return None
        return runs.iloc[0].to_dict()

    def get_issue_summary_by_type(self) -> pd.DataFrame:
        issues = DataQualityAuditor(self.db_path).list_issues()
        if issues.empty:
            return pd.DataFrame(columns=["issue_type", "count"])
        return issues.groupby("issue_type").size().reset_index(name="count").sort_values("count", ascending=False)

    def get_stale_stocks(self, max_lag_days: int = 7, price_df: pd.DataFrame | None = None, latest_date: str | None = None) -> pd.DataFrame:
        prices = self._prices(price_df)
        if prices.empty or "date" not in prices:
            return pd.DataFrame(columns=["code", "latest_date", "lag_days"])
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        latest = pd.to_datetime(latest_date or prices["date"].max())
        coverage = prices.groupby("code", as_index=False)["date"].max()
        coverage["lag_days"] = (latest - coverage["date"]).dt.days
        coverage["latest_date"] = coverage["date"].dt.strftime("%Y-%m-%d")
        return coverage.loc[coverage["lag_days"].gt(int(max_lag_days)), ["code", "latest_date", "lag_days"]].reset_index(drop=True)

    def get_data_coverage_by_code(self, price_df: pd.DataFrame | None = None) -> pd.DataFrame:
        prices = self._prices(price_df)
        if prices.empty:
            return pd.DataFrame(columns=["code", "start_date", "end_date", "row_count"])
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        grouped = prices.groupby("code")["date"].agg(["min", "max", "count"]).reset_index()
        grouped.columns = ["code", "start_date", "end_date", "row_count"]
        grouped["start_date"] = grouped["start_date"].dt.strftime("%Y-%m-%d")
        grouped["end_date"] = grouped["end_date"].dt.strftime("%Y-%m-%d")
        return grouped

    def get_adjust_type_summary(self, price_df: pd.DataFrame | None = None) -> pd.DataFrame:
        prices = self._prices(price_df)
        if prices.empty or "adjust_type" not in prices:
            return pd.DataFrame({"adjust_type": ["unknown"], "count": [int(len(prices))]})
        return prices.groupby("adjust_type").size().reset_index(name="count")

    def get_universe_bias_summary(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            frame = pd.read_sql_query("SELECT warnings_json FROM stock_universe_snapshots ORDER BY created_at DESC LIMIT 1", conn)
        if frame.empty:
            return {"warnings": ["no universe snapshot available"], "bias_flags": []}
        warnings = frame.iloc[0]["warnings_json"]
        return {"warnings": warnings, "bias_flags": ["survivorship_bias_warning"] if "survivorship" in str(warnings).lower() else []}

    def build_data_snapshot_info(
        self,
        adjust_type: str = "qfq",
        universe_config: dict[str, Any] | None = None,
        universe_warnings: list[str] | None = None,
        price_df: pd.DataFrame | None = None,
        stock_list: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        summary = self.get_database_health_summary(price_df=price_df, stock_list=stock_list)
        latest_run = self.get_latest_data_quality_run()
        snapshot = {
            "stock_count": summary["stock_count"],
            "daily_price_count": summary["daily_price_count"],
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
            "adjust_type": adjust_type,
            "latest_quality_run_id": latest_run.get("id") if latest_run else None,
            "critical_issue_count": int(latest_run.get("critical_issue_count", 0)) if latest_run else 0,
            "warning_issue_count": int(latest_run.get("warning_issue_count", 0)) if latest_run else 0,
            "universe_config": universe_config or {},
            "universe_warnings": universe_warnings or ([] if latest_run else ["no data_quality_run available"]),
        }
        return snapshot

    def _prices(self, price_df: pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(price_df, pd.DataFrame):
            return price_df.copy()
        try:
            return get_stock_daily(engine=get_engine(self.db_path))
        except Exception:
            return pd.DataFrame()

    def _stocks(self, stock_list: pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(stock_list, pd.DataFrame):
            return stock_list.copy()
        try:
            return get_stock_list(engine=get_engine(self.db_path))
        except Exception:
            try:
                status = get_database_status(engine=get_engine(self.db_path))
                return pd.DataFrame(index=range(int(status.get("stock_count", 0))))
            except Exception:
                return pd.DataFrame()

    def _dominant_adjust_type(self, prices: pd.DataFrame) -> str:
        if prices.empty or "adjust_type" not in prices:
            return "unknown"
        counts = prices["adjust_type"].fillna("unknown").astype(str).value_counts()
        return str(counts.index[0]) if not counts.empty else "unknown"
