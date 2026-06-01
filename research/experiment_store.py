from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATABASE_PATH, ensure_directories
from research.experiment_models import EXPERIMENT_RATINGS, EXPERIMENT_STATUSES, EXPERIMENT_TYPES


class ExperimentStoreError(RuntimeError):
    pass


class ExperimentStore:
    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path == DATABASE_PATH:
            ensure_directories()
        self.init_db()

    def init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_runs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        experiment_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        start_date TEXT,
                        end_date TEXT,
                        strategy_name TEXT,
                        description TEXT,
                        notes TEXT,
                        conclusion TEXT,
                        tags TEXT NOT NULL,
                        rating TEXT NOT NULL,
                        git_commit TEXT,
                        data_snapshot_info TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        metrics_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS experiment_equity_curves (
                        id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        date TEXT,
                        gross_equity REAL,
                        net_equity REAL,
                        drawdown REAL,
                        cash REAL,
                        market_value REAL,
                        position_count INTEGER
                    );

                    CREATE TABLE IF NOT EXISTS experiment_trades (
                        id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        code TEXT,
                        name TEXT,
                        strategy_name TEXT,
                        signal_date TEXT,
                        buy_date TEXT,
                        buy_price REAL,
                        sell_date TEXT,
                        sell_price REAL,
                        shares INTEGER,
                        gross_return REAL,
                        net_return REAL,
                        holding_days INTEGER,
                        exit_reason TEXT,
                        commission REAL,
                        stamp_tax REAL,
                        slippage_cost REAL,
                        total_cost REAL
                    );

                    CREATE TABLE IF NOT EXISTS experiment_orders (
                        id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        code TEXT,
                        name TEXT,
                        side TEXT,
                        signal_date TEXT,
                        order_date TEXT,
                        requested_price REAL,
                        fill_price REAL,
                        shares INTEGER,
                        status TEXT,
                        reason TEXT,
                        gross_amount REAL,
                        cost TEXT,
                        net_amount REAL
                    );

                    CREATE TABLE IF NOT EXISTS experiment_artifacts (
                        id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        artifact_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_experiment_runs_created_at
                    ON experiment_runs(created_at);
                    CREATE INDEX IF NOT EXISTS idx_experiment_equity_experiment_id
                    ON experiment_equity_curves(experiment_id);
                    CREATE INDEX IF NOT EXISTS idx_experiment_trades_experiment_id
                    ON experiment_trades(experiment_id);
                    CREATE INDEX IF NOT EXISTS idx_experiment_orders_experiment_id
                    ON experiment_orders(experiment_id);
                    CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_experiment_id
                    ON experiment_artifacts(experiment_id);
                    """
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to initialize experiment tables: {exc}") from exc

    def create_experiment_run(
        self,
        name: str,
        experiment_type: str,
        status: str = "completed",
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        strategy_name: str | None = None,
        description: str = "",
        notes: str = "",
        conclusion: str = "",
        tags: list[str] | None = None,
        rating: str = "unrated",
        git_commit: str = "",
        data_snapshot_info: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        if experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        if status not in EXPERIMENT_STATUSES:
            raise ValueError(f"Unknown status: {status}")
        if rating not in EXPERIMENT_RATINGS:
            raise ValueError(f"Unknown rating: {rating}")

        now = _now_iso()
        experiment_id = str(uuid.uuid4())
        record = {
            "id": experiment_id,
            "name": name,
            "experiment_type": experiment_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "start_date": _date_or_none(start_date),
            "end_date": _date_or_none(end_date),
            "strategy_name": strategy_name,
            "description": description,
            "notes": notes,
            "conclusion": conclusion,
            "tags": _json_dumps(tags or []),
            "rating": rating,
            "git_commit": git_commit,
            "data_snapshot_info": _json_dumps(data_snapshot_info or {}),
            "config_json": _json_dumps(config or {}),
            "metrics_json": _json_dumps(metrics or {}),
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO experiment_runs (
                        id, name, experiment_type, status, created_at, updated_at,
                        start_date, end_date, strategy_name, description, notes,
                        conclusion, tags, rating, git_commit, data_snapshot_info,
                        config_json, metrics_json
                    )
                    VALUES (
                        :id, :name, :experiment_type, :status, :created_at, :updated_at,
                        :start_date, :end_date, :strategy_name, :description, :notes,
                        :conclusion, :tags, :rating, :git_commit, :data_snapshot_info,
                        :config_json, :metrics_json
                    )
                    """,
                    record,
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to create experiment run: {exc}") from exc
        return experiment_id

    def save_metrics(self, experiment_id: str, metrics: dict[str, Any]) -> None:
        self._ensure_experiment(experiment_id)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE experiment_runs
                    SET metrics_json = :metrics_json, updated_at = :updated_at
                    WHERE id = :experiment_id
                    """,
                    {
                        "metrics_json": _json_dumps(metrics),
                        "updated_at": _now_iso(),
                        "experiment_id": experiment_id,
                    },
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to save metrics: {exc}") from exc

    def save_equity_curve(self, experiment_id: str, equity_curve: pd.DataFrame) -> None:
        self._ensure_experiment(experiment_id)
        rows = [_equity_record(experiment_id, row) for row in equity_curve.to_dict("records")] if not equity_curve.empty else []
        self._replace_rows("experiment_equity_curves", experiment_id, rows)

    def save_trades(self, experiment_id: str, trades: pd.DataFrame) -> None:
        self._ensure_experiment(experiment_id)
        rows = [_trade_record(experiment_id, row) for row in trades.to_dict("records")] if not trades.empty else []
        self._replace_rows("experiment_trades", experiment_id, rows)

    def save_orders(self, experiment_id: str, orders: pd.DataFrame) -> None:
        self._ensure_experiment(experiment_id)
        rows = [_order_record(experiment_id, row) for row in orders.to_dict("records")] if not orders.empty else []
        self._replace_rows("experiment_orders", experiment_id, rows)

    def update_notes(self, experiment_id: str, notes: str | None = None, conclusion: str | None = None) -> None:
        self._ensure_experiment(experiment_id)
        assignments = ["updated_at = :updated_at"]
        params: dict[str, Any] = {"experiment_id": experiment_id, "updated_at": _now_iso()}
        if notes is not None:
            assignments.append("notes = :notes")
            params["notes"] = notes
        if conclusion is not None:
            assignments.append("conclusion = :conclusion")
            params["conclusion"] = conclusion
        self._execute_update(f"UPDATE experiment_runs SET {', '.join(assignments)} WHERE id = :experiment_id", params)

    def update_tags(self, experiment_id: str, tags: list[str]) -> None:
        self._ensure_experiment(experiment_id)
        self._execute_update(
            "UPDATE experiment_runs SET tags = :tags, updated_at = :updated_at WHERE id = :experiment_id",
            {"tags": _json_dumps(tags), "updated_at": _now_iso(), "experiment_id": experiment_id},
        )

    def update_rating(self, experiment_id: str, rating: str) -> None:
        if rating not in EXPERIMENT_RATINGS:
            raise ValueError(f"Unknown rating: {rating}")
        self._ensure_experiment(experiment_id)
        self._execute_update(
            "UPDATE experiment_runs SET rating = :rating, updated_at = :updated_at WHERE id = :experiment_id",
            {"rating": rating, "updated_at": _now_iso(), "experiment_id": experiment_id},
        )

    def list_experiments(
        self,
        experiment_type: str | None = None,
        strategy_name: str | None = None,
        rating: str | None = None,
        created_start: str | pd.Timestamp | None = None,
        created_end: str | pd.Timestamp | None = None,
        tag: str | None = None,
        keyword: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if experiment_type and experiment_type != "all":
            conditions.append("experiment_type = :experiment_type")
            params["experiment_type"] = experiment_type
        if strategy_name and strategy_name != "all":
            conditions.append("strategy_name = :strategy_name")
            params["strategy_name"] = strategy_name
        if rating and rating != "all":
            conditions.append("rating = :rating")
            params["rating"] = rating
        if created_start is not None:
            conditions.append("date(created_at) >= date(:created_start)")
            params["created_start"] = _date_or_none(created_start)
        if created_end is not None:
            conditions.append("date(created_at) <= date(:created_end)")
            params["created_end"] = _date_or_none(created_end)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_sql = "LIMIT :limit" if limit else ""
        if limit:
            params["limit"] = int(limit)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM experiment_runs {where} ORDER BY created_at DESC {limit_sql}",
                    params,
                ).fetchall()
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to list experiments: {exc}") from exc
        frame = _runs_frame([_decode_run(dict(row)) for row in rows])
        if frame.empty:
            return frame
        if tag:
            needle = tag.strip().lower()
            frame = frame[frame["tags"].map(lambda tags: _tags_match(tags, needle))]
        if keyword:
            frame = frame[_keyword_mask(frame, keyword)]
        return frame.reset_index(drop=True)

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                run_row = conn.execute("SELECT * FROM experiment_runs WHERE id = ?", (experiment_id,)).fetchone()
                if run_row is None:
                    return {
                        "run": None,
                        "equity_curve": pd.DataFrame(),
                        "trades": pd.DataFrame(),
                        "orders": pd.DataFrame(),
                        "artifacts": pd.DataFrame(),
                    }
                equity = pd.read_sql_query(
                    "SELECT * FROM experiment_equity_curves WHERE experiment_id = ? ORDER BY date",
                    conn,
                    params=(experiment_id,),
                    parse_dates=["date"],
                )
                trades = pd.read_sql_query(
                    "SELECT * FROM experiment_trades WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,),
                    parse_dates=["signal_date", "buy_date", "sell_date"],
                )
                orders = pd.read_sql_query(
                    "SELECT * FROM experiment_orders WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,),
                    parse_dates=["signal_date", "order_date"],
                )
                artifacts = pd.read_sql_query(
                    "SELECT * FROM experiment_artifacts WHERE experiment_id = ? ORDER BY created_at",
                    conn,
                    params=(experiment_id,),
                    parse_dates=["created_at"],
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to get experiment detail: {exc}") from exc

        if not orders.empty and "cost" in orders.columns:
            orders["cost"] = orders["cost"].map(_json_loads)
        return {
            "run": _decode_run(dict(run_row)),
            "equity_curve": equity,
            "trades": trades,
            "orders": orders,
            "artifacts": artifacts,
        }

    def delete_experiment(self, experiment_id: str) -> bool:
        try:
            with self._connect() as conn:
                exists = conn.execute("SELECT 1 FROM experiment_runs WHERE id = ?", (experiment_id,)).fetchone()
                for table in [
                    "experiment_equity_curves",
                    "experiment_trades",
                    "experiment_orders",
                    "experiment_artifacts",
                    "experiment_runs",
                ]:
                    conn.execute(f"DELETE FROM {table} WHERE {'id' if table == 'experiment_runs' else 'experiment_id'} = ?", (experiment_id,))
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to delete experiment: {exc}") from exc
        return exists is not None

    def duplicate_experiment_config(self, experiment_id: str) -> dict[str, Any]:
        detail = self.get_experiment_detail(experiment_id)
        if detail["run"] is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return dict(detail["run"]["config_json"])

    def export_experiment_json(self, experiment_id: str) -> str:
        detail = self.get_experiment_detail(experiment_id)
        if detail["run"] is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        payload = {
            "run": detail["run"],
            "equity_curve": detail["equity_curve"].to_dict("records"),
            "trades": detail["trades"].to_dict("records"),
            "orders": detail["orders"].to_dict("records"),
            "artifacts": detail["artifacts"].to_dict("records"),
        }
        return _json_dumps(payload)

    def add_artifact(self, experiment_id: str, artifact_type: str, file_path: str | Path) -> str:
        self._ensure_experiment(experiment_id)
        artifact_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO experiment_artifacts(id, experiment_id, artifact_type, file_path, created_at)
                    VALUES (:id, :experiment_id, :artifact_type, :file_path, :created_at)
                    """,
                    {
                        "id": artifact_id,
                        "experiment_id": experiment_id,
                        "artifact_type": artifact_type,
                        "file_path": str(file_path),
                        "created_at": _now_iso(),
                    },
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to add artifact: {exc}") from exc
        return artifact_id

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_experiment(self, experiment_id: str) -> None:
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM experiment_runs WHERE id = ?", (experiment_id,)).fetchone()
        if exists is None:
            raise KeyError(f"Experiment not found: {experiment_id}")

    def _replace_rows(self, table: str, experiment_id: str, rows: list[dict[str, Any]]) -> None:
        try:
            with self._connect() as conn:
                conn.execute(f"DELETE FROM {table} WHERE experiment_id = ?", (experiment_id,))
                if rows:
                    columns = list(rows[0])
                    placeholders = ", ".join(f":{column}" for column in columns)
                    conn.executemany(
                        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                        rows,
                    )
                conn.execute(
                    "UPDATE experiment_runs SET updated_at = ? WHERE id = ?",
                    (_now_iso(), experiment_id),
                )
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to save {table}: {exc}") from exc

    def _execute_update(self, sql: str, params: dict[str, Any]) -> None:
        try:
            with self._connect() as conn:
                conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise ExperimentStoreError(f"Failed to update experiment: {exc}") from exc


def _runs_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for metric in [
        "final_equity",
        "total_return",
        "net_total_return",
        "CAGR",
        "net_CAGR",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "profit_loss_ratio",
        "expectancy",
        "total_cost",
        "cost_impact",
        "total_trades",
        "avg_holding_days",
        "exposure_ratio",
        "cash_drag",
    ]:
        frame[metric] = frame["metrics_json"].map(lambda metrics, key=metric: metrics.get(key))
    return frame


def _tags_match(tags: Any, needle: str) -> bool:
    if not needle:
        return True
    if not isinstance(tags, list):
        tags = _json_loads(tags, [])
    return any(needle in str(tag).lower() for tag in (tags or []))


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


def _decode_run(row: dict[str, Any]) -> dict[str, Any]:
    row["tags"] = _json_loads(row.get("tags"), [])
    row["data_snapshot_info"] = _json_loads(row.get("data_snapshot_info"), {})
    row["config_json"] = _json_loads(row.get("config_json"), {})
    row["metrics_json"] = _json_loads(row.get("metrics_json"), {})
    return row


def _equity_record(experiment_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "date": _date_or_none(row.get("date")),
        "gross_equity": _float_or_none(row.get("gross_equity", row.get("gross_total_equity", row.get("equity")))),
        "net_equity": _float_or_none(row.get("net_equity", row.get("total_equity", row.get("equity")))),
        "drawdown": _float_or_none(row.get("drawdown")),
        "cash": _float_or_none(row.get("cash")),
        "market_value": _float_or_none(row.get("market_value")),
        "position_count": _int_or_none(row.get("position_count", row.get("positions_count"))),
    }


def _trade_record(experiment_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "code": _str_or_none(row.get("code")),
        "name": _str_or_none(row.get("name")),
        "strategy_name": _str_or_none(row.get("strategy_name")),
        "signal_date": _date_or_none(row.get("signal_date")),
        "buy_date": _date_or_none(row.get("buy_date")),
        "buy_price": _float_or_none(row.get("buy_price")),
        "sell_date": _date_or_none(row.get("sell_date", row.get("exit_date"))),
        "sell_price": _float_or_none(row.get("sell_price", row.get("exit_price"))),
        "shares": _int_or_none(row.get("shares")),
        "gross_return": _float_or_none(row.get("gross_return")),
        "net_return": _float_or_none(row.get("net_return", row.get("return"))),
        "holding_days": _int_or_none(row.get("holding_days")),
        "exit_reason": _str_or_none(row.get("exit_reason")),
        "commission": _float_or_none(row.get("commission")),
        "stamp_tax": _float_or_none(row.get("stamp_tax")),
        "slippage_cost": _float_or_none(row.get("slippage_cost")),
        "total_cost": _float_or_none(row.get("total_cost")),
    }


def _order_record(experiment_id: str, row: dict[str, Any]) -> dict[str, Any]:
    cost = row.get("cost")
    if not isinstance(cost, dict):
        cost = {
            "commission": row.get("commission", 0),
            "stamp_tax": row.get("stamp_tax", 0),
            "regulatory_fee": row.get("regulatory_fee", 0),
            "transfer_fee": row.get("transfer_fee", 0),
            "total_cost": row.get("total_cost", 0),
        }
    return {
        "id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "code": _str_or_none(row.get("code")),
        "name": _str_or_none(row.get("name")),
        "side": _str_or_none(row.get("side")),
        "signal_date": _date_or_none(row.get("signal_date")),
        "order_date": _date_or_none(row.get("order_date")),
        "requested_price": _float_or_none(row.get("requested_price")),
        "fill_price": _float_or_none(row.get("fill_price")),
        "shares": _int_or_none(row.get("shares")),
        "status": _str_or_none(row.get("status")),
        "reason": _str_or_none(row.get("reason")),
        "gross_amount": _float_or_none(row.get("gross_amount")),
        "cost": _json_dumps(cost),
        "net_amount": _float_or_none(row.get("net_amount")),
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any | None = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(item) for item in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
