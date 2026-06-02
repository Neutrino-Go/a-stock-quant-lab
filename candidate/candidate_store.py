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
from workflow.research_store import CANDIDATE_STATUSES, RESEARCH_SCHEMA_SQL


REVALIDATION_STATUSES = {"completed", "failed", "draft"}
REVALIDATION_MODES = {"forward_only", "recent_window", "full_retest"}
RECOMMENDED_ACTIONS = {
    "keep_active",
    "downgrade_to_watching",
    "reject",
    "archive",
    "needs_more_data",
    "continue_monitoring",
}
LIFECYCLE_EVENTS = {"created", "revalidated", "status_changed", "note_updated", "archived", "rejected"}


class CandidateStoreError(RuntimeError):
    pass


class CandidateStore:
    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path == DATABASE_PATH:
            ensure_directories()
        self.init_db()

    def init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(RESEARCH_SCHEMA_SQL)
                conn.executescript(CANDIDATE_SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise CandidateStoreError(f"Failed to initialize candidate tables: {exc}") from exc

    def create_revalidation_run(
        self,
        candidate_id: str,
        strategy_name: str,
        candidate_name: str,
        status: str = "draft",
        mode: str = "recent_window",
        review_start_date: str | pd.Timestamp | None = None,
        review_end_date: str | pd.Timestamp | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        latest_metrics: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        recommended_action: str = "continue_monitoring",
        action_reason: str = "",
        applied_action: bool = False,
        old_candidate_status: str | None = None,
        new_candidate_status: str | None = None,
        source_research_run_id: str | None = None,
        source_experiment_id: str | None = None,
        generated_experiment_id: str | None = None,
        config: dict[str, Any] | None = None,
        notes: str = "",
        tags: list[str] | None = None,
        git_commit: str = "",
        completed_at: str | None = None,
        report_path: str | Path | None = None,
    ) -> str:
        _validate(status, REVALIDATION_STATUSES, "status")
        _validate(mode, REVALIDATION_MODES, "mode")
        _validate(recommended_action, RECOMMENDED_ACTIONS, "recommended_action")
        now = _now_iso()
        run_id = str(uuid.uuid4())
        record = {
            "id": run_id,
            "candidate_id": candidate_id,
            "strategy_name": strategy_name,
            "candidate_name": candidate_name,
            "status": status,
            "mode": mode,
            "created_at": now,
            "completed_at": completed_at or (now if status in {"completed", "failed"} else None),
            "review_start_date": _date_or_none(review_start_date),
            "review_end_date": _date_or_none(review_end_date),
            "baseline_metrics_json": _json_dumps(baseline_metrics or {}),
            "latest_metrics_json": _json_dumps(latest_metrics or {}),
            "comparison_json": _json_dumps(comparison or {}),
            "decision_json": _json_dumps(decision or {}),
            "recommended_action": recommended_action,
            "action_reason": action_reason,
            "applied_action": 1 if applied_action else 0,
            "old_candidate_status": old_candidate_status,
            "new_candidate_status": new_candidate_status,
            "source_research_run_id": source_research_run_id,
            "source_experiment_id": source_experiment_id,
            "generated_experiment_id": generated_experiment_id,
            "config_json": _json_dumps(config or {}),
            "notes": notes,
            "tags": _json_dumps(tags or []),
            "git_commit": git_commit,
            "report_path": str(report_path) if report_path else None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_revalidation_runs (
                    id, candidate_id, strategy_name, candidate_name, status, mode,
                    created_at, completed_at, review_start_date, review_end_date,
                    baseline_metrics_json, latest_metrics_json, comparison_json,
                    decision_json, recommended_action, action_reason, applied_action,
                    old_candidate_status, new_candidate_status, source_research_run_id,
                    source_experiment_id, generated_experiment_id, config_json,
                    notes, tags, git_commit, report_path
                )
                VALUES (
                    :id, :candidate_id, :strategy_name, :candidate_name, :status, :mode,
                    :created_at, :completed_at, :review_start_date, :review_end_date,
                    :baseline_metrics_json, :latest_metrics_json, :comparison_json,
                    :decision_json, :recommended_action, :action_reason, :applied_action,
                    :old_candidate_status, :new_candidate_status, :source_research_run_id,
                    :source_experiment_id, :generated_experiment_id, :config_json,
                    :notes, :tags, :git_commit, :report_path
                )
                """,
                record,
            )
        return run_id

    def update_revalidation_run(self, revalidation_run_id: str, **fields: Any) -> None:
        mapping = {
            "status": "status",
            "latest_metrics": "latest_metrics_json",
            "baseline_metrics": "baseline_metrics_json",
            "comparison": "comparison_json",
            "decision": "decision_json",
            "recommended_action": "recommended_action",
            "action_reason": "action_reason",
            "applied_action": "applied_action",
            "old_candidate_status": "old_candidate_status",
            "new_candidate_status": "new_candidate_status",
            "generated_experiment_id": "generated_experiment_id",
            "report_path": "report_path",
        }
        assignments: list[str] = []
        params: dict[str, Any] = {"id": revalidation_run_id}
        for key, value in fields.items():
            if key not in mapping:
                raise ValueError(f"Unsupported revalidation field: {key}")
            column = mapping[key]
            assignments.append(f"{column} = :{key}")
            if column.endswith("_json"):
                params[key] = _json_dumps(value or {})
            elif key == "applied_action":
                params[key] = 1 if value else 0
            else:
                params[key] = str(value) if isinstance(value, Path) else value
        if "status" in fields and fields["status"] in {"completed", "failed"}:
            assignments.append("completed_at = COALESCE(completed_at, :completed_at)")
            params["completed_at"] = _now_iso()
        if not assignments:
            return
        with self._connect() as conn:
            conn.execute(f"UPDATE candidate_revalidation_runs SET {', '.join(assignments)} WHERE id = :id", params)

    def get_revalidation_run(self, revalidation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM candidate_revalidation_runs WHERE id = ?", (revalidation_id,)).fetchone()
        return _decode_revalidation(dict(row)) if row else None

    def list_revalidation_runs(
        self,
        candidate_id: str | None = None,
        strategy_name: str | None = None,
        mode: str | None = None,
        recommended_action: str | None = None,
        created_start: str | pd.Timestamp | None = None,
        created_end: str | pd.Timestamp | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if candidate_id:
            conditions.append("candidate_id = :candidate_id")
            params["candidate_id"] = candidate_id
        if strategy_name and strategy_name != "all":
            conditions.append("strategy_name = :strategy_name")
            params["strategy_name"] = strategy_name
        if mode and mode != "all":
            conditions.append("mode = :mode")
            params["mode"] = mode
        if recommended_action and recommended_action != "all":
            conditions.append("recommended_action = :recommended_action")
            params["recommended_action"] = recommended_action
        if status and status != "all":
            conditions.append("status = :status")
            params["status"] = status
        if created_start is not None:
            conditions.append("date(created_at) >= date(:created_start)")
            params["created_start"] = _date_or_none(created_start)
        if created_end is not None:
            conditions.append("date(created_at) <= date(:created_end)")
            params["created_end"] = _date_or_none(created_end)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM candidate_revalidation_runs {where} ORDER BY created_at DESC", params).fetchall()
        return _revalidations_frame([_decode_revalidation(dict(row)) for row in rows])

    def save_metric_snapshot(
        self,
        candidate_id: str,
        revalidation_run_id: str,
        snapshot_date: str | pd.Timestamp,
        data_start_date: str | pd.Timestamp | None,
        data_end_date: str | pd.Timestamp | None,
        metrics: dict[str, Any],
    ) -> str:
        snapshot_id = str(uuid.uuid4())
        record = {
            "id": snapshot_id,
            "candidate_id": candidate_id,
            "revalidation_run_id": revalidation_run_id,
            "snapshot_date": _date_or_none(snapshot_date),
            "data_start_date": _date_or_none(data_start_date),
            "data_end_date": _date_or_none(data_end_date),
            "net_CAGR": _float_or_none(metrics.get("net_CAGR")),
            "max_drawdown": _float_or_none(metrics.get("max_drawdown")),
            "sharpe_ratio": _float_or_none(metrics.get("sharpe_ratio")),
            "win_rate": _float_or_none(metrics.get("win_rate")),
            "total_trades": _float_or_none(metrics.get("total_trades")),
            "walk_forward_positive_ratio": _float_or_none(metrics.get("walk_forward_positive_ratio")),
            "parameter_stability_score": _float_or_none(metrics.get("parameter_stability_score")),
            "overfit_risk": metrics.get("overfit_risk"),
            "stability_score": _float_or_none(metrics.get("stability_score")),
            "overall_score": _float_or_none(metrics.get("overall_score")),
            "cost_impact": _float_or_none(metrics.get("cost_impact")),
            "metrics_json": _json_dumps(metrics),
            "created_at": _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_metric_snapshots (
                    id, candidate_id, revalidation_run_id, snapshot_date,
                    data_start_date, data_end_date, net_CAGR, max_drawdown,
                    sharpe_ratio, win_rate, total_trades, walk_forward_positive_ratio,
                    parameter_stability_score, overfit_risk, stability_score,
                    overall_score, cost_impact, metrics_json, created_at
                )
                VALUES (
                    :id, :candidate_id, :revalidation_run_id, :snapshot_date,
                    :data_start_date, :data_end_date, :net_CAGR, :max_drawdown,
                    :sharpe_ratio, :win_rate, :total_trades, :walk_forward_positive_ratio,
                    :parameter_stability_score, :overfit_risk, :stability_score,
                    :overall_score, :cost_impact, :metrics_json, :created_at
                )
                """,
                record,
            )
        return snapshot_id

    def list_metric_snapshots(self, candidate_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_metric_snapshots WHERE candidate_id = ? ORDER BY snapshot_date DESC, created_at DESC",
                (candidate_id,),
            ).fetchall()
        frame = pd.DataFrame([_decode_snapshot(dict(row)) for row in rows])
        return frame.reset_index(drop=True) if not frame.empty else frame

    def create_lifecycle_event(
        self,
        candidate_id: str,
        event_type: str,
        old_status: str | None = None,
        new_status: str | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        _validate(event_type, LIFECYCLE_EVENTS, "event_type")
        event_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_lifecycle_events (
                    id, candidate_id, event_type, old_status, new_status,
                    message, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, candidate_id, event_type, old_status, new_status, message, _now_iso(), _json_dumps(metadata or {})),
            )
        return event_id

    def list_lifecycle_events(
        self,
        candidate_id: str | None = None,
        event_type: str | None = None,
        created_start: str | pd.Timestamp | None = None,
        created_end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if candidate_id:
            conditions.append("e.candidate_id = :candidate_id")
            params["candidate_id"] = candidate_id
        if event_type and event_type != "all":
            conditions.append("e.event_type = :event_type")
            params["event_type"] = event_type
        if created_start is not None:
            conditions.append("date(e.created_at) >= date(:created_start)")
            params["created_start"] = _date_or_none(created_start)
        if created_end is not None:
            conditions.append("date(e.created_at) <= date(:created_end)")
            params["created_end"] = _date_or_none(created_end)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.*, c.candidate_name, c.strategy_name
                FROM candidate_lifecycle_events e
                LEFT JOIN strategy_candidates c ON c.id = e.candidate_id
                {where}
                ORDER BY e.created_at DESC
                """,
                params,
            ).fetchall()
        frame = pd.DataFrame([_decode_event(dict(row)) for row in rows])
        return frame.reset_index(drop=True) if not frame.empty else frame

    def apply_candidate_status_change(self, candidate_id: str, new_status: str, reason: str) -> bool:
        _validate(new_status, CANDIDATE_STATUSES, "candidate status")
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM strategy_candidates WHERE id = ?", (candidate_id,)).fetchone()
            if row is None:
                return False
            old_status = row["status"]
            conn.execute("UPDATE strategy_candidates SET status = ?, updated_at = ? WHERE id = ?", (new_status, _now_iso(), candidate_id))
        event_type = "archived" if new_status == "archived" else "rejected" if new_status == "rejected" else "status_changed"
        self.create_lifecycle_event(candidate_id, event_type, old_status, new_status, reason)
        return True

    def list_candidates_due_for_review(self, max_age_days: int = 30) -> pd.DataFrame:
        with self._connect() as conn:
            candidates = conn.execute(
                "SELECT * FROM strategy_candidates WHERE status IN ('active', 'watching') ORDER BY created_at DESC"
            ).fetchall()
            latest_rows = conn.execute(
                """
                SELECT r.*
                FROM candidate_revalidation_runs r
                JOIN (
                    SELECT candidate_id, MAX(created_at) AS max_created_at
                    FROM candidate_revalidation_runs
                    WHERE status = 'completed'
                    GROUP BY candidate_id
                ) latest ON latest.candidate_id = r.candidate_id AND latest.max_created_at = r.created_at
                """
            ).fetchall()
        latest_by_candidate = {_decode_revalidation(dict(row))["candidate_id"]: _decode_revalidation(dict(row)) for row in latest_rows}
        now = pd.Timestamp.now(tz="UTC")
        rows = []
        for raw in candidates:
            candidate = _decode_candidate(dict(raw))
            latest = latest_by_candidate.get(candidate["id"])
            last_date = _to_timestamp(latest.get("created_at")) if latest else None
            reference_date = last_date or _to_timestamp(candidate.get("created_at"))
            days = int((now - reference_date).days) if reference_date is not None else None
            if latest is not None and days is not None and days <= max_age_days:
                continue
            latest_metrics = dict(latest.get("latest_metrics_json", {}) if latest else {})
            row = {
                **candidate,
                "last_revalidation_at": latest.get("created_at") if latest else None,
                "days_since_revalidation": days,
                "latest_recommended_action": latest.get("recommended_action") if latest else None,
                "latest_net_CAGR": latest_metrics.get("net_CAGR"),
                "latest_max_drawdown": latest_metrics.get("max_drawdown"),
                "latest_overfit_risk": latest_metrics.get("overfit_risk"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_candidate_with_latest_revalidation(self, candidate_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            candidate = conn.execute("SELECT * FROM strategy_candidates WHERE id = ?", (candidate_id,)).fetchone()
            latest = conn.execute(
                "SELECT * FROM candidate_revalidation_runs WHERE candidate_id = ? ORDER BY created_at DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return {
            "candidate": _decode_candidate(dict(candidate)) if candidate else None,
            "latest_revalidation": _decode_revalidation(dict(latest)) if latest else None,
        }

    def export_candidates_csv(self, candidates: pd.DataFrame, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "id",
            "candidate_name",
            "strategy_name",
            "status",
            "stability_score",
            "overfit_risk",
            "net_CAGR",
            "max_drawdown",
            "sharpe_ratio",
            "walk_forward_positive_ratio",
            "parameter_stability_score",
            "created_at",
        ]
        frame = candidates.copy() if isinstance(candidates, pd.DataFrame) and not candidates.empty else pd.DataFrame(columns=columns)
        if "code" in frame.columns:
            frame["code"] = frame["code"].map(lambda value: str(value).strip().zfill(6))
            frame["code_text"] = frame["code"].map(lambda value: f'="{value}"')
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


CANDIDATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candidate_revalidation_runs (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    candidate_name TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    review_start_date TEXT,
    review_end_date TEXT,
    baseline_metrics_json TEXT NOT NULL,
    latest_metrics_json TEXT NOT NULL,
    comparison_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    action_reason TEXT,
    applied_action INTEGER NOT NULL DEFAULT 0,
    old_candidate_status TEXT,
    new_candidate_status TEXT,
    source_research_run_id TEXT,
    source_experiment_id TEXT,
    generated_experiment_id TEXT,
    config_json TEXT NOT NULL,
    notes TEXT,
    tags TEXT NOT NULL,
    git_commit TEXT,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS candidate_metric_snapshots (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    revalidation_run_id TEXT NOT NULL,
    snapshot_date TEXT,
    data_start_date TEXT,
    data_end_date TEXT,
    net_CAGR REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    win_rate REAL,
    total_trades REAL,
    walk_forward_positive_ratio REAL,
    parameter_stability_score REAL,
    overfit_risk TEXT,
    stability_score REAL,
    overall_score REAL,
    cost_impact REAL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_lifecycle_events (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    message TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidate_revalidation_candidate ON candidate_revalidation_runs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_snapshots_candidate ON candidate_metric_snapshots(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_lifecycle_candidate ON candidate_lifecycle_events(candidate_id);
"""


def _revalidations_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["latest_net_CAGR"] = frame["latest_metrics_json"].map(lambda item: dict(item or {}).get("net_CAGR"))
    frame["latest_max_drawdown"] = frame["latest_metrics_json"].map(lambda item: dict(item or {}).get("max_drawdown"))
    frame["latest_overfit_risk"] = frame["latest_metrics_json"].map(lambda item: dict(item or {}).get("overfit_risk"))
    return frame.reset_index(drop=True)


def _decode_revalidation(row: dict[str, Any]) -> dict[str, Any]:
    for key in ["baseline_metrics_json", "latest_metrics_json", "comparison_json", "decision_json", "config_json", "tags"]:
        row[key] = _json_loads(row.get(key), [] if key == "tags" else {})
    row["applied_action"] = bool(row.get("applied_action"))
    return row


def _decode_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    row["metrics_json"] = _json_loads(row.get("metrics_json"), {})
    return row


def _decode_event(row: dict[str, Any]) -> dict[str, Any]:
    row["metadata_json"] = _json_loads(row.get("metadata_json"), {})
    return row


def _decode_candidate(row: dict[str, Any]) -> dict[str, Any]:
    for key in ["params_json", "scorecard_json", "metrics_json"]:
        row[key] = _json_loads(row.get(key), {})
    row["tags"] = _json_loads(row.get("tags"), [])
    return row


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any | None = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, dict | list):
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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return pd.to_datetime(value).date().isoformat()


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(ts) else ts


def _validate(value: str, allowed: set[str], field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"Unknown {field_name}: {value}")
