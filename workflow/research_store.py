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


RESEARCH_STATUSES = {"draft", "running", "completed", "failed", "cancelled"}
RESEARCH_DECISIONS = {"continue_research", "add_to_candidates", "watch", "reject", "failed"}
CANDIDATE_STATUSES = {"active", "watching", "rejected", "archived"}


class ResearchStoreError(RuntimeError):
    pass


class ResearchStore:
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
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"Failed to initialize research tables: {exc}") from exc

    def create_research_run(
        self,
        name: str,
        strategy_name: str,
        status: str = "draft",
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        research_template: str = "standard",
        config: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        decision: str = "failed",
        decision_reason: str = "",
        best_candidate_id: str | None = None,
        notes: str = "",
        tags: list[str] | None = None,
        git_commit: str = "",
    ) -> str:
        _validate_status(status)
        _validate_decision(decision)
        now = _now_iso()
        run_id = str(uuid.uuid4())
        record = {
            "id": run_id,
            "name": name,
            "strategy_name": strategy_name,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "start_date": _date_or_none(start_date),
            "end_date": _date_or_none(end_date),
            "research_template": research_template,
            "config_json": _json_dumps(config or {}),
            "summary_json": _json_dumps(summary or {}),
            "decision": decision,
            "decision_reason": decision_reason,
            "best_candidate_id": best_candidate_id,
            "notes": notes,
            "tags": _json_dumps(tags or []),
            "git_commit": git_commit,
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO research_runs (
                        id, name, strategy_name, status, created_at, updated_at,
                        completed_at, start_date, end_date, research_template,
                        config_json, summary_json, decision, decision_reason,
                        best_candidate_id, notes, tags, git_commit
                    )
                    VALUES (
                        :id, :name, :strategy_name, :status, :created_at, :updated_at,
                        :completed_at, :start_date, :end_date, :research_template,
                        :config_json, :summary_json, :decision, :decision_reason,
                        :best_candidate_id, :notes, :tags, :git_commit
                    )
                    """,
                    record,
                )
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"Failed to create research run: {exc}") from exc
        return run_id

    def update_research_status(self, research_run_id: str, status: str) -> None:
        _validate_status(status)
        completed_at = _now_iso() if status in {"completed", "failed", "cancelled"} else None
        self._execute(
            """
            UPDATE research_runs
            SET status = :status, updated_at = :updated_at, completed_at = COALESCE(:completed_at, completed_at)
            WHERE id = :id
            """,
            {"id": research_run_id, "status": status, "updated_at": _now_iso(), "completed_at": completed_at},
        )

    def save_research_step(
        self,
        research_run_id: str,
        step_name: str,
        status: str,
        message: str = "",
        result: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> str:
        step_id = str(uuid.uuid4())
        now = _now_iso()
        record = {
            "id": step_id,
            "research_run_id": research_run_id,
            "step_name": step_name,
            "status": status,
            "started_at": started_at or now,
            "completed_at": completed_at or now,
            "message": message,
            "result_json": _json_dumps(result or {}),
            "experiment_id": experiment_id,
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO research_steps (
                        id, research_run_id, step_name, status, started_at,
                        completed_at, message, result_json, experiment_id
                    )
                    VALUES (
                        :id, :research_run_id, :step_name, :status, :started_at,
                        :completed_at, :message, :result_json, :experiment_id
                    )
                    """,
                    record,
                )
                conn.execute("UPDATE research_runs SET updated_at = ? WHERE id = ?", (_now_iso(), research_run_id))
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"Failed to save research step: {exc}") from exc
        return step_id

    def update_research_summary(
        self,
        research_run_id: str,
        summary: dict[str, Any],
        status: str | None = None,
        decision: str | None = None,
        decision_reason: str | None = None,
        best_candidate_id: str | None = None,
    ) -> None:
        if status:
            _validate_status(status)
        if decision:
            _validate_decision(decision)
        completed_at = _now_iso() if status in {"completed", "failed", "cancelled"} else None
        self._execute(
            """
            UPDATE research_runs
            SET summary_json = :summary_json,
                status = COALESCE(:status, status),
                decision = COALESCE(:decision, decision),
                decision_reason = COALESCE(:decision_reason, decision_reason),
                best_candidate_id = COALESCE(:best_candidate_id, best_candidate_id),
                completed_at = COALESCE(:completed_at, completed_at),
                updated_at = :updated_at
            WHERE id = :id
            """,
            {
                "id": research_run_id,
                "summary_json": _json_dumps(summary),
                "status": status,
                "decision": decision,
                "decision_reason": decision_reason,
                "best_candidate_id": best_candidate_id,
                "completed_at": completed_at,
                "updated_at": _now_iso(),
            },
        )

    def list_research_runs(
        self,
        strategy_name: str | None = None,
        status: str | None = None,
        decision: str | None = None,
        created_start: str | pd.Timestamp | None = None,
        created_end: str | pd.Timestamp | None = None,
        tag: str | None = None,
        keyword: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if strategy_name and strategy_name != "all":
            conditions.append("strategy_name = :strategy_name")
            params["strategy_name"] = strategy_name
        if status and status != "all":
            conditions.append("status = :status")
            params["status"] = status
        if decision and decision != "all":
            conditions.append("decision = :decision")
            params["decision"] = decision
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
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM research_runs {where} ORDER BY created_at DESC {limit_sql}", params).fetchall()
        frame = _runs_frame([_decode_run(dict(row)) for row in rows])
        if frame.empty:
            return frame
        if tag:
            frame = frame[frame["tags"].map(lambda tags: _tags_match(tags, tag))]
        if keyword:
            frame = frame[_keyword_mask(frame, keyword)]
        return frame.reset_index(drop=True)

    def get_research_run_detail(self, research_run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM research_runs WHERE id = ?", (research_run_id,)).fetchone()
            if run is None:
                return {"run": None, "steps": pd.DataFrame(), "artifacts": pd.DataFrame()}
            steps = pd.read_sql_query(
                "SELECT * FROM research_steps WHERE research_run_id = ? ORDER BY started_at",
                conn,
                params=(research_run_id,),
                parse_dates=["started_at", "completed_at"],
            )
            artifacts = pd.read_sql_query(
                "SELECT * FROM research_artifacts WHERE research_run_id = ? ORDER BY created_at",
                conn,
                params=(research_run_id,),
                parse_dates=["created_at"],
            )
        if not steps.empty:
            steps["result_json"] = steps["result_json"].map(_json_loads)
        return {"run": _decode_run(dict(run)), "steps": steps, "artifacts": artifacts}

    def delete_research_run(self, research_run_id: str) -> bool:
        try:
            with self._connect() as conn:
                exists = conn.execute("SELECT 1 FROM research_runs WHERE id = ?", (research_run_id,)).fetchone()
                conn.execute("DELETE FROM research_steps WHERE research_run_id = ?", (research_run_id,))
                conn.execute("DELETE FROM research_artifacts WHERE research_run_id = ?", (research_run_id,))
                conn.execute("DELETE FROM research_runs WHERE id = ?", (research_run_id,))
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"Failed to delete research run: {exc}") from exc
        return exists is not None

    def create_strategy_candidate(
        self,
        strategy_name: str,
        candidate_name: str,
        params: dict[str, Any],
        scorecard: dict[str, Any],
        metrics: dict[str, Any],
        status: str = "watching",
        source_research_run_id: str | None = None,
        source_experiment_id: str | None = None,
        stability_score: float | None = None,
        overfit_risk: str | None = None,
        decision_reason: str = "",
        notes: str = "",
        tags: list[str] | None = None,
    ) -> str:
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"Unknown candidate status: {status}")
        now = _now_iso()
        candidate_id = str(uuid.uuid4())
        record = {
            "id": candidate_id,
            "strategy_name": strategy_name,
            "candidate_name": candidate_name,
            "status": status,
            "source_research_run_id": source_research_run_id,
            "source_experiment_id": source_experiment_id,
            "created_at": now,
            "updated_at": now,
            "params_json": _json_dumps(params),
            "scorecard_json": _json_dumps(scorecard),
            "metrics_json": _json_dumps(metrics),
            "stability_score": _float_or_none(stability_score if stability_score is not None else scorecard.get("overall_score")),
            "overfit_risk": overfit_risk or str(metrics.get("overfit_risk", "")),
            "net_CAGR": _float_or_none(metrics.get("net_CAGR")),
            "max_drawdown": _float_or_none(metrics.get("max_drawdown")),
            "sharpe_ratio": _float_or_none(metrics.get("sharpe_ratio")),
            "walk_forward_positive_ratio": _float_or_none(metrics.get("walk_forward_positive_ratio")),
            "parameter_stability_score": _float_or_none(metrics.get("parameter_stability_score")),
            "regime_stability_score": _float_or_none(metrics.get("regime_stability_score")),
            "decision_reason": decision_reason,
            "notes": notes,
            "tags": _json_dumps(tags or []),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO strategy_candidates (
                    id, strategy_name, candidate_name, status, source_research_run_id,
                    source_experiment_id, created_at, updated_at, params_json,
                    scorecard_json, metrics_json, stability_score, overfit_risk,
                    net_CAGR, max_drawdown, sharpe_ratio, walk_forward_positive_ratio,
                    parameter_stability_score, regime_stability_score, decision_reason,
                    notes, tags
                )
                VALUES (
                    :id, :strategy_name, :candidate_name, :status, :source_research_run_id,
                    :source_experiment_id, :created_at, :updated_at, :params_json,
                    :scorecard_json, :metrics_json, :stability_score, :overfit_risk,
                    :net_CAGR, :max_drawdown, :sharpe_ratio, :walk_forward_positive_ratio,
                    :parameter_stability_score, :regime_stability_score, :decision_reason,
                    :notes, :tags
                )
                """,
                record,
            )
        return candidate_id

    def update_strategy_candidate(self, candidate_id: str, **fields: Any) -> None:
        allowed = {"candidate_name", "status", "decision_reason", "notes", "tags"}
        assignments = ["updated_at = :updated_at"]
        params: dict[str, Any] = {"id": candidate_id, "updated_at": _now_iso()}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Unsupported candidate field: {key}")
            if key == "status" and value not in CANDIDATE_STATUSES:
                raise ValueError(f"Unknown candidate status: {value}")
            assignments.append(f"{key} = :{key}")
            params[key] = _json_dumps(value) if key == "tags" else value
        self._execute(f"UPDATE strategy_candidates SET {', '.join(assignments)} WHERE id = :id", params)

    def list_strategy_candidates(
        self,
        strategy_name: str | None = None,
        status: str | None = None,
        rating: str | None = None,
        overfit_risk: str | None = None,
        tag: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if strategy_name and strategy_name != "all":
            conditions.append("strategy_name = :strategy_name")
            params["strategy_name"] = strategy_name
        if status and status != "all":
            conditions.append("status = :status")
            params["status"] = status
        if overfit_risk and overfit_risk != "all":
            conditions.append("overfit_risk = :overfit_risk")
            params["overfit_risk"] = overfit_risk
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM strategy_candidates {where} ORDER BY created_at DESC", params).fetchall()
        frame = _candidates_frame([_decode_candidate(dict(row)) for row in rows])
        if frame.empty:
            return frame
        if rating and rating != "all":
            frame = frame[frame["scorecard_json"].map(lambda item: item.get("final_rating") == rating)]
        if tag:
            frame = frame[frame["tags"].map(lambda tags: _tags_match(tags, tag))]
        return frame.reset_index(drop=True)

    def get_strategy_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM strategy_candidates WHERE id = ?", (candidate_id,)).fetchone()
        return _decode_candidate(dict(row)) if row else None

    def archive_strategy_candidate(self, candidate_id: str) -> None:
        self.update_strategy_candidate(candidate_id, status="archived")

    def save_research_artifact(self, research_run_id: str, artifact_type: str, file_path: str | Path) -> str:
        artifact_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_artifacts(id, research_run_id, artifact_type, file_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (artifact_id, research_run_id, artifact_type, str(file_path), _now_iso()),
            )
        return artifact_id

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, sql: str, params: dict[str, Any]) -> None:
        try:
            with self._connect() as conn:
                conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"Failed to update research store: {exc}") from exc


RESEARCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    start_date TEXT,
    end_date TEXT,
    research_template TEXT NOT NULL,
    config_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    decision_reason TEXT,
    best_candidate_id TEXT,
    notes TEXT,
    tags TEXT NOT NULL,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS research_steps (
    id TEXT PRIMARY KEY,
    research_run_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    message TEXT,
    result_json TEXT NOT NULL,
    experiment_id TEXT
);

CREATE TABLE IF NOT EXISTS strategy_candidates (
    id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    candidate_name TEXT NOT NULL,
    status TEXT NOT NULL,
    source_research_run_id TEXT,
    source_experiment_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    params_json TEXT NOT NULL,
    scorecard_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    stability_score REAL,
    overfit_risk TEXT,
    net_CAGR REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    walk_forward_positive_ratio REAL,
    parameter_stability_score REAL,
    regime_stability_score REAL,
    decision_reason TEXT,
    notes TEXT,
    tags TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_artifacts (
    id TEXT PRIMARY KEY,
    research_run_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_runs_created_at ON research_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_research_steps_run ON research_steps(research_run_id);
CREATE INDEX IF NOT EXISTS idx_strategy_candidates_strategy ON strategy_candidates(strategy_name);
CREATE INDEX IF NOT EXISTS idx_research_artifacts_run ON research_artifacts(research_run_id);
"""


def _runs_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["overall_score"] = frame["summary_json"].map(lambda item: dict(item.get("scorecard", {}) or {}).get("overall_score"))
    frame["final_rating"] = frame["summary_json"].map(lambda item: dict(item.get("scorecard", {}) or {}).get("final_rating"))
    return frame


def _candidates_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _decode_run(row: dict[str, Any]) -> dict[str, Any]:
    row["config_json"] = _json_loads(row.get("config_json"), {})
    row["summary_json"] = _json_loads(row.get("summary_json"), {})
    row["tags"] = _json_loads(row.get("tags"), [])
    return row


def _decode_candidate(row: dict[str, Any]) -> dict[str, Any]:
    row["params_json"] = _json_loads(row.get("params_json"), {})
    row["scorecard_json"] = _json_loads(row.get("scorecard_json"), {})
    row["metrics_json"] = _json_loads(row.get("metrics_json"), {})
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return pd.to_datetime(value).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _validate_status(status: str) -> None:
    if status not in RESEARCH_STATUSES:
        raise ValueError(f"Unknown research status: {status}")


def _validate_decision(decision: str) -> None:
    if decision not in RESEARCH_DECISIONS:
        raise ValueError(f"Unknown research decision: {decision}")


def _tags_match(tags: Any, needle: str) -> bool:
    text = str(needle or "").strip().lower()
    if not text:
        return True
    if not isinstance(tags, list):
        tags = _json_loads(tags, [])
    return any(text in str(tag).lower() for tag in tags)


def _keyword_mask(frame: pd.DataFrame, keyword: str) -> pd.Series:
    needle = str(keyword or "").strip().lower()
    if not needle:
        return pd.Series(True, index=frame.index)
    columns = [column for column in ["name", "notes", "decision_reason"] if column in frame.columns]
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        mask = mask | frame[column].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
    return mask
