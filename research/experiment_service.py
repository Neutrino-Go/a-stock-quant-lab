from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.database import get_database_status
from data_quality.data_health_service import DataHealthService
from research.experiment_store import ExperimentStore


class ExperimentService:
    def __init__(self, store: ExperimentStore | None = None) -> None:
        self.store = store or ExperimentStore()

    def save_portfolio_backtest_result(
        self,
        name: str,
        result: dict[str, object],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        config: dict[str, Any],
        tags: list[str] | None = None,
        notes: str = "",
        conclusion: str = "",
        data_snapshot_info: dict[str, Any] | None = None,
    ) -> str:
        metrics = _clean_metrics(dict(result.get("metrics", {}) or {}))
        experiment_id = self.store.create_experiment_run(
            name=name,
            experiment_type="portfolio_backtest",
            start_date=start_date,
            end_date=end_date,
            strategy_name="ensemble_strategy",
            notes=notes,
            conclusion=conclusion,
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=data_snapshot_info or current_data_snapshot(),
            config=_normalize_config(config),
            metrics=metrics,
        )
        self.store.save_equity_curve(experiment_id, _frame(result.get("portfolio_equity_curve")))
        self.store.save_trades(experiment_id, _frame(result.get("trades")))
        self.store.save_orders(experiment_id, _frame(result.get("orders")))
        return experiment_id

    def save_single_strategy_backtest_result(
        self,
        name: str,
        strategy_name: str,
        result: dict[str, object],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        config: dict[str, Any],
        tags: list[str] | None = None,
        notes: str = "",
        conclusion: str = "",
        data_snapshot_info: dict[str, Any] | None = None,
    ) -> str:
        metrics = {
            key: value
            for key, value in result.items()
            if key not in {"trade_details", "equity_curve", "drawdown_curve"} and not isinstance(value, pd.DataFrame)
        }
        experiment_id = self.store.create_experiment_run(
            name=name,
            experiment_type="single_strategy_backtest",
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            notes=notes,
            conclusion=conclusion,
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=data_snapshot_info or current_data_snapshot(),
            config=_normalize_config(config),
            metrics=_clean_metrics(metrics),
        )
        self.store.save_equity_curve(experiment_id, _frame(result.get("equity_curve")))
        self.store.save_trades(experiment_id, _frame(result.get("trade_details")))
        self.store.save_orders(experiment_id, pd.DataFrame())
        return experiment_id

    def save_optimizer_result(
        self,
        name: str,
        result: pd.DataFrame,
        strategy_name: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        config: dict[str, Any],
        tags: list[str] | None = None,
    ) -> str:
        metrics = {
            "total_rows": int(len(result)),
            "optimizer_ranking": result.to_dict("records") if not result.empty else [],
        }
        return self.store.create_experiment_run(
            name=name,
            experiment_type="parameter_optimization",
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=current_data_snapshot(),
            config=_normalize_config(config),
            metrics=_clean_metrics(metrics),
        )

    def save_leaderboard_snapshot(
        self,
        name: str,
        leaderboard: pd.DataFrame,
        as_of: str | pd.Timestamp,
        config: dict[str, Any],
        tags: list[str] | None = None,
    ) -> str:
        best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
        metrics = {
            "total_rows": int(len(leaderboard)),
            "best_strategy": best,
            "leaderboard": leaderboard.to_dict("records") if not leaderboard.empty else [],
        }
        return self.store.create_experiment_run(
            name=name,
            experiment_type="strategy_leaderboard",
            start_date=as_of,
            end_date=as_of,
            strategy_name="all_strategies",
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=current_data_snapshot(),
            config=_normalize_config(config),
            metrics=_clean_metrics(metrics),
        )

    def save_stability_analysis_result(
        self,
        name: str,
        result: dict[str, object],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        strategy_name: str,
        config: dict[str, Any],
        tags: list[str] | None = None,
        notes: str = "",
        conclusion: str = "",
        status: str = "completed",
        data_snapshot_info: dict[str, Any] | None = None,
    ) -> str:
        metrics = _clean_metrics(dict(result))
        out_sample = dict(result.get("out_of_sample_result", {}) or {})
        stability = dict(result.get("stability_score", {}) or {})
        overfit = dict(result.get("overfit_result", {}) or {})
        metrics.update(
            {
                "strategy_name": strategy_name,
                "net_CAGR": out_sample.get("net_CAGR", metrics.get("net_CAGR", 0)),
                "max_drawdown": out_sample.get("max_drawdown", metrics.get("max_drawdown", 0)),
                "sharpe_ratio": out_sample.get("sharpe_ratio", metrics.get("sharpe_ratio", 0)),
                "total_trades": out_sample.get("total_trades", metrics.get("total_trades", 0)),
                "stability_score_value": stability.get("stability_score", 0),
                "overfit_risk": overfit.get("overfit_risk", ""),
                "overfit_score": overfit.get("overfit_score", 0),
                "recommendation": result.get("recommendation", overfit.get("recommendation", "")),
            }
        )
        return self.store.create_experiment_run(
            name=name,
            experiment_type="strategy_stability_analysis",
            status=status,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            notes=notes,
            conclusion=conclusion,
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=data_snapshot_info or current_data_snapshot(),
            config=_normalize_config(config),
            metrics=metrics,
        )

    def save_candidate_revalidation_result(
        self,
        name: str,
        candidate_id: str,
        result: dict[str, object],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        strategy_name: str,
        config: dict[str, Any],
        tags: list[str] | None = None,
        notes: str = "",
        status: str = "completed",
        data_snapshot_info: dict[str, Any] | None = None,
    ) -> str:
        latest = dict(result.get("latest_metrics", {}) or {})
        comparison = dict(result.get("comparison", {}) or {})
        decision = dict(result.get("decision", {}) or {})
        metrics = _clean_metrics(
            {
                "candidate_id": candidate_id,
                "revalidation_run_id": result.get("revalidation_id"),
                "recommended_action": decision.get("recommended_action"),
                "baseline_metrics": result.get("baseline_metrics", {}),
                "latest_metrics": latest,
                "comparison": comparison,
                "decision": decision,
                "net_CAGR": latest.get("net_CAGR"),
                "max_drawdown": latest.get("max_drawdown"),
                "sharpe_ratio": latest.get("sharpe_ratio"),
                "total_trades": latest.get("total_trades"),
                "overfit_risk": latest.get("overfit_risk"),
                "stability_score": latest.get("stability_score"),
            }
        )
        return self.store.create_experiment_run(
            name=name,
            experiment_type="candidate_revalidation",
            status=status,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            notes=notes,
            tags=tags or [],
            git_commit=current_git_commit(),
            data_snapshot_info=data_snapshot_info or current_data_snapshot(),
            config=_normalize_config(config),
            metrics=metrics,
        )

    def build_experiment_summary(self) -> pd.DataFrame:
        return self.store.list_experiments()

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
        return self.store.list_experiments(
            experiment_type=experiment_type,
            strategy_name=strategy_name,
            rating=rating,
            created_start=created_start,
            created_end=created_end,
            tag=tag,
            keyword=keyword,
            limit=limit,
        )

    def build_experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        return self.store.get_experiment_detail(experiment_id)

    def get_recent_experiments(self, limit: int = 20) -> pd.DataFrame:
        return self.store.list_experiments(limit=limit)

    def get_best_experiments(self, metric: str = "net_CAGR", limit: int = 20) -> pd.DataFrame:
        experiments = self.store.list_experiments()
        if experiments.empty or metric not in experiments.columns:
            return pd.DataFrame()
        result = experiments.copy()
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
        return result.dropna(subset=[metric]).sort_values(metric, ascending=False).head(limit).reset_index(drop=True)

    def get_failed_or_empty_experiments(self) -> pd.DataFrame:
        experiments = self.store.list_experiments()
        if experiments.empty:
            return experiments
        metrics_empty = experiments["metrics_json"].map(lambda value: not bool(value))
        return experiments.loc[experiments["status"].eq("failed") | metrics_empty].reset_index(drop=True)


def current_git_commit(project_root: str | Path | None = None) -> str:
    root = Path(project_root or Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def current_data_snapshot() -> dict[str, Any]:
    try:
        return DataHealthService().build_data_snapshot_info()
    except Exception:
        try:
            legacy = get_database_status()
        except Exception:
            return {"universe_warnings": ["data snapshot unavailable"]}
        legacy["universe_warnings"] = ["data quality snapshot unavailable; using legacy database status"]
        return legacy


def _frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if is_dataclass(value) and not isinstance(value, type):
            normalized[key] = asdict(value)
        elif isinstance(value, dict):
            normalized[key] = _normalize_config(value)
        else:
            normalized[key] = value
    return normalized


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            continue
        clean[key] = value
    if "net_CAGR" not in clean:
        clean["net_CAGR"] = clean.get("CAGR", clean.get("annual_return", 0))
    if "net_total_return" not in clean:
        clean["net_total_return"] = clean.get("total_return", clean.get("cumulative_return", 0))
    return clean
