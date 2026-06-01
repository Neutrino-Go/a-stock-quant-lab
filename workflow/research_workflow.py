from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.database import get_stock_daily
from execution.cost_model import TradingCostConfig
from factor.factor_engine import compute_factors
from research.experiment_service import ExperimentService
from stability.stability_service import DEFAULT_STABILITY_PARAMETER_GRIDS, StrategyStabilityService
from stability.walk_forward import iter_parameter_grid
from workflow.research_models import ResearchConfig, ResearchRunResult, ResearchStepResult
from workflow.research_report import export_research_json_summary, export_research_markdown_report
from workflow.research_scorecard import build_strategy_scorecard
from workflow.research_store import ResearchStore


class StrategyResearchWorkflow:
    def __init__(
        self,
        research_store: ResearchStore | None = None,
        stability_service: StrategyStabilityService | None = None,
        experiment_service: ExperimentService | None = None,
        factor_data_loader=None,
        report_output_dir: str | Path | None = None,
        git_commit_loader=None,
    ) -> None:
        self.store = research_store or ResearchStore()
        self.stability_service = stability_service or StrategyStabilityService()
        self.experiment_service = experiment_service or getattr(self.stability_service, "experiment_service", None)
        self.factor_data_loader = factor_data_loader or _load_factor_data
        self.report_output_dir = Path(report_output_dir) if report_output_dir else None
        self.git_commit_loader = git_commit_loader or current_git_commit

    def run(self, config: ResearchConfig) -> ResearchRunResult:
        research_run_id = self.store.create_research_run(
            name=config.name or f"{config.strategy_name} {config.research_template} research",
            strategy_name=config.strategy_name,
            status="running",
            start_date=config.start_date,
            end_date=config.end_date,
            research_template=config.research_template,
            config=config.to_dict(),
            notes=config.notes,
            tags=config.tags,
            git_commit=self.git_commit_loader(),
        )
        steps: list[ResearchStepResult] = []
        artifacts: list[dict[str, Any]] = []
        try:
            factor_data = self._data_check(config, research_run_id, steps)
            optimization_result = self._parameter_optimization(config, research_run_id, steps, factor_data)
            stability_results = self._stability_analysis(config, research_run_id, steps, optimization_result)
            scorecard = self._scorecard_generation(config, research_run_id, steps, optimization_result, stability_results)
            candidate_id = self._candidate_promotion(config, research_run_id, steps, scorecard, stability_results)
            summary = self._summary(scorecard, stability_results, candidate_id, optimization_result)
            self.store.update_research_summary(
                research_run_id,
                summary,
                decision=scorecard["decision"],
                decision_reason=scorecard["decision_reason"],
                best_candidate_id=candidate_id,
            )
            report_artifacts = self._report_export(research_run_id, steps)
            artifacts.extend(report_artifacts)
            self.store.update_research_summary(
                research_run_id,
                summary,
                status="completed",
                decision=scorecard["decision"],
                decision_reason=scorecard["decision_reason"],
                best_candidate_id=candidate_id,
            )
            return ResearchRunResult(
                research_run_id=research_run_id,
                strategy_name=config.strategy_name,
                status="completed",
                summary=summary,
                steps=steps,
                scorecard=scorecard,
                decision=scorecard["decision"],
                candidate_id=candidate_id,
                artifacts=artifacts,
            )
        except Exception as exc:
            step = self._save_step(
                research_run_id,
                "workflow_failed",
                "failed",
                f"{type(exc).__name__}: {exc}",
                {"error": str(exc)},
            )
            steps.append(step)
            summary = {"error": str(exc), "scorecard": {"overall_score": 0, "decision": "failed"}}
            self.store.update_research_summary(
                research_run_id,
                summary,
                status="failed",
                decision="failed",
                decision_reason=str(exc),
            )
            return ResearchRunResult(
                research_run_id=research_run_id,
                strategy_name=config.strategy_name,
                status="failed",
                summary=summary,
                steps=steps,
                scorecard=summary["scorecard"],
                decision="failed",
                candidate_id=None,
                artifacts=[],
            )

    def _data_check(self, config: ResearchConfig, research_run_id: str, steps: list[ResearchStepResult]) -> pd.DataFrame:
        factor_data = self.factor_data_loader(start_date=config.start_date, end_date=config.end_date)
        if factor_data is None or factor_data.empty:
            raise ValueError("No local factor data is available for the selected research period.")
        dates = pd.to_datetime(factor_data["date"], errors="coerce").dropna() if "date" in factor_data.columns else pd.Series(dtype="datetime64[ns]")
        if dates.nunique() < 60:
            raise ValueError("Research period has fewer than 60 valid trading days.")
        result = {
            "start_date": str(config.start_date),
            "end_date": str(config.end_date),
            "trade_days": int(dates.nunique()),
            "rows": int(len(factor_data)),
            "stock_count": int(factor_data["code"].nunique()) if "code" in factor_data.columns else 0,
        }
        steps.append(self._save_step(research_run_id, "data_check", "completed", "Data check completed.", result))
        return factor_data

    def _parameter_optimization(
        self,
        config: ResearchConfig,
        research_run_id: str,
        steps: list[ResearchStepResult],
        factor_data: pd.DataFrame,
    ) -> dict[str, Any]:
        grid = config.parameter_grid or DEFAULT_STABILITY_PARAMETER_GRIDS.get(config.strategy_name, {})
        candidates = iter_parameter_grid(grid)
        if not candidates:
            raise ValueError(f"No parameter grid configured for {config.strategy_name}.")

        if hasattr(self.stability_service, "_run_parameter_search"):
            search_rows = self.stability_service._run_parameter_search(
                strategy_name=config.strategy_name,
                parameter_grid=grid,
                start_date=str(config.start_date),
                end_date=str(config.end_date),
                factor_data=factor_data,
                cost_config=_cost_config(config.cost_config),
                exit_config=config.exit_config,
            )
        else:
            search_rows = [{"params": params, "metrics": {}} for params in candidates]

        ranked_rows = _rank_parameter_rows(search_rows)
        top_params = [
            {"rank": idx + 1, "params": dict(row.get("params", {}) or {}), "metrics": dict(row.get("metrics", {}) or {})}
            for idx, row in enumerate(ranked_rows[: config.top_n_params])
        ]
        result = {"top_params": top_params, "total_combinations": len(candidates), "top_n_params": config.top_n_params}
        experiment_id = self._save_optimizer_experiment(config, result) if config.save_intermediate_experiments else None
        steps.append(
            self._save_step(
                research_run_id,
                "parameter_optimization",
                "completed",
                "Top parameter sets selected by parameter search.",
                result,
                experiment_id=experiment_id,
            )
        )
        return result

    def _stability_analysis(
        self,
        config: ResearchConfig,
        research_run_id: str,
        steps: list[ResearchStepResult],
        optimization_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results = []
        for row in optimization_result.get("top_params", []):
            params = dict(row.get("params", {}) or {})
            single_value_grid = {key: [value] for key, value in params.items()}
            result = self.stability_service.run_stability_analysis(
                strategy_name=config.strategy_name,
                parameter_grid=single_value_grid,
                start_date=config.start_date,
                end_date=config.end_date,
                train_ratio=config.train_ratio,
                walk_forward_config=config.walk_forward_config,
                cost_config=_cost_config(config.cost_config),
                position_config=config.position_config,
                exit_config=config.exit_config,
                market_config=config.market_config,
                save_as_experiment=config.save_intermediate_experiments,
                experiment_name=f"{config.strategy_name} stability {row.get('rank')}",
                tags=config.tags,
                notes=config.notes,
            )
            results.append(result)
        steps.append(
            self._save_step(
                research_run_id,
                "stability_analysis",
                "completed",
                "Stability analysis completed for selected parameter sets.",
                {"count": len(results), "experiment_ids": [item.get("analysis_id") for item in results]},
                experiment_id=results[0].get("analysis_id") if results else None,
            )
        )
        return results

    def _scorecard_generation(
        self,
        config: ResearchConfig,
        research_run_id: str,
        steps: list[ResearchStepResult],
        optimization_result: dict[str, Any],
        stability_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scorecard = build_strategy_scorecard(
            optimization_result=optimization_result,
            stability_results=stability_results,
            overfit_results=[dict(item.get("overfit_result", {}) or {}) for item in stability_results],
            sensitivity_results=[dict(item.get("sensitivity_result", {}) or {}) for item in stability_results],
            regime_results=[dict(item.get("market_regime_breakdown", {}) or {}) for item in stability_results],
            thresholds=config.decision_thresholds,
        )
        steps.append(self._save_step(research_run_id, "scorecard_generation", "completed", "Research scorecard generated.", scorecard))
        return scorecard

    def _candidate_promotion(
        self,
        config: ResearchConfig,
        research_run_id: str,
        steps: list[ResearchStepResult],
        scorecard: dict[str, Any],
        stability_results: list[dict[str, Any]],
    ) -> str | None:
        if scorecard.get("decision") != "add_to_candidates":
            steps.append(self._save_step(research_run_id, "candidate_promotion", "completed", "No candidate was created.", {"candidate_id": None}))
            return None
        best = _best_stability_result(stability_results, scorecard.get("best_experiment_id"))
        metrics = _candidate_metrics(best)
        candidate_id = self.store.create_strategy_candidate(
            strategy_name=config.strategy_name,
            candidate_name=f"{config.strategy_name} candidate {scorecard.get('overall_score', 0):.1f}",
            source_research_run_id=research_run_id,
            source_experiment_id=scorecard.get("best_experiment_id"),
            params=dict(scorecard.get("best_params", {}) or {}),
            scorecard=scorecard,
            metrics=metrics,
            stability_score=float(scorecard.get("overall_score", 0)),
            overfit_risk=metrics.get("overfit_risk"),
            decision_reason=scorecard.get("decision_reason", ""),
            tags=config.tags,
            status="active",
        )
        steps.append(self._save_step(research_run_id, "candidate_promotion", "completed", "Candidate created.", {"candidate_id": candidate_id}))
        return candidate_id

    def _report_export(self, research_run_id: str, steps: list[ResearchStepResult]) -> list[dict[str, Any]]:
        output_dir = self.report_output_dir
        md_path = export_research_markdown_report(self.store, research_run_id, output_dir=output_dir) if output_dir else export_research_markdown_report(self.store, research_run_id)
        json_path = export_research_json_summary(self.store, research_run_id, output_dir=output_dir) if output_dir else export_research_json_summary(self.store, research_run_id)
        result = {"markdown_report": str(md_path), "json_summary": str(json_path)}
        steps.append(self._save_step(research_run_id, "report_export", "completed", "Research report exported.", result))
        return [{"artifact_type": "markdown_report", "file_path": str(md_path)}, {"artifact_type": "json_summary", "file_path": str(json_path)}]

    def _save_optimizer_experiment(self, config: ResearchConfig, result: dict[str, Any]) -> str | None:
        if self.experiment_service is None:
            return None
        rows = []
        for item in result.get("top_params", []) or []:
            metrics = dict(item.get("metrics", {}) or {})
            rows.append(
                {
                    "rank": item.get("rank"),
                    "strategy_name": config.strategy_name,
                    "parameters": item.get("params", {}),
                    "win_rate": metrics.get("win_rate", 0),
                    "annual_return": metrics.get("net_CAGR", metrics.get("CAGR", 0)),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "total_trades": metrics.get("total_trades", 0),
                }
            )
        try:
            return self.experiment_service.save_optimizer_result(
                name=f"{config.strategy_name} research parameter optimization",
                result=pd.DataFrame(rows),
                strategy_name=config.strategy_name,
                start_date=config.start_date,
                end_date=config.end_date,
                config={"research_config": config.to_dict()},
                tags=config.tags,
            )
        except Exception:
            return None

    def _summary(
        self,
        scorecard: dict[str, Any],
        stability_results: list[dict[str, Any]],
        candidate_id: str | None,
        optimization_result: dict[str, Any],
    ) -> dict[str, Any]:
        best = _best_stability_result(stability_results, scorecard.get("best_experiment_id"))
        metrics = _candidate_metrics(best)
        return {
            "scorecard": scorecard,
            "candidate_id": candidate_id,
            "optimization_result": optimization_result,
            "core_metrics": {
                "out_of_sample_net_CAGR": metrics.get("net_CAGR", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "walk_forward_positive_ratio": metrics.get("walk_forward_positive_ratio", 0),
                "parameter_stability_score": metrics.get("parameter_stability_score", 0),
                "overfit_risk": metrics.get("overfit_risk", ""),
                "cost_impact": metrics.get("cost_impact", 0),
            },
            "stability_results": stability_results,
        }

    def _save_step(
        self,
        research_run_id: str,
        step_name: str,
        status: str,
        message: str,
        result: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> ResearchStepResult:
        self.store.save_research_step(research_run_id, step_name, status, message=message, result=result or {}, experiment_id=experiment_id)
        return ResearchStepResult(step_name=step_name, status=status, message=message, result=result or {}, experiment_id=experiment_id)


def _load_factor_data(start_date=None, end_date=None) -> pd.DataFrame:
    daily = get_stock_daily(start_date=start_date, end_date=end_date)
    return compute_factors(daily) if not daily.empty else daily


def _cost_config(value: dict[str, Any] | TradingCostConfig) -> TradingCostConfig:
    if isinstance(value, TradingCostConfig):
        return value
    return TradingCostConfig(**dict(value or {}))


def _best_stability_result(results: list[dict[str, Any]], experiment_id: Any) -> dict[str, Any]:
    for result in results:
        if result.get("analysis_id") == experiment_id:
            return result
    return results[0] if results else {}


def _candidate_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("out_of_sample_result", {}) or {})
    walk = dict(dict(result.get("walk_forward_result", {}) or {}).get("summary", {}) or {})
    sensitivity = dict(result.get("sensitivity_result", {}) or {})
    stability = dict(result.get("stability_score", {}) or {})
    overfit = dict(result.get("overfit_result", {}) or {})
    return {
        "net_CAGR": metrics.get("net_CAGR", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "walk_forward_positive_ratio": walk.get("positive_test_fold_ratio", 0),
        "parameter_stability_score": sensitivity.get("sensitivity_score", 0),
        "regime_stability_score": stability.get("regime_stability_score", 0),
        "overfit_risk": overfit.get("overfit_risk", ""),
        "cost_impact": metrics.get("cost_impact", 0),
        "total_trades": metrics.get("total_trades", 0),
    }


def _rank_parameter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = dict(row.get("metrics", {}) or {})
        return (
            _metric(metrics, "net_CAGR", "CAGR", "annual_return"),
            _metric(metrics, "sharpe_ratio"),
            _metric(metrics, "max_drawdown"),
            _metric(metrics, "win_rate"),
        )

    return sorted(rows, key=key, reverse=True)


def _metric(metrics: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in metrics:
            try:
                return float(metrics.get(name) or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def current_git_commit(project_root: str | Path | None = None) -> str:
    root = Path(project_root or Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def config_to_dict(value: Any) -> Any:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return {key: config_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    return value
