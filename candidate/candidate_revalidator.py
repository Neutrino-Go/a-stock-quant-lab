from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from candidate.candidate_decision import evaluate_candidate_lifecycle
from candidate.candidate_models import CandidateComparison, CandidateRevalidationConfig, CandidateRevalidationResult
from candidate.candidate_store import CandidateStore
from data.database import get_database_status
from execution.cost_model import TradingCostConfig
from research.experiment_service import ExperimentService
from stability.stability_service import StrategyStabilityService
from workflow.research_store import ResearchStore


class CandidateRevalidator:
    def __init__(
        self,
        research_store: ResearchStore | None = None,
        candidate_store: CandidateStore | None = None,
        stability_service: StrategyStabilityService | None = None,
        experiment_service: ExperimentService | None = None,
        latest_date_loader=None,
        git_commit_loader=None,
    ) -> None:
        self.research_store = research_store or ResearchStore()
        self.candidate_store = candidate_store or CandidateStore()
        self.stability_service = stability_service or StrategyStabilityService()
        self.experiment_service = experiment_service or ExperimentService()
        self.latest_date_loader = latest_date_loader or latest_local_data_date
        self.git_commit_loader = git_commit_loader or current_git_commit

    def run_revalidation(self, config: CandidateRevalidationConfig) -> CandidateRevalidationResult:
        candidate = self.research_store.get_strategy_candidate(config.candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate not found: {config.candidate_id}")
        baseline = _baseline_metrics(candidate)
        params = dict(candidate.get("params_json", {}) or {})
        if not params:
            return self._failed_result(config, candidate, baseline, "Candidate params_json is empty; revalidation cannot run.")

        window = self._resolve_review_window(candidate, config, self.latest_date_loader())
        start_date, end_date = window.start_date, window.end_date
        if start_date is None or end_date is None or pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            latest = {"total_trades": 0, "net_CAGR": None, "max_drawdown": None, "overfit_risk": "UNKNOWN"}
            comparison = build_candidate_comparison(baseline, latest).to_dict()
            decision = evaluate_candidate_lifecycle(baseline, latest, comparison, config.decision_thresholds)
            _attach_window_warnings(decision, window, invalid_period=True)
            run_id = self._save_run(config, candidate, baseline, latest, comparison, decision, start_date, end_date, status="completed", window=window)
            self.candidate_store.create_lifecycle_event(config.candidate_id, "revalidated", candidate.get("status"), candidate.get("status"), decision["action_reason"], {"revalidation_id": run_id})
            return CandidateRevalidationResult(
                revalidation_id=run_id,
                candidate_id=config.candidate_id,
                status="completed",
                mode=config.mode,
                baseline_metrics=baseline,
                latest_metrics=latest,
                comparison=comparison,
                decision=decision,
                recommended_action=decision["recommended_action"],
                action_reason=decision["action_reason"],
            )

        stability_result = self.stability_service.run_stability_analysis(
            strategy_name=candidate["strategy_name"],
            parameter_grid={key: [value] for key, value in params.items()},
            start_date=start_date,
            end_date=end_date,
            walk_forward_config=config.walk_forward_config or {"train_months": 24, "test_months": 6, "step_months": 6},
            cost_config=_cost_config(config.cost_config),
            position_config=config.position_config,
            exit_config=config.exit_config,
            market_config={"use_market_regime": config.use_market_regime},
            save_as_experiment=False,
            tags=config.tags,
            notes=config.notes,
        )
        latest = _latest_metrics(stability_result)
        comparison_obj = build_candidate_comparison(baseline, latest)
        comparison = comparison_obj.to_dict()
        decision = evaluate_candidate_lifecycle(baseline, latest, comparison, config.decision_thresholds)
        _attach_window_warnings(decision, window)
        run_id = self._save_run(
            config,
            candidate,
            baseline,
            latest,
            comparison,
            decision,
            start_date,
            end_date,
            status="completed",
            window=window,
        )
        generated_experiment_id = None
        if config.save_as_experiment:
            generated_experiment_id = self.experiment_service.save_candidate_revalidation_result(
                name=f"{candidate['candidate_name']} candidate revalidation",
                candidate_id=config.candidate_id,
                result={
                    "revalidation_id": run_id,
                    "baseline_metrics": baseline,
                    "latest_metrics": latest,
                    "comparison": comparison,
                    "decision": decision,
                    "revalidation_result": stability_result,
                },
                start_date=start_date,
                end_date=end_date,
                strategy_name=candidate["strategy_name"],
                config=config.to_dict(),
                tags=config.tags,
                notes=config.notes,
            )
            self.candidate_store.update_revalidation_run(run_id, generated_experiment_id=generated_experiment_id)
        self.candidate_store.save_metric_snapshot(
            config.candidate_id,
            run_id,
            snapshot_date=end_date,
            data_start_date=start_date,
            data_end_date=end_date,
            metrics={**latest, "overall_score": latest.get("stability_score")},
        )
        self.candidate_store.create_lifecycle_event(
            config.candidate_id,
            "revalidated",
            candidate.get("status"),
            candidate.get("status"),
            decision["action_reason"],
            {"revalidation_id": run_id, "recommended_action": decision["recommended_action"]},
        )
        return CandidateRevalidationResult(
            revalidation_id=run_id,
            candidate_id=config.candidate_id,
            status="completed",
            mode=config.mode,
            baseline_metrics=baseline,
            latest_metrics=latest,
            comparison=comparison,
            decision=decision,
            recommended_action=decision["recommended_action"],
            action_reason=decision["action_reason"],
            generated_experiment_id=generated_experiment_id,
        )

    def _resolve_review_window(
        self,
        candidate: dict[str, Any],
        config: CandidateRevalidationConfig,
        latest_date: str | pd.Timestamp | None,
    ) -> "ReviewWindow":
        source_period = self._resolve_source_period(candidate) if config.mode == "full_retest" else None
        return determine_review_window(candidate, config, latest_date, source_period=source_period)

    def _resolve_source_period(self, candidate: dict[str, Any]) -> tuple[str | None, str | None, str] | None:
        research_run_id = candidate.get("source_research_run_id")
        if research_run_id:
            detail = self.research_store.get_research_run_detail(str(research_run_id))
            run = dict(detail.get("run") or {})
            period = _period_from_mapping(run)
            if period:
                return period[0], period[1], "source_research_run"

        experiment_id = candidate.get("source_experiment_id")
        if experiment_id and hasattr(self.experiment_service, "build_experiment_detail"):
            detail = self.experiment_service.build_experiment_detail(str(experiment_id))
            run = dict(detail.get("run") or {})
            period = _period_from_mapping(run)
            if period:
                return period[0], period[1], "source_experiment"

        for key in ["config_json", "metrics_json", "scorecard_json"]:
            period = _period_from_mapping(candidate.get(key))
            if period:
                return period[0], period[1], f"candidate.{key}"
        return None

    def _failed_result(
        self,
        config: CandidateRevalidationConfig,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        reason: str,
    ) -> CandidateRevalidationResult:
        decision = {
            "recommended_action": "continue_monitoring",
            "action_reason": reason,
            "risk_level": "HIGH",
            "flags": ["revalidation_failed"],
            "decision_score": 0,
        }
        run_id = self._save_run(
            config,
            candidate,
            baseline,
            {},
            {},
            decision,
            config.review_start_date,
            config.review_end_date,
            status="failed",
        )
        return CandidateRevalidationResult(
            revalidation_id=run_id,
            candidate_id=config.candidate_id,
            status="failed",
            mode=config.mode,
            baseline_metrics=baseline,
            latest_metrics={},
            comparison={},
            decision=decision,
            recommended_action=decision["recommended_action"],
            action_reason=reason,
        )

    def _save_run(
        self,
        config: CandidateRevalidationConfig,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        latest: dict[str, Any],
        comparison: dict[str, Any],
        decision: dict[str, Any],
        start_date: str | None,
        end_date: str | None,
        status: str,
        generated_experiment_id: str | None = None,
        window: "ReviewWindow | None" = None,
    ) -> str:
        config_payload = config.to_dict()
        if window is not None:
            config_payload["resolved_review_window"] = window.to_dict()
        return self.candidate_store.create_revalidation_run(
            candidate_id=config.candidate_id,
            strategy_name=candidate.get("strategy_name", ""),
            candidate_name=candidate.get("candidate_name", ""),
            status=status,
            mode=config.mode,
            review_start_date=start_date,
            review_end_date=end_date,
            baseline_metrics=baseline,
            latest_metrics=latest,
            comparison=comparison,
            decision=decision,
            recommended_action=decision["recommended_action"],
            action_reason=decision["action_reason"],
            old_candidate_status=candidate.get("status"),
            new_candidate_status=candidate.get("status"),
            source_research_run_id=candidate.get("source_research_run_id"),
            source_experiment_id=candidate.get("source_experiment_id"),
            generated_experiment_id=generated_experiment_id,
            config=config_payload,
            notes=config.notes,
            tags=config.tags,
            git_commit=self.git_commit_loader(),
        )


@dataclass(frozen=True)
class ReviewWindow:
    start_date: str | None
    end_date: str | None
    source: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def determine_review_window(
    candidate: dict[str, Any],
    config: CandidateRevalidationConfig,
    latest_date: str | pd.Timestamp | None,
    source_period: tuple[str | None, str | None, str] | None = None,
) -> ReviewWindow:
    warnings: list[str] = []
    latest = pd.to_datetime(latest_date, errors="coerce")
    if pd.isna(latest):
        return ReviewWindow(None, None, "latest_date_unavailable", ["latest local data date is unavailable"])

    user_end = pd.to_datetime(config.review_end_date, errors="coerce") if config.review_end_date else None
    end = user_end if user_end is not None and not pd.isna(user_end) else latest
    source = "user" if user_end is not None and not pd.isna(user_end) else "latest_date"
    if end > latest:
        end = latest
        warnings.append("review_end_date clipped to latest available local data date")

    if config.mode == "recent_window":
        start = pd.to_datetime(config.review_start_date, errors="coerce") if config.review_start_date else end - pd.DateOffset(months=int(config.recent_months or 6))
        return ReviewWindow(_date_iso(start), _date_iso(end), "user" if config.review_start_date or config.review_end_date else "recent_window", warnings)

    if config.mode == "full_retest":
        source_start = source_end = source_name = None
        if source_period:
            source_start, source_end, source_name = source_period
        start = pd.to_datetime(config.review_start_date, errors="coerce") if config.review_start_date else pd.to_datetime(source_start, errors="coerce")
        if config.review_end_date is None and source_end:
            end = pd.to_datetime(source_end, errors="coerce")
            if end > latest:
                end = latest
                warnings.append("source end_date clipped to latest available local data date")
        if pd.isna(start):
            start = pd.to_datetime(candidate.get("created_at"), errors="coerce")
            warnings.append("full_retest source period unavailable; fallback to candidate.created_at")
            source_name = "fallback_candidate_created_at"
        resolved_source = "user" if config.review_start_date or config.review_end_date else source_name or source
        return ReviewWindow(_date_iso(start), _date_iso(end), resolved_source, warnings)

    if config.mode == "forward_only":
        created = pd.to_datetime(candidate.get("created_at"), errors="coerce")
        start = pd.to_datetime(config.review_start_date, errors="coerce") if config.review_start_date else created
        if pd.isna(start):
            return ReviewWindow(None, _date_iso(end), "forward_only", warnings)
        if not pd.isna(created) and start < created:
            start = created
        return ReviewWindow(_date_iso(start), _date_iso(end), "forward_only", warnings)
    raise ValueError(f"Unknown revalidation mode: {config.mode}")


def build_candidate_comparison(baseline: dict[str, Any], latest: dict[str, Any]) -> CandidateComparison:
    return CandidateComparison(
        baseline_net_CAGR=_number(baseline.get("net_CAGR")),
        latest_net_CAGR=_number(latest.get("net_CAGR")),
        net_CAGR_delta=_delta(latest.get("net_CAGR"), baseline.get("net_CAGR")),
        baseline_max_drawdown=_number(baseline.get("max_drawdown")),
        latest_max_drawdown=_number(latest.get("max_drawdown")),
        max_drawdown_delta=_delta(latest.get("max_drawdown"), baseline.get("max_drawdown")),
        baseline_sharpe=_number(baseline.get("sharpe_ratio")),
        latest_sharpe=_number(latest.get("sharpe_ratio")),
        sharpe_delta=_delta(latest.get("sharpe_ratio"), baseline.get("sharpe_ratio")),
        baseline_stability_score=_number(baseline.get("stability_score")),
        latest_stability_score=_number(latest.get("stability_score")),
        stability_score_delta=_delta(latest.get("stability_score"), baseline.get("stability_score")),
        baseline_overfit_risk=baseline.get("overfit_risk"),
        latest_overfit_risk=latest.get("overfit_risk"),
        trade_count_delta=_delta(latest.get("total_trades"), baseline.get("total_trades")),
        cost_impact_delta=_delta(latest.get("cost_impact"), baseline.get("cost_impact")),
    )


def latest_local_data_date() -> str | None:
    try:
        status = get_database_status()
        return status.get("max_date")
    except Exception:
        return pd.Timestamp.today().date().isoformat()


def current_git_commit(project_root: str | Path | None = None) -> str:
    root = Path(project_root or Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _attach_window_warnings(decision: dict[str, Any], window: ReviewWindow, invalid_period: bool = False) -> None:
    warnings = list(window.warnings)
    if invalid_period:
        warnings.append("review window is empty or invalid; start_date must be before end_date")
        reason = "Review window is empty or invalid; start_date must be before end_date."
        if window.warnings:
            reason = f"{reason} {window.warnings[0]}"
        decision["action_reason"] = reason
        flags = list(decision.get("flags", []) or [])
        if "invalid_review_window" not in flags:
            flags.append("invalid_review_window")
        decision["flags"] = flags
    if warnings:
        decision["warnings"] = warnings


def _period_from_mapping(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    key_pairs = [
        ("start_date", "end_date"),
        ("research_start_date", "research_end_date"),
        ("data_start_date", "data_end_date"),
        ("review_start_date", "review_end_date"),
    ]
    for start_key, end_key in key_pairs:
        start = _date_iso(pd.to_datetime(value.get(start_key), errors="coerce"))
        end = _date_iso(pd.to_datetime(value.get(end_key), errors="coerce"))
        if start and end:
            return start, end
    for item in value.values():
        nested = _period_from_mapping(item)
        if nested:
            return nested
    return None


def _date_iso(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _baseline_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics_json", {}) or {})
    scorecard = dict(candidate.get("scorecard_json", {}) or {})
    metrics.setdefault("stability_score", candidate.get("stability_score") or scorecard.get("overall_score"))
    metrics.setdefault("overall_score", scorecard.get("overall_score"))
    metrics.setdefault("overfit_risk", candidate.get("overfit_risk"))
    return metrics


def _latest_metrics(result: dict[str, Any]) -> dict[str, Any]:
    out_sample = dict(result.get("out_of_sample_result", {}) or {})
    walk = dict(dict(result.get("walk_forward_result", {}) or {}).get("summary", {}) or {})
    sensitivity = dict(result.get("sensitivity_result", {}) or {})
    stability = dict(result.get("stability_score", {}) or {})
    overfit = dict(result.get("overfit_result", {}) or {})
    return {
        "net_CAGR": out_sample.get("net_CAGR"),
        "max_drawdown": out_sample.get("max_drawdown"),
        "sharpe_ratio": out_sample.get("sharpe_ratio"),
        "win_rate": out_sample.get("win_rate"),
        "total_trades": out_sample.get("total_trades", 0),
        "walk_forward_positive_ratio": walk.get("positive_test_fold_ratio"),
        "parameter_stability_score": sensitivity.get("sensitivity_score"),
        "regime_stability_score": stability.get("regime_stability_score"),
        "stability_score": stability.get("stability_score"),
        "overfit_risk": overfit.get("overfit_risk"),
        "cost_impact": out_sample.get("cost_impact"),
    }


def _cost_config(value: dict[str, Any] | TradingCostConfig | None) -> TradingCostConfig:
    if isinstance(value, TradingCostConfig):
        return value
    return TradingCostConfig(**dict(value or {}))


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(latest: Any, baseline: Any) -> float | None:
    latest_value = _number(latest)
    baseline_value = _number(baseline)
    if latest_value is None or baseline_value is None:
        return None
    return latest_value - baseline_value
