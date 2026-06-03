from __future__ import annotations

import pandas as pd

from candidate.candidate_models import CandidateRevalidationConfig
from candidate.candidate_revalidator import CandidateRevalidator
from candidate.candidate_store import CandidateStore
from candidate.candidate_ui_helpers import run_batch_revalidation
from research.experiment_service import ExperimentService
from research.experiment_store import ExperimentStore
from workflow.research_store import ResearchStore


class FakeStabilityService:
    def __init__(self, latest_net_cagr: float = 0.10):
        self.latest_net_cagr = latest_net_cagr
        self.calls: list[dict[str, object]] = []

    def run_stability_analysis(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "analysis_id": "stability-exp-1" if kwargs.get("save_as_experiment") else "analysis-1",
            "out_of_sample_result": {
                "net_CAGR": self.latest_net_cagr,
                "max_drawdown": -0.12,
                "sharpe_ratio": 1.1,
                "total_trades": 48,
                "cost_impact": 0.01,
            },
            "walk_forward_result": {"summary": {"positive_test_fold_ratio": 0.60}},
            "sensitivity_result": {"sensitivity_score": 74},
            "stability_score": {"stability_score": 75, "regime_stability_score": 68},
            "overfit_result": {"overfit_risk": "LOW", "overfit_score": 25, "risk_flags": []},
        }


class FakeExperimentService:
    def __init__(self):
        self.saved: list[dict[str, object]] = []

    def save_candidate_revalidation_result(self, **kwargs):
        self.saved.append(kwargs)
        return "candidate-revalidation-exp"


def _stores(tmp_path):
    db_path = tmp_path / "candidate.db"
    return ResearchStore(db_path), CandidateStore(db_path)


def _candidate(research_store: ResearchStore, **overrides) -> str:
    payload = {
        "strategy_name": "trend_strength_strategy",
        "candidate_name": "Trend candidate",
        "params": {"return_20d": 0.1},
        "scorecard": {"overall_score": 80},
        "metrics": {
            "net_CAGR": 0.15,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.4,
            "total_trades": 80,
            "walk_forward_positive_ratio": 0.70,
            "parameter_stability_score": 82,
            "overfit_risk": "LOW",
        },
        "stability_score": 80,
        "overfit_risk": "LOW",
        "status": "active",
    }
    payload.update(overrides)
    return research_store.create_strategy_candidate(**payload)


def _revalidator(research_store, candidate_store, stability=None, experiment=None, latest="2026-06-01"):
    return CandidateRevalidator(
        research_store=research_store,
        candidate_store=candidate_store,
        stability_service=stability or FakeStabilityService(),
        experiment_service=experiment or FakeExperimentService(),
        latest_date_loader=lambda: latest,
        git_commit_loader=lambda: "abc123",
    )


def test_missing_params_records_failed_revalidation(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store, params={})

    result = _revalidator(research_store, candidate_store).run_revalidation(
        CandidateRevalidationConfig(candidate_id=candidate_id, mode="full_retest")
    )

    assert result.status == "failed"
    assert "params" in result.action_reason
    assert candidate_store.list_revalidation_runs(candidate_id=candidate_id).iloc[0]["status"] == "failed"


def test_no_new_forward_data_returns_needs_more_data(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    with candidate_store._connect() as conn:
        conn.execute("UPDATE strategy_candidates SET created_at = ? WHERE id = ?", ("2026-06-03T00:00:00+00:00", candidate_id))

    result = _revalidator(research_store, candidate_store, latest="2026-06-01").run_revalidation(
        CandidateRevalidationConfig(candidate_id=candidate_id, mode="forward_only")
    )

    assert result.status == "completed"
    assert result.recommended_action == "needs_more_data"
    assert result.latest_metrics["total_trades"] == 0


def test_forward_only_does_not_use_candidate_creation_before_date(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    with candidate_store._connect() as conn:
        conn.execute("UPDATE strategy_candidates SET created_at = ? WHERE id = ?", ("2026-01-15T00:00:00+00:00", candidate_id))
    stability = FakeStabilityService()

    _revalidator(research_store, candidate_store, stability=stability).run_revalidation(
        CandidateRevalidationConfig(candidate_id=candidate_id, mode="forward_only")
    )

    assert pd.to_datetime(stability.calls[0]["start_date"]) >= pd.Timestamp("2026-01-15")


def test_recent_window_date_range_uses_recent_months(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    stability = FakeStabilityService()

    _revalidator(research_store, candidate_store, stability=stability).run_revalidation(
        CandidateRevalidationConfig(candidate_id=candidate_id, mode="recent_window", recent_months=3)
    )

    assert str(stability.calls[0]["start_date"]).startswith("2026-03")
    assert str(stability.calls[0]["end_date"]).startswith("2026-06-01")


def test_full_retest_mode_runs_with_user_dates(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    stability = FakeStabilityService()

    result = _revalidator(research_store, candidate_store, stability=stability).run_revalidation(
        CandidateRevalidationConfig(
            candidate_id=candidate_id,
            mode="full_retest",
            review_start_date="2020-01-01",
            review_end_date="2024-12-31",
        )
    )

    assert result.status == "completed"
    assert stability.calls[0]["start_date"] == "2020-01-01"
    assert stability.calls[0]["end_date"] == "2024-12-31"


def test_candidate_revalidation_can_be_saved_as_experiment(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    experiment = FakeExperimentService()

    result = _revalidator(research_store, candidate_store, experiment=experiment).run_revalidation(
        CandidateRevalidationConfig(candidate_id=candidate_id, mode="recent_window", save_as_experiment=True)
    )

    assert result.generated_experiment_id == "candidate-revalidation-exp"
    assert experiment.saved[0]["strategy_name"] == "trend_strength_strategy"
    assert experiment.saved[0]["result"]["revalidation_id"] == result.revalidation_id


def test_candidate_revalidation_experiment_detail_can_be_read(tmp_path):
    db_path = tmp_path / "candidate.db"
    research_store = ResearchStore(db_path)
    candidate_store = CandidateStore(db_path)
    experiment_service = ExperimentService(ExperimentStore(db_path))
    candidate_id = _candidate(research_store)

    result = CandidateRevalidator(
        research_store=research_store,
        candidate_store=candidate_store,
        stability_service=FakeStabilityService(),
        experiment_service=experiment_service,
        latest_date_loader=lambda: "2026-06-01",
        git_commit_loader=lambda: "abc123",
    ).run_revalidation(CandidateRevalidationConfig(candidate_id=candidate_id, mode="recent_window", save_as_experiment=True))

    detail = experiment_service.build_experiment_detail(result.generated_experiment_id)

    assert detail["run"]["experiment_type"] == "candidate_revalidation"
    assert detail["run"]["metrics_json"]["candidate_id"] == candidate_id
    assert detail["run"]["metrics_json"]["revalidation_run_id"] == result.revalidation_id


def test_batch_revalidation_continues_after_candidate_failure(tmp_path):
    research_store, candidate_store = _stores(tmp_path)
    good_id = _candidate(research_store, candidate_name="Good")
    bad_id = _candidate(research_store, candidate_name="Bad", params={})
    revalidator = _revalidator(research_store, candidate_store)

    results = run_batch_revalidation(
        revalidator,
        [
            CandidateRevalidationConfig(candidate_id=bad_id, mode="full_retest"),
            CandidateRevalidationConfig(candidate_id=good_id, mode="recent_window"),
        ],
    )

    assert len(results) == 2
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "completed"
