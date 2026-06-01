from __future__ import annotations

import pandas as pd

from workflow.research_models import ResearchConfig
from workflow.research_store import ResearchStore
from workflow.research_workflow import StrategyResearchWorkflow


class FakeStabilityService:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def run_stability_analysis(self, **kwargs):
        # workflow passes a single-value grid per candidate.
        flat_params = {key: values[0] for key, values in kwargs["parameter_grid"].items()}
        score = 0.2 if flat_params.get("return_20d") == 0.1 else 0.05
        self.calls.append(flat_params)
        return {
            "analysis_id": f"exp-{len(self.calls)}",
            "strategy_name": kwargs["strategy_name"],
            "best_params": flat_params,
            "out_of_sample_result": {
                "net_CAGR": score,
                "max_drawdown": -0.1,
                "sharpe_ratio": 1.5,
                "calmar_ratio": 2.0,
                "expectancy": 0.01,
                "total_trades": 80,
                "cost_impact": 0.01,
            },
            "walk_forward_result": {"summary": {"positive_test_fold_ratio": 0.7, "avg_test_net_CAGR": score, "fold_count": 4}},
            "sensitivity_result": {"sensitivity_score": 85},
            "stability_score": {"stability_score": 86, "final_rating": "A"},
            "overfit_result": {"overfit_risk": "LOW", "overfit_score": 20, "risk_flags": []},
            "market_regime_breakdown": {
                "BULL": {"trade_count": 30, "net_CAGR": 0.2},
                "BEAR": {"trade_count": 20, "net_CAGR": 0.03},
                "SIDEWAYS": {"trade_count": 20, "net_CAGR": 0.08},
                "HIGH_VOLATILITY": {"trade_count": 10, "net_CAGR": 0.01},
            },
        }


def test_research_workflow_runs_full_pipeline_and_promotes_candidate(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    stability = FakeStabilityService()
    workflow = StrategyResearchWorkflow(
        research_store=store,
        stability_service=stability,
        factor_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(
            {"date": pd.date_range("2021-01-01", periods=260, freq="B"), "code": "000001", "close": 10.0}
        ),
        report_output_dir=tmp_path / "reports",
    )
    config = ResearchConfig(
        strategy_name="trend_strength_strategy",
        start_date="2021-01-01",
        end_date="2023-01-01",
        research_template="quick",
        parameter_grid={"return_20d": [0.05, 0.1, 0.15]},
        top_n_params=2,
        train_ratio=0.7,
        walk_forward_config={"train_months": 18, "test_months": 6, "step_months": 6},
        tags=["workflow"],
        notes="run synchronously",
    )

    result = workflow.run(config)
    detail = store.get_research_run_detail(result.research_run_id)
    candidates = store.list_strategy_candidates()
    report_paths = detail["artifacts"].loc[detail["artifacts"]["artifact_type"].eq("markdown_report"), "file_path"].tolist()

    assert result.status == "completed"
    assert result.decision == "add_to_candidates"
    assert result.candidate_id is not None
    assert len(stability.calls) == 2
    assert set(detail["steps"]["step_name"]) >= {
        "data_check",
        "parameter_optimization",
        "stability_analysis",
        "scorecard_generation",
        "candidate_promotion",
        "report_export",
    }
    assert candidates.iloc[0]["source_research_run_id"] == result.research_run_id
    assert detail["run"]["summary_json"]["scorecard"]["decision"] == "add_to_candidates"
    assert report_paths
    assert "add_to_candidates" in open(report_paths[0], encoding="utf-8").read()


def test_research_workflow_records_failed_status_when_data_check_fails(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    workflow = StrategyResearchWorkflow(
        research_store=store,
        stability_service=FakeStabilityService(),
        factor_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(),
        report_output_dir=tmp_path / "reports",
    )
    config = ResearchConfig(
        strategy_name="trend_strength_strategy",
        start_date="2021-01-01",
        end_date="2021-02-01",
        research_template="quick",
        parameter_grid={"return_20d": [0.05]},
    )

    result = workflow.run(config)
    detail = store.get_research_run_detail(result.research_run_id)

    assert result.status == "failed"
    assert result.decision == "failed"
    assert detail["run"]["status"] == "failed"
    assert detail["steps"].iloc[-1]["status"] == "failed"
