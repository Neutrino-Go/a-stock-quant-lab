from __future__ import annotations

import json

import pandas as pd

from workflow.research_report import export_research_json_summary, export_research_markdown_report
from workflow.research_store import ResearchStore


def test_export_research_markdown_report_records_artifact_and_avoids_forbidden_language(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    run_id = store.create_research_run(
        name="Trend research",
        strategy_name="trend_strength_strategy",
        start_date="2021-01-01",
        end_date="2023-01-01",
        research_template="standard",
        git_commit="abc123",
        summary={
            "scorecard": {
                "overall_score": 82,
                "final_rating": "A",
                "decision": "add_to_candidates",
                "decision_reason": "Historical research result is stable enough for candidate tracking.",
                "best_params": {"return_20d": 0.1},
                "best_experiment_id": "exp-1",
                "sub_scores": {"return_score": 80, "risk_score": 75},
                "flags": ["sample_checked"],
            },
            "core_metrics": {
                "out_of_sample_net_CAGR": 0.18,
                "max_drawdown": -0.12,
                "sharpe_ratio": 1.4,
                "walk_forward_positive_ratio": 0.7,
                "parameter_stability_score": 82,
                "overfit_risk": "LOW",
                "cost_impact": 0.01,
            },
            "candidate_id": "candidate-1",
        },
        decision="add_to_candidates",
        decision_reason="Historical research result is stable enough for candidate tracking.",
    )
    store.save_research_step(run_id, "data_check", "completed", result={"daily_rows": 100})

    path = export_research_markdown_report(store, run_id, output_dir=tmp_path / "reports")
    content = path.read_text(encoding="utf-8")
    detail = store.get_research_run_detail(run_id)

    assert "# 策略研究报告" in content
    assert "## 最终评分卡" in content
    assert "历史回测不代表未来收益" in content
    assert "建议买入" not in content
    assert "保证盈利" not in content
    assert detail["artifacts"].iloc[0]["artifact_type"] == "markdown_report"


def test_export_research_json_summary_is_deserializable(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    run_id = store.create_research_run(
        name="Research",
        strategy_name="trend_strength_strategy",
        summary={"scorecard": {"overall_score": 60}},
    )

    path = export_research_json_summary(store, run_id, output_dir=tmp_path / "reports")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run"]["id"] == run_id
    assert payload["run"]["summary_json"]["scorecard"]["overall_score"] == 60


def test_export_candidates_csv_handles_empty_candidates(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    run_id = store.create_research_run(name="Research", strategy_name="trend_strength_strategy")

    from workflow.research_report import export_research_candidates_csv

    path = export_research_candidates_csv(store, run_id, pd.DataFrame(), output_dir=tmp_path / "reports")
    frame = pd.read_csv(path)

    assert frame.empty
    assert "candidate_name" in frame.columns
