from __future__ import annotations

from workflow.research_models import ResearchConfig, ResearchRunResult, ResearchStepResult, StrategyCandidate


def test_research_config_round_trips_to_dict():
    config = ResearchConfig(
        strategy_name="trend_strength_strategy",
        start_date="2021-01-01",
        end_date="2022-01-01",
        research_template="standard",
        parameter_grid={"return_20d": [0.05, 0.10]},
        top_n_params=5,
        train_ratio=0.7,
        walk_forward_config={"train_months": 24, "test_months": 6, "step_months": 6},
        tags=["workflow", "standard"],
        notes="sample note",
    )

    payload = config.to_dict()
    restored = ResearchConfig.from_dict(payload)

    assert payload["start_date"] == "2021-01-01"
    assert restored.strategy_name == "trend_strength_strategy"
    assert restored.parameter_grid["return_20d"] == [0.05, 0.10]
    assert restored.tags == ["workflow", "standard"]


def test_step_result_run_result_and_candidate_are_serializable():
    step = ResearchStepResult(
        step_name="parameter_optimization",
        status="completed",
        message="ok",
        result={"rows": 3},
        experiment_id="exp-1",
    )
    run = ResearchRunResult(
        research_run_id="run-1",
        strategy_name="trend_strength_strategy",
        status="completed",
        summary={"decision": "watch"},
        steps=[step],
        scorecard={"overall_score": 60},
        decision="watch",
        candidate_id=None,
        artifacts=[{"artifact_type": "markdown_report"}],
    )
    candidate = StrategyCandidate(
        id="candidate-1",
        strategy_name="trend_strength_strategy",
        candidate_name="Trend 0.05",
        params={"return_20d": 0.05},
        metrics={"net_CAGR": 0.12},
        scorecard={"final_rating": "B"},
        status="watching",
        source_research_run_id="run-1",
        source_experiment_id="exp-1",
    )

    assert ResearchStepResult.from_dict(step.to_dict()).result["rows"] == 3
    assert ResearchRunResult.from_dict(run.to_dict()).steps[0].step_name == "parameter_optimization"
    assert StrategyCandidate.from_dict(candidate.to_dict()).params["return_20d"] == 0.05
