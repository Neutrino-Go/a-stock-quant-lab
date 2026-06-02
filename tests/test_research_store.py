from __future__ import annotations

from workflow.research_store import ResearchStore


def _store(tmp_path) -> ResearchStore:
    return ResearchStore(tmp_path / "research.db")


def test_create_update_list_and_detail_research_run(tmp_path):
    store = _store(tmp_path)
    run_id = store.create_research_run(
        name="Trend standard research",
        strategy_name="trend_strength_strategy",
        start_date="2021-01-01",
        end_date="2023-01-01",
        research_template="standard",
        config={"top_n_params": 5},
        tags=["trend", "standard"],
        git_commit="abc123",
    )
    store.update_research_status(run_id, "running")
    store.save_research_step(
        run_id,
        "parameter_optimization",
        "completed",
        message="optimized",
        result={"top_n": 5},
        experiment_id="exp-opt",
    )
    store.update_research_summary(
        run_id,
        {"overall_score": 66},
        status="completed",
        decision="continue_research",
        decision_reason="Needs more review.",
    )

    listed = store.list_research_runs(strategy_name="trend_strength_strategy", status="completed", tag="trend")
    detail = store.get_research_run_detail(run_id)

    assert listed.iloc[0]["id"] == run_id
    assert detail["run"]["summary_json"]["overall_score"] == 66
    assert detail["steps"].iloc[0]["experiment_id"] == "exp-opt"
    assert detail["run"]["decision"] == "continue_research"


def test_research_run_filters_by_decision_date_and_keyword(tmp_path):
    store = _store(tmp_path)
    keep_id = store.create_research_run(
        name="Momentum workflow",
        strategy_name="trend_strength_strategy",
        status="completed",
        decision="watch",
        notes="liquidity check",
        tags=["alpha"],
    )
    store.create_research_run(
        name="Other workflow",
        strategy_name="pullback_ma20_strategy",
        status="failed",
        decision="failed",
        notes="bad data",
        tags=["risk"],
    )
    created = store.get_research_run_detail(keep_id)["run"]["created_at"][:10]

    filtered = store.list_research_runs(
        strategy_name="trend_strength_strategy",
        status="completed",
        decision="watch",
        created_start=created,
        created_end=created,
        tag="alpha",
        keyword="liquidity",
    )

    assert list(filtered["id"]) == [keep_id]


def test_delete_research_run_cascades_steps_and_artifacts_but_keeps_candidate(tmp_path):
    store = _store(tmp_path)
    run_id = store.create_research_run(name="Run", strategy_name="trend_strength_strategy")
    store.save_research_step(run_id, "data_check", "completed")
    store.save_research_artifact(run_id, "markdown_report", tmp_path / "report.md")
    candidate_id = store.create_strategy_candidate(
        strategy_name="trend_strength_strategy",
        candidate_name="Candidate",
        source_research_run_id=run_id,
        params={"return_20d": 0.1},
        scorecard={"overall_score": 88},
        metrics={"net_CAGR": 0.2},
        status="active",
    )

    assert store.delete_research_run(run_id) is True
    detail = store.get_research_run_detail(run_id)
    candidate = store.get_strategy_candidate(candidate_id)

    assert detail["run"] is None
    assert candidate["source_research_run_id"] == run_id
    assert store.delete_research_run("missing") is False


def test_strategy_candidate_filters_update_and_archive(tmp_path):
    store = _store(tmp_path)
    candidate_id = store.create_strategy_candidate(
        strategy_name="trend_strength_strategy",
        candidate_name="Candidate A",
        params={"return_20d": 0.1},
        scorecard={"final_rating": "A"},
        metrics={"net_CAGR": 0.2, "max_drawdown": -0.1},
        stability_score=86,
        overfit_risk="LOW",
        tags=["trend"],
        status="active",
    )

    store.update_strategy_candidate(
        candidate_id,
        notes="review monthly",
        status="watching",
        tags=["trend", "edited"],
    )
    filtered = store.list_strategy_candidates(
        strategy_name="trend_strength_strategy",
        status="watching",
        rating="A",
        overfit_risk="LOW",
        tag="edited",
    )
    store.archive_strategy_candidate(candidate_id)
    archived = store.get_strategy_candidate(candidate_id)

    assert filtered.iloc[0]["id"] == candidate_id
    assert filtered.iloc[0]["net_CAGR"] == 0.2
    assert filtered.iloc[0]["notes"] == "review monthly"
    assert filtered.iloc[0]["tags"] == ["trend", "edited"]
    assert filtered.iloc[0]["overall_score"] is None
    assert filtered.iloc[0]["final_rating"] == "A"
    assert archived["status"] == "archived"
