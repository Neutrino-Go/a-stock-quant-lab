from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from candidate.candidate_store import CandidateStore
from workflow.research_store import ResearchStore


def _stores(tmp_path):
    db_path = tmp_path / "candidate.db"
    return ResearchStore(db_path), CandidateStore(db_path)


def _candidate(research_store: ResearchStore, **overrides) -> str:
    payload = {
        "strategy_name": "trend_strength_strategy",
        "candidate_name": "Trend candidate",
        "params": {"return_20d": 0.1},
        "scorecard": {"overall_score": 82, "final_rating": "A"},
        "metrics": {"net_CAGR": 0.15, "max_drawdown": -0.12, "sharpe_ratio": 1.2, "total_trades": 80},
        "stability_score": 82,
        "overfit_risk": "LOW",
        "status": "active",
        "tags": ["alpha"],
    }
    payload.update(overrides)
    return research_store.create_strategy_candidate(**payload)


def test_revalidation_run_can_be_created_and_queried(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store)

    run_id = store.create_revalidation_run(
        candidate_id=candidate_id,
        strategy_name="trend_strength_strategy",
        candidate_name="Trend candidate",
        status="completed",
        mode="recent_window",
        review_start_date="2024-01-01",
        review_end_date="2024-12-31",
        baseline_metrics={"net_CAGR": 0.15},
        latest_metrics={"net_CAGR": 0.10, "max_drawdown": -0.11, "overfit_risk": "LOW"},
        comparison={"net_CAGR_delta": -0.05},
        decision={"recommended_action": "keep_active"},
        recommended_action="keep_active",
        action_reason="Latest metrics remain acceptable for research monitoring.",
        tags=["recheck"],
    )
    row = store.get_revalidation_run(run_id)
    listed = store.list_revalidation_runs(candidate_id=candidate_id, mode="recent_window")

    assert row["candidate_id"] == candidate_id
    assert row["baseline_metrics_json"]["net_CAGR"] == 0.15
    assert listed.iloc[0]["latest_net_CAGR"] == 0.10


def test_metric_snapshot_and_lifecycle_event_are_saved(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    run_id = store.create_revalidation_run(candidate_id, "trend_strength_strategy", "Trend candidate")

    store.save_metric_snapshot(
        candidate_id=candidate_id,
        revalidation_run_id=run_id,
        snapshot_date="2026-06-03",
        data_start_date="2026-01-01",
        data_end_date="2026-06-01",
        metrics={"net_CAGR": 0.09, "max_drawdown": -0.08, "overall_score": 76, "overfit_risk": "LOW"},
    )
    store.create_lifecycle_event(candidate_id, "revalidated", "active", "active", "Recent window checked.")

    snapshots = store.list_metric_snapshots(candidate_id)
    events = store.list_lifecycle_events(candidate_id)

    assert snapshots.iloc[0]["net_CAGR"] == 0.09
    assert events.iloc[0]["event_type"] == "revalidated"
    assert events.iloc[0]["candidate_name"] == "Trend candidate"


def test_apply_candidate_status_change_records_lifecycle_event(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store, status="active")

    assert store.apply_candidate_status_change(candidate_id, "watching", "Risk increased.") is True
    candidate = research_store.get_strategy_candidate(candidate_id)
    events = store.list_lifecycle_events(candidate_id)

    assert candidate["status"] == "watching"
    assert events.iloc[0]["event_type"] == "status_changed"
    assert events.iloc[0]["old_status"] == "active"
    assert events.iloc[0]["new_status"] == "watching"


def test_list_candidates_due_for_review_returns_missing_or_stale_revalidation(tmp_path):
    research_store, store = _stores(tmp_path)
    stale_id = _candidate(research_store, candidate_name="Stale")
    fresh_id = _candidate(research_store, candidate_name="Fresh")
    run_id = store.create_revalidation_run(fresh_id, "trend_strength_strategy", "Fresh", status="completed")
    old_date = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    recent_date = datetime.now(timezone.utc).isoformat()
    with store._connect() as conn:
        conn.execute("UPDATE candidate_revalidation_runs SET created_at = ? WHERE id = ?", (recent_date, run_id))
        conn.execute("UPDATE strategy_candidates SET created_at = ? WHERE id = ?", (old_date, stale_id))

    due = store.list_candidates_due_for_review(max_age_days=1)

    assert stale_id in set(due["id"])
    assert fresh_id not in set(due["id"])
    assert "days_since_revalidation" in due.columns


def test_export_candidates_csv_uses_utf8_sig_and_handles_empty_frame(tmp_path):
    _, store = _stores(tmp_path)

    path = store.export_candidates_csv(pd.DataFrame(), tmp_path / "candidates.csv")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    exported = pd.read_csv(path)
    assert "candidate_name" in exported.columns


def test_get_candidate_with_latest_revalidation_includes_latest_result(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    store.create_revalidation_run(
        candidate_id,
        "trend_strength_strategy",
        "Trend candidate",
        latest_metrics={"net_CAGR": 0.11, "overfit_risk": "LOW"},
        recommended_action="keep_active",
    )

    detail = store.get_candidate_with_latest_revalidation(candidate_id)

    assert detail["candidate"]["id"] == candidate_id
    assert detail["latest_revalidation"]["recommended_action"] == "keep_active"
