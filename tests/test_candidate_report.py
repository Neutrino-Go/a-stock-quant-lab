from __future__ import annotations

import pandas as pd

from candidate.candidate_report import (
    export_lifecycle_events_csv,
    export_revalidation_history_csv,
    export_revalidation_markdown_report,
)
from candidate.candidate_store import CandidateStore
from workflow.research_store import ResearchStore


def _stores(tmp_path):
    db_path = tmp_path / "candidate.db"
    return ResearchStore(db_path), CandidateStore(db_path)


def _candidate(research_store: ResearchStore) -> str:
    return research_store.create_strategy_candidate(
        strategy_name="trend_strength_strategy",
        candidate_name="Trend candidate",
        params={"return_20d": 0.1},
        scorecard={"overall_score": 82},
        metrics={"net_CAGR": 0.15, "max_drawdown": -0.12},
        status="active",
    )


def test_revalidation_markdown_report_contains_risk_notice(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    run_id = store.create_revalidation_run(
        candidate_id,
        "trend_strength_strategy",
        "Trend candidate",
        status="completed",
        mode="recent_window",
        baseline_metrics={"net_CAGR": 0.15},
        latest_metrics={"net_CAGR": 0.10, "max_drawdown": -0.10},
        comparison={"net_CAGR_delta": -0.05},
        decision={"recommended_action": "keep_active", "risk_level": "LOW", "flags": []},
        recommended_action="keep_active",
        action_reason="Continue monitoring in research workflow.",
    )

    path = export_revalidation_markdown_report(store, run_id, output_dir=tmp_path / "reports")
    content = path.read_text(encoding="utf-8")
    refreshed = store.get_revalidation_run(run_id)

    assert "# 候选策略复检报告" in content
    assert "历史表现不代表未来收益" in content
    assert "候选策略不是买入建议" in content
    assert "建议买入" not in content
    assert refreshed["report_path"] == str(path)


def test_revalidation_history_csv_uses_utf8_sig_and_empty_headers(tmp_path):
    _, store = _stores(tmp_path)

    path = export_revalidation_history_csv(pd.DataFrame(), tmp_path / "history.csv")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "candidate_name" in pd.read_csv(path).columns


def test_lifecycle_events_csv_uses_utf8_sig(tmp_path):
    research_store, store = _stores(tmp_path)
    candidate_id = _candidate(research_store)
    store.create_lifecycle_event(candidate_id, "note_updated", "active", "active", "Manual note.")

    path = export_lifecycle_events_csv(store.list_lifecycle_events(candidate_id), tmp_path / "events.csv")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert pd.read_csv(path).iloc[0]["event_type"] == "note_updated"
