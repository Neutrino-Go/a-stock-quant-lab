from __future__ import annotations

import pandas as pd

from candidate.candidate_ui_helpers import (
    action_to_candidate_status,
    candidate_option_label,
    format_revalidation_mode,
    revalidation_summary_metrics,
    status_change_confirmation_matches,
)


def test_action_to_candidate_status_maps_safe_actions():
    assert action_to_candidate_status("keep_active", "watching") == "active"
    assert action_to_candidate_status("downgrade_to_watching", "active") == "watching"
    assert action_to_candidate_status("reject", "active") == "rejected"
    assert action_to_candidate_status("archive", "active") == "archived"
    assert action_to_candidate_status("needs_more_data", "active") == "active"


def test_status_change_confirmation_matches_candidate_name_exactly():
    assert status_change_confirmation_matches("Trend candidate", " Trend candidate ")
    assert not status_change_confirmation_matches("Trend candidate", "trend candidate")
    assert not status_change_confirmation_matches("Trend candidate", "")


def test_candidate_option_label_includes_name_strategy_and_status():
    label = candidate_option_label(
        {
            "candidate_name": "Trend candidate",
            "strategy_name": "trend_strength_strategy",
            "status": "active",
            "id": "abcdef123456",
        }
    )

    assert "Trend candidate" in label
    assert "trend_strength_strategy" in label
    assert "abcdef12" in label


def test_revalidation_summary_metrics_handles_empty_history():
    summary = revalidation_summary_metrics(pd.DataFrame())

    assert summary["total_revalidations"] == 0
    assert summary["last_revalidation_at"] == "-"


def test_format_revalidation_mode_has_human_readable_labels():
    assert "新增样本" in format_revalidation_mode("forward_only")
    assert "重新测试" in format_revalidation_mode("full_retest")
