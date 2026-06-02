from __future__ import annotations

from candidate.candidate_models import CandidateComparison, CandidateRevalidationConfig, CandidateRevalidationResult


def test_candidate_revalidation_config_round_trips():
    config = CandidateRevalidationConfig(
        candidate_id="candidate-1",
        mode="forward_only",
        recent_months=9,
        tags=["alpha"],
        save_as_experiment=True,
    )

    restored = CandidateRevalidationConfig.from_dict(config.to_dict())

    assert restored.candidate_id == "candidate-1"
    assert restored.mode == "forward_only"
    assert restored.tags == ["alpha"]
    assert restored.save_as_experiment is True


def test_candidate_comparison_allows_missing_values():
    comparison = CandidateComparison.from_dict({"latest_net_CAGR": 0.12})

    assert comparison.latest_net_CAGR == 0.12
    assert comparison.baseline_net_CAGR is None
    assert comparison.to_dict()["net_CAGR_delta"] is None


def test_candidate_revalidation_result_round_trips():
    result = CandidateRevalidationResult(
        revalidation_id="run-1",
        candidate_id="candidate-1",
        status="completed",
        mode="recent_window",
        recommended_action="keep_active",
        latest_metrics={"net_CAGR": 0.1},
    )

    restored = CandidateRevalidationResult.from_dict(result.to_dict())

    assert restored.revalidation_id == "run-1"
    assert restored.latest_metrics["net_CAGR"] == 0.1
    assert restored.recommended_action == "keep_active"


def test_candidate_revalidation_config_from_empty_dict_is_safe():
    config = CandidateRevalidationConfig.from_dict({})

    assert config.candidate_id == ""
    assert config.mode == "recent_window"
    assert config.tags == []
    assert config.cost_config == {}
