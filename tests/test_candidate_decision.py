from __future__ import annotations

from candidate.candidate_decision import evaluate_candidate_lifecycle


def test_keep_active_decision_triggers_for_stable_positive_candidate():
    result = evaluate_candidate_lifecycle(
        baseline_metrics={"net_CAGR": 0.14, "max_drawdown": -0.18, "sharpe_ratio": 1.2, "stability_score": 80},
        latest_metrics={
            "net_CAGR": 0.12,
            "max_drawdown": -0.16,
            "sharpe_ratio": 1.1,
            "overfit_risk": "LOW",
            "total_trades": 55,
            "walk_forward_positive_ratio": 0.65,
            "stability_score": 76,
        },
        comparison={"stability_score_delta": -4, "sharpe_delta": -0.1},
        thresholds={"min_trades": 30},
    )

    assert result["recommended_action"] == "keep_active"
    assert result["risk_level"] == "LOW"
    assert 0 <= result["decision_score"] <= 100


def test_downgrade_to_watching_when_stability_deteriorates():
    result = evaluate_candidate_lifecycle(
        baseline_metrics={"net_CAGR": 0.18, "sharpe_ratio": 1.4, "stability_score": 82},
        latest_metrics={
            "net_CAGR": 0.03,
            "max_drawdown": -0.20,
            "sharpe_ratio": 0.5,
            "overfit_risk": "MEDIUM",
            "total_trades": 45,
            "walk_forward_positive_ratio": 0.50,
            "stability_score": 55,
        },
        comparison={"stability_score_delta": -27, "sharpe_delta": -0.9},
        thresholds={"min_trades": 30},
    )

    assert result["recommended_action"] == "downgrade_to_watching"
    assert "stability_deteriorated" in result["flags"]


def test_reject_when_latest_performance_and_risk_fail():
    result = evaluate_candidate_lifecycle(
        baseline_metrics={"net_CAGR": 0.12, "max_drawdown": -0.18},
        latest_metrics={
            "net_CAGR": -0.04,
            "max_drawdown": -0.38,
            "sharpe_ratio": -0.2,
            "overfit_risk": "HIGH",
            "total_trades": 60,
            "walk_forward_positive_ratio": 0.30,
            "stability_score": 40,
        },
        comparison={"stability_score_delta": -35},
        thresholds={"min_trades": 30},
    )

    assert result["recommended_action"] == "reject"
    assert result["risk_level"] == "HIGH"
    assert "overfit_high" in result["flags"]


def test_needs_more_data_when_trade_count_is_too_small():
    result = evaluate_candidate_lifecycle(
        baseline_metrics={"net_CAGR": 0.10},
        latest_metrics={"net_CAGR": 0.08, "max_drawdown": -0.12, "total_trades": 8, "overfit_risk": "LOW"},
        comparison={"stability_score_delta": 0},
        thresholds={"min_trades": 30},
    )

    assert result["recommended_action"] == "needs_more_data"
    assert "insufficient_trades" in result["flags"]


def test_lifecycle_decision_text_avoids_trading_promises():
    result = evaluate_candidate_lifecycle(
        baseline_metrics={},
        latest_metrics={"net_CAGR": 0.01, "max_drawdown": -0.1, "total_trades": 40, "overfit_risk": "LOW"},
        comparison={},
        thresholds={"min_trades": 30},
    )

    forbidden = ["买入", "稳赚", "保证收益", "保证盈利"]
    assert not any(word in result["action_reason"] for word in forbidden)
