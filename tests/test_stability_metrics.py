from __future__ import annotations

from stability.stability_metrics import calculate_stability_score


def test_stability_score_good_strategy_receives_a_rating():
    result = calculate_stability_score(
        in_sample_metrics={"net_CAGR": 0.18, "sharpe_ratio": 1.5, "max_drawdown": -0.12, "total_trades": 80},
        out_of_sample_metrics={"net_CAGR": 0.15, "sharpe_ratio": 1.3, "max_drawdown": -0.14, "total_trades": 45},
        walk_forward_summary={"positive_test_fold_ratio": 0.8, "walk_forward_score": 82},
        sensitivity_result={"sensitivity_score": 85},
        market_regime_breakdown={"BULL": {"trade_count": 20, "net_CAGR": 0.18}, "BEAR": {"trade_count": 15, "net_CAGR": 0.04}},
        overfit_result={"overfit_risk": "LOW"},
    )

    assert 0 <= result["stability_score"] <= 100
    assert result["final_rating"] == "A"


def test_stability_score_high_overfit_forces_d_rating():
    result = calculate_stability_score(
        in_sample_metrics={"net_CAGR": 0.30, "sharpe_ratio": 2.0, "max_drawdown": -0.05, "total_trades": 30},
        out_of_sample_metrics={"net_CAGR": -0.05, "sharpe_ratio": 0.1, "max_drawdown": -0.35, "total_trades": 8},
        walk_forward_summary={"positive_test_fold_ratio": 0.2, "walk_forward_score": 20},
        sensitivity_result={"sensitivity_score": 20},
        market_regime_breakdown={},
        overfit_result={"overfit_risk": "HIGH"},
    )

    assert result["final_rating"] == "D"
    assert result["stability_score"] < 50


def test_stability_score_handles_missing_or_sparse_regime_data():
    result = calculate_stability_score(
        in_sample_metrics={},
        out_of_sample_metrics={"net_CAGR": 0.0, "total_trades": 0},
        walk_forward_summary={},
        sensitivity_result={},
        market_regime_breakdown={"BULL": {"trade_count": 1, "sample_note": "insufficient_sample"}},
        overfit_result={"overfit_risk": "MEDIUM"},
    )

    assert 0 <= result["regime_stability_score"] <= 100
    assert result["final_rating"] in {"C", "D"}


def test_stability_score_drawdown_direction_treats_less_negative_as_better():
    result = calculate_stability_score(
        in_sample_metrics={"net_CAGR": 0.10, "max_drawdown": -0.30, "total_trades": 60},
        out_of_sample_metrics={"net_CAGR": 0.09, "max_drawdown": -0.10, "total_trades": 40},
        walk_forward_summary={"positive_test_fold_ratio": 0.7, "walk_forward_score": 70},
        sensitivity_result={"sensitivity_score": 70},
        market_regime_breakdown={},
        overfit_result={"overfit_risk": "LOW"},
    )

    assert result["generalization_score"] >= 70
