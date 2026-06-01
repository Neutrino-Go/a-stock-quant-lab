from __future__ import annotations

from stability.overfit_detector import OverfitDetector


def test_overfit_detector_flags_high_risk_when_oos_collapses():
    result = OverfitDetector().detect(
        in_sample_metrics={"net_CAGR": 0.35, "sharpe_ratio": 2.0, "max_drawdown": -0.08},
        out_of_sample_metrics={"net_CAGR": -0.05, "sharpe_ratio": 0.2, "max_drawdown": -0.30},
        walk_forward_summary={"positive_test_fold_ratio": 0.25},
        sensitivity_result={"sensitivity_score": 25},
        trade_count=12,
        parameter_count=8,
    )

    assert result["overfit_risk"] == "HIGH"
    assert result["overfit_score"] >= 70
    assert "out_of_sample_non_positive" in result["risk_flags"]
    assert "low_trade_count" in result["risk_flags"]


def test_overfit_detector_flags_low_risk_for_consistent_results():
    result = OverfitDetector().detect(
        in_sample_metrics={"net_CAGR": 0.16, "sharpe_ratio": 1.4, "max_drawdown": -0.12},
        out_of_sample_metrics={"net_CAGR": 0.13, "sharpe_ratio": 1.2, "max_drawdown": -0.14},
        walk_forward_summary={"positive_test_fold_ratio": 0.8},
        sensitivity_result={"sensitivity_score": 82},
        trade_count=90,
        parameter_count=4,
    )

    assert result["overfit_risk"] == "LOW"
    assert result["overfit_score"] < 35
    assert result["recommendation"] == "continue_research"


def test_overfit_detector_trade_sample_and_parameter_count_flags():
    result = OverfitDetector().detect(
        in_sample_metrics={"net_CAGR": 0.10, "sharpe_ratio": 1.0},
        out_of_sample_metrics={"net_CAGR": 0.08, "sharpe_ratio": 0.9},
        walk_forward_summary={"positive_test_fold_ratio": 0.7},
        sensitivity_result={"sensitivity_score": 70},
        trade_count=10,
        parameter_count=20,
    )

    assert "low_trade_count" in result["risk_flags"]
    assert "too_many_parameters_for_sample" in result["risk_flags"]


def test_overfit_detector_cost_impact_turns_oos_negative_flag():
    result = OverfitDetector().detect(
        in_sample_metrics={"net_CAGR": 0.12, "sharpe_ratio": 1.0},
        out_of_sample_metrics={"gross_CAGR": 0.02, "net_CAGR": -0.01, "cost_impact": 0.03},
        walk_forward_summary={"positive_test_fold_ratio": 0.6},
        sensitivity_result={"sensitivity_score": 75},
        trade_count=40,
        parameter_count=3,
    )

    assert "cost_impact_turns_oos_negative" in result["risk_flags"]
