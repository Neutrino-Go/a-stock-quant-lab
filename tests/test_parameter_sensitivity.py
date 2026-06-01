from __future__ import annotations

import pandas as pd

from stability.parameter_sensitivity import ParameterSensitivityAnalyzer


def test_parameter_sensitivity_scores_stable_neighborhood_high():
    metrics = pd.DataFrame(
        [
            {"max_distance": 0.01, "net_CAGR": 0.09},
            {"max_distance": 0.02, "net_CAGR": 0.10},
            {"max_distance": 0.03, "net_CAGR": 0.095},
        ]
    )

    result = ParameterSensitivityAnalyzer(
        strategy_name="pullback_ma20_strategy",
        parameter_grid={"max_distance": [0.01, 0.02, 0.03]},
        metrics_by_param=metrics,
    ).analyze()

    assert result["best_params"] == {"max_distance": 0.02}
    assert result["sensitivity_score"] >= 70
    assert "isolated_best_parameter" not in result["instability_flags"]


def test_parameter_sensitivity_detects_parameter_island():
    metrics = pd.DataFrame(
        [
            {"min_volume_ratio": 1.5, "net_CAGR": 0.01},
            {"min_volume_ratio": 1.8, "net_CAGR": 0.50},
            {"min_volume_ratio": 2.0, "net_CAGR": -0.02},
        ]
    )

    result = ParameterSensitivityAnalyzer(
        strategy_name="volume_breakout_strategy",
        parameter_grid={"min_volume_ratio": [1.5, 1.8, 2.0]},
        metrics_by_param=metrics,
    ).analyze()

    assert result["sensitivity_score"] < 60
    assert "isolated_best_parameter" in result["instability_flags"]


def test_parameter_sensitivity_supports_two_dimensional_grid():
    metrics = pd.DataFrame(
        [
            {"min_amount": 100_000_000, "min_return_20d": 0.05, "net_CAGR": 0.04},
            {"min_amount": 100_000_000, "min_return_20d": 0.10, "net_CAGR": 0.08},
            {"min_amount": 200_000_000, "min_return_20d": 0.05, "net_CAGR": 0.09},
            {"min_amount": 200_000_000, "min_return_20d": 0.10, "net_CAGR": 0.10},
        ]
    )

    result = ParameterSensitivityAnalyzer(
        strategy_name="trend_strength_strategy",
        parameter_grid={"min_amount": [100_000_000, 200_000_000], "min_return_20d": [0.05, 0.10]},
        metrics_by_param=metrics,
    ).analyze()

    assert result["best_params"] == {"min_amount": 200_000_000, "min_return_20d": 0.10}
    assert len(result["top_10_params"]) == 4
    assert result["parameter_table"][0]["rank"] == 1


def test_parameter_sensitivity_handles_empty_metrics():
    result = ParameterSensitivityAnalyzer(
        strategy_name="trend_strength_strategy",
        parameter_grid={"min_amount": [100_000_000]},
        metrics_by_param=pd.DataFrame(),
    ).analyze()

    assert result["parameter_table"] == []
    assert result["sensitivity_score"] == 0
    assert "no_parameter_metrics" in result["instability_flags"]
