from __future__ import annotations

import pandas as pd

from stability.walk_forward import WalkForwardAnalyzer


def test_walk_forward_uses_train_metrics_to_select_params_not_test_metrics():
    calls: list[dict[str, object]] = []

    def runner(strategy_name: str, params: dict[str, object], start_date, end_date, **kwargs) -> dict[str, object]:
        calls.append({"params": dict(params), "start_date": str(start_date), "end_date": str(end_date)})
        value = float(params["threshold"])
        is_train = pd.Timestamp(start_date) < pd.Timestamp("2020-07-01")
        if is_train:
            score = 0.20 if value == 2 else 0.05
        else:
            score = 0.50 if value == 1 else -0.10
        return {"net_CAGR": score, "sharpe_ratio": score * 3, "max_drawdown": -0.1, "total_trades": 40}

    analyzer = WalkForwardAnalyzer(backtest_runner=runner)
    result = analyzer.run(
        strategy_name="mock_strategy",
        parameter_grid={"threshold": [1, 2]},
        start_date="2020-01-01",
        end_date="2020-09-30",
        train_months=6,
        test_months=3,
        step_months=3,
    )

    assert result["folds"][0]["best_params"] == {"threshold": 2}
    assert result["folds"][0]["test_metrics"]["net_CAGR"] == -0.10
    assert not any(call["params"] == {"threshold": 1} and call["start_date"] == "2020-07-01" for call in calls)


def test_walk_forward_degradation_is_negative_when_out_of_sample_is_worse():
    def runner(strategy_name: str, params: dict[str, object], start_date, end_date, **kwargs) -> dict[str, object]:
        is_train = pd.Timestamp(start_date) < pd.Timestamp("2020-07-01")
        return {"net_CAGR": 0.30 if is_train else 0.05, "sharpe_ratio": 1.0, "max_drawdown": -0.1}

    result = WalkForwardAnalyzer(backtest_runner=runner).run(
        "mock_strategy",
        {"threshold": [1]},
        "2020-01-01",
        "2020-09-30",
        train_months=6,
        test_months=3,
        step_months=3,
    )

    assert result["folds"][0]["degradation"] < 0
    assert result["summary"]["avg_degradation"] < 0


def test_walk_forward_positive_test_fold_ratio_is_calculated():
    fold_scores = iter([0.10, 0.05, 0.10, -0.02])

    def runner(strategy_name: str, params: dict[str, object], start_date, end_date, **kwargs) -> dict[str, object]:
        return {"net_CAGR": next(fold_scores), "sharpe_ratio": 1.0, "max_drawdown": -0.1}

    result = WalkForwardAnalyzer(backtest_runner=runner).run(
        "mock_strategy",
        {"threshold": [1]},
        "2020-01-01",
        "2020-09-30",
        train_months=3,
        test_months=3,
        step_months=3,
    )

    assert result["summary"]["fold_count"] == 2
    assert result["summary"]["positive_test_fold_ratio"] == 0.5


def test_walk_forward_supports_max_drawdown_as_optimization_metric():
    def runner(strategy_name: str, params: dict[str, object], start_date, end_date, **kwargs) -> dict[str, object]:
        value = float(params["threshold"])
        return {"net_CAGR": 0.1, "sharpe_ratio": 1.0, "max_drawdown": -0.10 if value == 1 else -0.30}

    result = WalkForwardAnalyzer(backtest_runner=runner).run(
        "mock_strategy",
        {"threshold": [1, 2]},
        "2020-01-01",
        "2020-09-30",
        train_months=6,
        test_months=3,
        step_months=3,
        optimization_metric="max_drawdown",
    )

    assert result["folds"][0]["best_params"] == {"threshold": 1}


def test_walk_forward_empty_splits_return_empty_summary():
    result = WalkForwardAnalyzer(backtest_runner=lambda *args, **kwargs: {}).run(
        "mock_strategy",
        {"threshold": [1]},
        "2020-01-01",
        "2020-02-01",
        train_months=12,
        test_months=6,
        step_months=6,
    )

    assert result["folds"] == []
    assert result["summary"]["fold_count"] == 0
