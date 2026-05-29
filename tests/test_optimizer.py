import pandas as pd

from optimizer.parameter_optimizer import optimize_strategy


def _optimizer_factor_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = []
    for code, amount, ret20, exits in [
        ("000001", 350_000_000, 0.18, [10.0, 10.0, 10.0, 10.0, 12.0, 12.5]),
        ("000002", 150_000_000, 0.08, [10.0, 10.0, 10.0, 10.0, 9.5, 9.0]),
    ]:
        for idx, current_date in enumerate(dates):
            close = exits[idx]
            rows.append(
                {
                    "code": code,
                    "name": f"股票{code}",
                    "date": current_date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": amount,
                    "turnover": 1.0,
                    "ma20": 9.5,
                    "ma60": 9.0,
                    "return_5d": 0.01,
                    "return_20d": ret20,
                    "volume_ratio": 1.0,
                    "breakout_60d": True,
                    "distance_to_ma20": close / 9.5 - 1,
                    "max_drawdown_20d": 0.0,
                    "return_1d": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_optimize_strategy_returns_ranked_parameter_grid():
    result = optimize_strategy(
        _optimizer_factor_frame(),
        strategy_name="trend_strength_strategy",
        start_date="2024-01-01",
        end_date="2024-01-04",
        param_grid={"min_amount": [100_000_000, 300_000_000], "min_return_20d": [0.05, 0.15]},
        holding_days=2,
        top_n=4,
    )

    assert len(result) == 4
    assert result["sharpe_ratio"].is_monotonic_decreasing
    assert {"parameters", "win_rate", "annual_return", "max_drawdown", "sharpe_ratio"}.issubset(result.columns)
    assert result.iloc[-1]["parameters"] == {"min_amount": 100_000_000, "min_return_20d": 0.05}
