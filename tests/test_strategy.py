import pandas as pd

from strategy.multi_factor_score import multi_factor_score_strategy
from strategy.pullback_ma20 import pullback_ma20_strategy
from strategy.trend_strength import trend_strength_strategy
from strategy.volume_breakout import volume_breakout_strategy


SIGNAL_COLUMNS = {
    "code",
    "name",
    "signal_date",
    "strategy_name",
    "score",
    "signal_reason",
    "suggested_buy_price",
    "suggested_stop_loss",
    "suggested_take_profit",
    "risk_level",
}


def _factor_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "平安银行",
                "date": pd.Timestamp("2024-04-01"),
                "open": 12.0,
                "high": 12.8,
                "low": 11.8,
                "close": 12.5,
                "volume": 2_000_000,
                "amount": 350_000_000,
                "turnover": 2.1,
                "ma20": 11.2,
                "ma60": 10.4,
                "return_5d": 0.01,
                "return_20d": 0.18,
                "volume_ratio": 1.05,
                "breakout_60d": True,
                "distance_to_ma20": 12.5 / 11.2 - 1,
                "max_drawdown_20d": -0.03,
                "return_1d": 0.05,
            },
            {
                "code": "000002",
                "name": "*ST测试",
                "date": pd.Timestamp("2024-04-01"),
                "open": 12.0,
                "high": 12.8,
                "low": 11.8,
                "close": 12.5,
                "volume": 2_000_000,
                "amount": 350_000_000,
                "turnover": 2.1,
                "ma20": 11.2,
                "ma60": 10.4,
                "return_5d": 0.01,
                "return_20d": 0.18,
                "volume_ratio": 1.05,
                "breakout_60d": True,
                "distance_to_ma20": 12.5 / 11.2 - 1,
                "max_drawdown_20d": -0.03,
                "return_1d": 0.05,
            },
            {
                "code": "000003",
                "name": "回踩样本",
                "date": pd.Timestamp("2024-04-01"),
                "open": 10.1,
                "high": 10.3,
                "low": 9.8,
                "close": 10.1,
                "volume": 1_000_000,
                "amount": 180_000_000,
                "turnover": 1.0,
                "ma20": 10.0,
                "ma60": 9.5,
                "return_5d": -0.02,
                "return_20d": 0.04,
                "volume_ratio": 0.95,
                "breakout_60d": False,
                "distance_to_ma20": 0.01,
                "max_drawdown_20d": -0.06,
                "return_1d": 0.0,
            },
            {
                "code": "000004",
                "name": "放量突破",
                "date": pd.Timestamp("2024-04-01"),
                "open": 20.0,
                "high": 21.4,
                "low": 19.8,
                "close": 21.0,
                "volume": 3_000_000,
                "amount": 420_000_000,
                "turnover": 3.2,
                "ma20": 18.0,
                "ma60": 17.0,
                "return_5d": 0.06,
                "return_20d": 0.16,
                "volume_ratio": 2.1,
                "breakout_60d": True,
                "distance_to_ma20": 21.0 / 18.0 - 1,
                "max_drawdown_20d": -0.02,
                "return_1d": 0.05,
            },
        ]
    )


def test_trend_strength_strategy_excludes_st_and_returns_standard_columns():
    result = trend_strength_strategy(_factor_rows(), signal_date="2024-04-01")

    assert set(result.columns) == SIGNAL_COLUMNS
    assert "000001" in set(result["code"])
    assert "000002" not in set(result["code"])
    assert set(result["strategy_name"]) == {"trend_strength_strategy"}


def test_pullback_ma20_strategy_detects_ma20_pullback():
    result = pullback_ma20_strategy(_factor_rows(), signal_date="2024-04-01")

    assert list(result["code"]) == ["000003"]
    assert result.iloc[0]["risk_level"] in {"low", "medium", "high"}


def test_volume_breakout_strategy_detects_volume_breakout():
    result = volume_breakout_strategy(_factor_rows(), signal_date="2024-04-01")

    assert list(result["code"]) == ["000004"]
    assert result.iloc[0]["suggested_take_profit"] > result.iloc[0]["suggested_buy_price"]


def test_multi_factor_score_strategy_sorts_by_score_and_limits_results():
    result = multi_factor_score_strategy(_factor_rows(), signal_date="2024-04-01", top_n=2)

    assert len(result) == 2
    assert result["score"].is_monotonic_decreasing
    assert "000002" not in set(result["code"])
