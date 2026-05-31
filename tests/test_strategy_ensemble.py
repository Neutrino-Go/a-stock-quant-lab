import pandas as pd

from portfolio.strategy_ensemble import combine_strategy_signals


def _signal(strategy: str, code: str, score: float, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [code],
            "name": ["Sample"],
            "signal_date": [pd.Timestamp("2024-01-01")],
            "strategy_name": [strategy],
            "score": [score],
            "signal_reason": [reason],
            "suggested_buy_price": [10.0],
            "suggested_stop_loss": [9.0],
            "suggested_take_profit": [12.0],
            "risk_level": ["medium"],
        }
    )


def test_ensemble_merges_same_stock_same_date_and_keeps_triggered_strategies():
    combined = combine_strategy_signals(
        {
            "trend_strength_strategy": _signal("trend_strength_strategy", "000001", 80, "trend"),
            "pullback_ma20_strategy": _signal("pullback_ma20_strategy", "000001", 60, "pullback"),
        },
        weights={"trend_strength_strategy": 0.25, "pullback_ma20_strategy": 0.75},
    )

    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["strategy_name"] == "ensemble_strategy"
    assert row["triggered_strategies"] == "pullback_ma20_strategy,trend_strength_strategy"
    assert row["score"] == 65.0
    assert "trend" in row["signal_reason"]
    assert "pullback" in row["signal_reason"]


def test_ensemble_supports_daily_top_n_sorting_by_score():
    combined = combine_strategy_signals(
        {
            "trend_strength_strategy": pd.concat(
                [
                    _signal("trend_strength_strategy", "000001", 50, "low"),
                    _signal("trend_strength_strategy", "000002", 90, "high"),
                ],
                ignore_index=True,
            )
        },
        top_n_per_day=1,
    )

    assert list(combined["code"]) == ["000002"]
