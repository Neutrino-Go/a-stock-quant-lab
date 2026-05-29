import pandas as pd

from market.regime_engine import classify_market_regime, summarize_trades_by_regime


def test_classify_market_regime_uses_current_and_past_ma60():
    dates = pd.bdate_range("2024-01-01", periods=65)
    index_df = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0] * 60 + [110.0, 112.0, 90.0, 88.0, 87.0],
        }
    )

    regimes = classify_market_regime(index_df)

    assert regimes.iloc[0]["regime"] == "Sideways"
    assert regimes.iloc[60]["regime"] == "Bull"
    assert regimes.iloc[62]["regime"] == "Bear"
    assert regimes.iloc[60]["ma60"] == index_df["close"].iloc[1:61].mean()


def test_summarize_trades_by_regime_groups_trade_performance():
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "regime": ["Bull", "Bear", "Sideways"],
        }
    )
    trades = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "return": [0.10, -0.05, 0.02],
        }
    )

    summary = summarize_trades_by_regime(trades, market)

    assert set(summary["regime"]) == {"Bull", "Bear", "Sideways"}
    bull = summary.loc[summary["regime"].eq("Bull")].iloc[0]
    assert bull["total_trades"] == 1
    assert bull["win_rate"] == 1.0
