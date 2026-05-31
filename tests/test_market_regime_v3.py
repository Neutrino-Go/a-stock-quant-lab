import pandas as pd

from market.regime_engine import build_market_proxy_from_prices, classify_market_regime


def test_enhanced_market_regime_outputs_required_factor_columns():
    dates = pd.bdate_range("2024-01-01", periods=130)
    close = list(range(100, 230))
    index_df = pd.DataFrame({"date": dates, "close": close})

    regimes = classify_market_regime(index_df)

    required = {"ma20", "ma60", "ma120", "return_20d", "volatility_20d", "trend_slope_60d", "regime"}
    assert required.issubset(regimes.columns)
    assert regimes.iloc[-1]["regime"] == "BULL"


def test_enhanced_market_regime_detects_bear_market():
    dates = pd.bdate_range("2024-01-01", periods=130)
    close = [230 * (0.995**idx) for idx in range(130)]
    index_df = pd.DataFrame({"date": dates, "close": close})

    regimes = classify_market_regime(index_df)

    assert regimes.iloc[-1]["regime"] == "BEAR"


def test_enhanced_market_regime_detects_high_volatility():
    dates = pd.bdate_range("2024-01-01", periods=130)
    close = [100.0]
    for idx in range(1, 130):
        close.append(close[-1] * (1.08 if idx % 2 == 0 else 0.92))
    index_df = pd.DataFrame({"date": dates, "close": close})

    regimes = classify_market_regime(index_df)

    assert "HIGH_VOLATILITY" in set(regimes["regime"])


def test_market_proxy_uses_equal_weighted_normalized_prices():
    prices = pd.DataFrame(
        {
            "code": ["000001", "000002", "000001", "000002"],
            "date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
            ],
            "close": [1000.0, 10.0, 1000.0, 20.0],
        }
    )

    proxy = build_market_proxy_from_prices(prices)

    assert list(proxy["date"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert list(proxy["close"].round(4)) == [1.0, 1.5]


def test_market_proxy_without_code_falls_back_to_simple_average_sorted():
    prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "close": [30.0, 10.0, 20.0],
        }
    )

    proxy = build_market_proxy_from_prices(prices)

    assert list(proxy["date"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert list(proxy["close"]) == [15.0, 30.0]


def test_market_proxy_empty_data_does_not_error():
    proxy = build_market_proxy_from_prices(pd.DataFrame())

    assert list(proxy.columns) == ["date", "close"]
    assert proxy.empty
