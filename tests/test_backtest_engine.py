import pandas as pd

from backtest.backtest_engine import BacktestEngine, backtest_signals


def _price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    return pd.DataFrame(
        {
            "code": "000001",
            "name": "平安银行",
            "date": dates,
            "open": [10.0, 11.0, 11.4, 11.8, 12.0, 12.2, 12.4, 12.5],
            "high": [10.5, 11.5, 12.2, 12.0, 12.4, 12.6, 12.7, 12.8],
            "low": [9.8, 10.8, 11.2, 11.5, 11.8, 12.0, 12.1, 12.2],
            "close": [10.2, 11.2, 12.0, 11.9, 12.1, 12.3, 12.5, 12.6],
            "volume": [1000] * 8,
            "amount": [250_000_000] * 8,
            "turnover": [1.0] * 8,
            "ma20": [9.8, 10.5, 11.0, 11.6, 11.9, 12.1, 12.2, 12.3],
        }
    )


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["000001"],
            "name": ["平安银行"],
            "signal_date": [pd.Timestamp("2024-01-01")],
            "strategy_name": ["unit_strategy"],
            "score": [88.0],
            "signal_reason": ["unit test"],
            "suggested_buy_price": [10.2],
            "suggested_stop_loss": [9.5],
            "suggested_take_profit": [12.0],
            "risk_level": ["medium"],
        }
    )


def test_backtest_buys_next_trading_day_open_and_sells_at_take_profit():
    result = backtest_signals(
        _price_frame(),
        _signal_frame(),
        holding_days=5,
        take_profit_pct=0.10,
        stop_loss_pct=0.08,
        sell_on_ma20_break=True,
    )

    trade = result["trade_details"].iloc[0]
    assert trade["buy_date"] == pd.Timestamp("2024-01-02")
    assert trade["buy_price"] == 11.0
    assert trade["exit_date"] == pd.Timestamp("2024-01-03")
    assert trade["exit_reason"] == "take_profit"
    assert round(trade["return"], 6) == 0.10
    assert result["total_trades"] == 1
    assert result["win_rate"] == 1.0


def test_backtest_uses_fixed_holding_exit_when_no_risk_rule_triggers():
    prices = _price_frame()
    prices["high"] = prices["open"] * 1.01
    prices["low"] = prices["open"] * 0.99
    prices["ma20"] = prices["close"] * 0.95

    result = BacktestEngine(
        holding_days=3,
        take_profit_pct=None,
        stop_loss_pct=None,
        sell_on_ma20_break=True,
    ).run(prices, _signal_frame())

    trade = result["trade_details"].iloc[0]
    assert trade["buy_date"] == pd.Timestamp("2024-01-02")
    assert trade["exit_date"] == pd.Timestamp("2024-01-04")
    assert trade["exit_reason"] == "holding_period"


def test_backtest_returns_empty_metrics_without_valid_trades():
    signal = _signal_frame()
    signal["signal_date"] = pd.Timestamp("2024-01-10")

    result = backtest_signals(_price_frame(), signal)

    assert result["total_trades"] == 0
    assert result["win_rate"] == 0
    assert result["trade_details"].empty


def test_backtest_skips_overlapping_positions_for_same_stock_and_strategy():
    signals = pd.concat([_signal_frame(), _signal_frame()], ignore_index=True)
    signals.loc[1, "signal_date"] = pd.Timestamp("2024-01-02")

    result = backtest_signals(
        _price_frame(),
        signals,
        holding_days=3,
        take_profit_pct=None,
        stop_loss_pct=None,
        sell_on_ma20_break=False,
    )

    assert result["total_trades"] == 1
    assert result["trade_details"].iloc[0]["signal_date"] == pd.Timestamp("2024-01-01")


def test_backtest_reliability_metrics_use_single_clean_equity_curve():
    trades = pd.DataFrame(
        {
            "code": ["000001", "000002", "000003"],
            "name": ["A", "B", "C"],
            "strategy_name": ["unit", "unit", "unit"],
            "signal_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "buy_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "exit_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "buy_price": [10.0, 10.0, 10.0],
            "exit_price": [11.0, 9.0, 12.0],
            "exit_reason": ["holding_period", "holding_period", "holding_period"],
            "holding_days": [1, 1, 1],
            "return": [0.10, -0.10, 0.20],
        }
    )

    metrics = BacktestEngine._metrics(trades)

    assert metrics["total_trades"] == 3
    assert round(metrics["cumulative_return"], 6) == 0.188
    assert round(metrics["max_drawdown"], 6) == -0.10
    assert round(metrics["expectancy"], 6) == round((2 / 3) * 0.15 - (1 / 3) * 0.10, 6)
    assert set(["equity_curve", "drawdown_curve", "CAGR", "sharpe_ratio"]).issubset(metrics)
    assert metrics["equity_curve"].iloc[0]["equity"] == 1.0
    assert metrics["drawdown_curve"]["drawdown"].min() == metrics["max_drawdown"]
