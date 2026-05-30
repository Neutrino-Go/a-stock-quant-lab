from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtest_engine import calculate_trade_metrics


REGIME_ORDER = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]
REGIME_LABEL_MAP = {
    "Bull": "BULL",
    "Bear": "BEAR",
    "Sideways": "SIDEWAYS",
    "High Volatility": "HIGH_VOLATILITY",
}


def classify_market_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "close",
                "ma20",
                "ma60",
                "ma120",
                "return_20d",
                "volatility_20d",
                "trend_slope_60d",
                "regime",
            ]
        )

    required = {"date", "close"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")

    result = index_df.loc[:, ["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    result["ma20"] = result["close"].rolling(20, min_periods=1).mean()
    result["ma60"] = result["close"].rolling(60, min_periods=1).mean()
    result["ma120"] = result["close"].rolling(120, min_periods=1).mean()
    result["return_20d"] = result["close"].pct_change(20, fill_method=None)
    daily_return = result["close"].pct_change(fill_method=None)
    result["volatility_20d"] = daily_return.rolling(20, min_periods=2).std()
    result["trend_slope_60d"] = result["close"].pct_change(60, fill_method=None) / 60
    vol_q75 = result["volatility_20d"].rolling(252, min_periods=20).quantile(0.75)

    bull = result["close"].gt(result["ma60"]) & result["ma60"].gt(result["ma120"]) & result["return_20d"].gt(0)
    bear = result["close"].lt(result["ma60"]) & result["ma60"].lt(result["ma120"]) & result["return_20d"].lt(0)
    high_vol = result["volatility_20d"].gt(vol_q75) & result["volatility_20d"].gt(1e-6)

    result["regime"] = np.select(
        [high_vol, bull, bear],
        ["HIGH_VOLATILITY", "BULL", "BEAR"],
        default="SIDEWAYS",
    )
    return result


def build_market_proxy_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=["date", "close"])
    required = {"date", "close"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing price columns for market proxy: {sorted(missing)}")

    proxy = price_df.copy()
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy["close"] = pd.to_numeric(proxy["close"], errors="coerce")
    return (
        proxy.dropna(subset=["date", "close"])
        .groupby("date", as_index=False)["close"]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )


def summarize_trades_by_regime(
    trade_details: pd.DataFrame,
    market_regime: pd.DataFrame,
    trade_date_column: str = "signal_date",
) -> pd.DataFrame:
    if trade_details.empty or market_regime.empty:
        return _empty_regime_summary()

    trades = trade_details.copy()
    trades[trade_date_column] = pd.to_datetime(trades[trade_date_column])
    regimes = market_regime.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    if "regime" not in regimes.columns:
        regimes = classify_market_regime(regimes)
    regimes["regime"] = regimes["regime"].map(lambda value: REGIME_LABEL_MAP.get(str(value), str(value)))

    merged = pd.merge_asof(
        trades.sort_values(trade_date_column),
        regimes.loc[:, ["date", "regime"]].sort_values("date"),
        left_on=trade_date_column,
        right_on="date",
        direction="backward",
    )
    merged["regime"] = merged["regime"].fillna("SIDEWAYS")

    rows = []
    return_column = "net_return" if "net_return" in merged.columns else "return"
    for regime, group in merged.groupby("regime", sort=False):
        metrics_input = group.rename(columns={return_column: "return"})
        metrics = calculate_trade_metrics(metrics_input)
        rows.append(
            {
                "regime": regime,
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "avg_return": metrics["avg_return"],
                "CAGR": metrics["CAGR"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "profit_loss_ratio": metrics["profit_loss_ratio"],
            }
        )

    result = pd.DataFrame(rows)
    result["regime"] = pd.Categorical(result["regime"], categories=REGIME_ORDER, ordered=True)
    return result.sort_values("regime").reset_index(drop=True)


def _empty_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "total_trades",
            "win_rate",
            "avg_return",
            "CAGR",
            "max_drawdown",
            "sharpe_ratio",
            "profit_loss_ratio",
        ]
    )
