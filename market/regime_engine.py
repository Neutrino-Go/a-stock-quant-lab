from __future__ import annotations

import pandas as pd

from backtest.backtest_engine import calculate_trade_metrics


REGIME_ORDER = ["Bull", "Bear", "Sideways"]


def classify_market_regime(
    index_df: pd.DataFrame,
    ma_window: int = 60,
) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame(columns=["date", "close", "ma60", "regime"])

    required = {"date", "close"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")

    result = index_df.loc[:, ["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    result["ma60"] = result["close"].rolling(ma_window, min_periods=1).mean()
    result["regime"] = "Sideways"
    result.loc[result["close"].gt(result["ma60"]), "regime"] = "Bull"
    result.loc[result["close"].lt(result["ma60"]), "regime"] = "Bear"
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

    trades = trade_details.copy()
    trades[trade_date_column] = pd.to_datetime(trades[trade_date_column])
    regimes = market_regime.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    if "regime" not in regimes.columns:
        regimes = classify_market_regime(regimes)

    merged = pd.merge_asof(
        trades.sort_values(trade_date_column),
        regimes.loc[:, ["date", "regime"]].sort_values("date"),
        left_on=trade_date_column,
        right_on="date",
        direction="backward",
    )
    merged["regime"] = merged["regime"].fillna("Sideways")

    rows = []
    for regime, group in merged.groupby("regime", sort=False):
        metrics = calculate_trade_metrics(group)
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
