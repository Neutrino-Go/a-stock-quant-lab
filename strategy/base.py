from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
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
]


def empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def latest_signal_rows(df: pd.DataFrame, signal_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    if signal_date is None:
        signal_ts = prepared["date"].max()
    else:
        signal_ts = pd.to_datetime(signal_date)
    rows = prepared[prepared["date"].eq(signal_ts)].copy()
    return rows.sort_values(["code", "date"])


def non_st_mask(rows: pd.DataFrame) -> pd.Series:
    names = rows["name"].fillna("").astype(str).str.upper()
    return ~names.str.contains("ST", regex=False)


def risk_level(max_drawdown_20d: pd.Series, distance_to_ma20: pd.Series) -> pd.Series:
    drawdown = max_drawdown_20d.fillna(0)
    distance = distance_to_ma20.abs().fillna(0)
    return pd.Series(
        np.select(
            [
                drawdown.lt(-0.15) | distance.gt(0.18),
                drawdown.lt(-0.08) | distance.gt(0.08),
            ],
            ["high", "medium"],
            default="low",
        ),
        index=max_drawdown_20d.index,
    )


def build_signal_frame(
    rows: pd.DataFrame,
    strategy_name: str,
    reason: str | pd.Series,
    stop_loss_pct: float = 0.07,
    take_profit_pct: float = 0.15,
) -> pd.DataFrame:
    if rows.empty:
        return empty_signal_frame()

    output = pd.DataFrame(index=rows.index)
    output["code"] = rows["code"].astype(str).str.zfill(6)
    output["name"] = rows["name"].astype(str)
    output["signal_date"] = pd.to_datetime(rows["date"])
    output["strategy_name"] = strategy_name
    score = rows["score"] if "score" in rows.columns else pd.Series(70, index=rows.index)
    output["score"] = score.astype(float).clip(0, 100).round(2)
    output["signal_reason"] = reason if isinstance(reason, pd.Series) else str(reason)
    output["suggested_buy_price"] = rows["close"].astype(float).round(3)
    output["suggested_stop_loss"] = (rows["close"].astype(float) * (1 - stop_loss_pct)).round(3)
    output["suggested_take_profit"] = (rows["close"].astype(float) * (1 + take_profit_pct)).round(3)
    if "risk_level" in rows.columns:
        output["risk_level"] = rows["risk_level"]
    else:
        drawdown = rows["max_drawdown_20d"] if "max_drawdown_20d" in rows.columns else pd.Series(index=rows.index)
        distance = rows["distance_to_ma20"] if "distance_to_ma20" in rows.columns else pd.Series(index=rows.index)
        output["risk_level"] = risk_level(drawdown, distance)
    return output.loc[:, SIGNAL_COLUMNS].sort_values("score", ascending=False).reset_index(drop=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing factor columns: {sorted(missing)}")
