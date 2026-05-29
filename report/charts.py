from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_price_with_ma(price_df: pd.DataFrame, title: str = "Price") -> go.Figure:
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig = go.Figure()
    if {"open", "high", "low", "close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="K",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="close"))

    if "ma20" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], mode="lines", name="MA20"))
    if "ma60" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], mode="lines", name="MA60"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_trade_returns(trade_details: pd.DataFrame) -> go.Figure:
    df = trade_details.copy()
    if df.empty:
        return go.Figure()
    df["label"] = df["code"].astype(str) + " " + pd.to_datetime(df["buy_date"]).dt.strftime("%Y-%m-%d")
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["return"] * 100,
            marker_color=df["return"].map(lambda value: "#d94f45" if value >= 0 else "#2f7f5f"),
            name="Return %",
        )
    )
    fig.update_layout(
        title="Trade Returns",
        xaxis_title="Trade",
        yaxis_title="Return (%)",
        height=360,
        margin=dict(l=10, r=10, t=48, b=80),
    )
    return fig


def plot_equity_curve(trade_details: pd.DataFrame) -> go.Figure:
    df = trade_details.copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("exit_date")
    df["equity"] = (1 + df["return"]).cumprod()
    fig = go.Figure(
        go.Scatter(
            x=pd.to_datetime(df["exit_date"]),
            y=df["equity"],
            mode="lines+markers",
            name="Equity",
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Exit Date",
        yaxis_title="Equity Multiple",
        height=360,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_drawdown_curve(drawdown_curve: pd.DataFrame) -> go.Figure:
    df = drawdown_curve.copy()
    if df.empty:
        return go.Figure()
    fig = go.Figure(
        go.Scatter(
            x=pd.to_datetime(df["date"]),
            y=df["drawdown"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=360,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig
