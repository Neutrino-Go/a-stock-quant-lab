from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_price_with_ma(price_df: pd.DataFrame, title: str = "Price") -> go.Figure:
    df = price_df.copy()
    if df.empty:
        return _empty_figure(title, "No price data")
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
        return _empty_figure("Trade Returns", "No trades")
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
        return _empty_figure("Equity Curve", "No trades")
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
        return _empty_figure("Drawdown Curve", "No drawdown data")
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


def plot_portfolio_equity_curve(equity_curve: pd.DataFrame) -> go.Figure:
    df = equity_curve.copy()
    if df.empty or "total_equity" not in df.columns:
        return _empty_figure("Portfolio Equity Curve", "No portfolio equity data")
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["total_equity"], mode="lines", name="Net Equity"))
    if "gross_total_equity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["gross_total_equity"],
                mode="lines",
                name="Gross Equity",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=380,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_portfolio_drawdown_curve(drawdown_curve: pd.DataFrame) -> go.Figure:
    df = drawdown_curve.copy()
    if df.empty or "drawdown" not in df.columns:
        return _empty_figure("Portfolio Drawdown Curve", "No drawdown data")
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Portfolio Drawdown Curve",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=380,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_cost_breakdown(trades_or_orders: pd.DataFrame) -> go.Figure:
    df = trades_or_orders.copy()
    if df.empty:
        return _empty_figure("Cost Breakdown", "No cost data")
    items = {
        "Commission": _sum_column(df, "commission"),
        "Stamp Tax": _sum_column(df, "stamp_tax"),
        "Regulatory Fee": _sum_column(df, "regulatory_fee"),
        "Transfer Fee": _sum_column(df, "transfer_fee"),
        "Slippage": _sum_column(df, "slippage_cost"),
    }
    if sum(items.values()) <= 0:
        return _empty_figure("Cost Breakdown", "No trading costs")
    fig = go.Figure(go.Bar(x=list(items.keys()), y=list(items.values()), name="Cost"))
    fig.update_layout(
        title="Cost Breakdown",
        xaxis_title="Cost Type",
        yaxis_title="Amount",
        height=340,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def plot_position_count(equity_curve: pd.DataFrame) -> go.Figure:
    df = equity_curve.copy()
    if df.empty or "positions_count" not in df.columns:
        return _empty_figure("Position Count", "No position data")
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure(go.Scatter(x=df["date"], y=df["positions_count"], mode="lines", name="Positions"))
    fig.update_layout(
        title="Position Count",
        xaxis_title="Date",
        yaxis_title="Positions",
        height=320,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_monthly_returns(equity_curve: pd.DataFrame) -> go.Figure:
    df = equity_curve.copy()
    if df.empty or "total_equity" not in df.columns:
        return _empty_figure("Monthly Returns", "No equity data")
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.set_index("date")["total_equity"]
        .resample("ME")
        .last()
        .pct_change(fill_method=None)
        .dropna()
    )
    if monthly.empty:
        return _empty_figure("Monthly Returns", "Not enough data for monthly returns")
    labels = monthly.index.strftime("%Y-%m")
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=monthly * 100,
            marker_color=["#d94f45" if value >= 0 else "#2f7f5f" for value in monthly],
            name="Monthly Return",
        )
    )
    fig.update_layout(
        title="Monthly Returns",
        xaxis_title="Month",
        yaxis_title="Return (%)",
        height=340,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def plot_gross_vs_net_return(metrics: dict[str, object] | pd.DataFrame) -> go.Figure:
    if isinstance(metrics, pd.DataFrame):
        if metrics.empty:
            return _empty_figure("Gross vs Net Return", "No metrics")
        record = metrics.iloc[0].to_dict()
    else:
        record = dict(metrics)
    if not record:
        return _empty_figure("Gross vs Net Return", "No metrics")
    labels = ["Gross", "Net", "Cost Impact"]
    values = [
        float(record.get("gross_total_return", 0)) * 100,
        float(record.get("net_total_return", record.get("total_return", 0))) * 100,
        float(record.get("cost_impact", 0)) * 100,
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, name="Return"))
    fig.update_layout(
        title="Gross vs Net Return",
        xaxis_title="Metric",
        yaxis_title="Return (%)",
        height=320,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        title=title,
        height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())
