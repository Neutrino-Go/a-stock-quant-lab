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


def plot_in_sample_vs_out_sample(
    in_sample_metrics: dict[str, object],
    out_of_sample_metrics: dict[str, object],
) -> go.Figure:
    if not in_sample_metrics and not out_of_sample_metrics:
        return _empty_figure("In-Sample vs Out-of-Sample", "No stability metrics")
    metrics = ["net_CAGR", "sharpe_ratio", "max_drawdown", "win_rate"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[_metric_value(in_sample_metrics, metric) for metric in metrics],
            name="In-Sample",
        )
    )
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[_metric_value(out_of_sample_metrics, metric) for metric in metrics],
            name="Out-of-Sample",
        )
    )
    fig.update_layout(
        title="In-Sample vs Out-of-Sample",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def plot_walk_forward_returns(walk_forward_result: dict[str, object]) -> go.Figure:
    folds = list(walk_forward_result.get("folds", []) or [])
    if not folds:
        return _empty_figure("Walk-forward Returns", "No walk-forward folds")
    labels = [f"Fold {fold.get('fold')}" for fold in folds]
    values = [_metric_value(fold.get("test_metrics", {}), "net_CAGR") * 100 for fold in folds]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#d94f45" if value >= 0 else "#2f7f5f" for value in values],
            name="Test net CAGR",
        )
    )
    fig.update_layout(
        title="Walk-forward Test Returns",
        xaxis_title="Fold",
        yaxis_title="Test net CAGR (%)",
        height=340,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def plot_parameter_sensitivity_heatmap(sensitivity_result: dict[str, object]) -> go.Figure:
    table = pd.DataFrame(sensitivity_result.get("parameter_table", []) or [])
    if table.empty:
        return _empty_figure("Parameter Sensitivity", "No parameter sensitivity data")
    metric = "net_CAGR" if "net_CAGR" in table.columns else "metric"
    param_columns = list(sensitivity_result.get("param_columns", []) or [])
    if not param_columns:
        param_columns = [
            column
            for column in table.columns
            if column not in {metric, "rank", "sharpe_ratio", "max_drawdown", "total_trades"}
        ]
    if len(param_columns) >= 2:
        x_col, y_col = param_columns[0], param_columns[1]
        pivot = table.pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean")
        fig = go.Figure(
            go.Heatmap(
                x=[str(item) for item in pivot.columns],
                y=[str(item) for item in pivot.index],
                z=pivot.values,
                colorbar=dict(title=metric),
            )
        )
        fig.update_layout(
            title="Parameter Sensitivity Heatmap",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=360,
            margin=dict(l=10, r=10, t=48, b=60),
        )
        return fig
    x_col = param_columns[0] if param_columns else "rank"
    fig = go.Figure(go.Bar(x=table[x_col].astype(str), y=pd.to_numeric(table[metric], errors="coerce"), name=metric))
    fig.update_layout(
        title="Parameter Sensitivity",
        xaxis_title=x_col,
        yaxis_title=metric,
        height=340,
        margin=dict(l=10, r=10, t=48, b=60),
    )
    return fig


def plot_overfit_risk_flags(overfit_result: dict[str, object]) -> go.Figure:
    flags = list(overfit_result.get("risk_flags", []) or [])
    if not flags:
        return _empty_figure("Overfit Risk Flags", "No overfitting flags")
    fig = go.Figure(
        go.Table(
            header=dict(values=["Risk Flag"]),
            cells=dict(values=[flags]),
        )
    )
    fig.update_layout(title="Overfit Risk Flags", height=300, margin=dict(l=10, r=10, t=48, b=10))
    return fig


def plot_regime_stability(market_regime_breakdown: dict[str, dict[str, object]]) -> go.Figure:
    if not market_regime_breakdown:
        return _empty_figure("Market Regime Stability", "No market regime data")
    regimes = [regime for regime in market_regime_breakdown if not regime.startswith("_")]
    if not regimes:
        return _empty_figure("Market Regime Stability", "No market regime data")
    values = [_metric_value(market_regime_breakdown[regime], "net_CAGR") * 100 for regime in regimes]
    fig = go.Figure(go.Bar(x=regimes, y=values, name="Net CAGR"))
    fig.update_layout(
        title="Market Regime Stability",
        xaxis_title="Regime",
        yaxis_title="Net CAGR (%)",
        height=340,
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


def _metric_value(metrics: dict[str, object], metric: str) -> float:
    value = metrics.get(metric)
    if value is None and metric == "net_CAGR":
        value = metrics.get("CAGR", metrics.get("annual_return", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
