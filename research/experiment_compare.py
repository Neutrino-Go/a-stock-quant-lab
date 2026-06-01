from __future__ import annotations

from typing import Any

import pandas as pd

from research.experiment_store import ExperimentStore


COMPARISON_METRICS = [
    "final_equity",
    "total_return",
    "net_total_return",
    "CAGR",
    "net_CAGR",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_loss_ratio",
    "expectancy",
    "total_cost",
    "cost_impact",
    "total_trades",
    "avg_holding_days",
    "exposure_ratio",
    "cash_drag",
]

LOWER_IS_BETTER = {"total_cost", "cost_impact", "cash_drag"}


def compare_experiments(
    experiment_ids: list[str],
    store: ExperimentStore | None = None,
) -> dict[str, Any]:
    if not 2 <= len(experiment_ids) <= 5:
        raise ValueError("compare_experiments requires 2 to 5 experiment ids")
    store = store or ExperimentStore()
    details = [store.get_experiment_detail(experiment_id) for experiment_id in experiment_ids]
    details = [detail for detail in details if detail["run"] is not None]
    if len(details) < 2:
        raise ValueError("At least two existing experiments are required")

    metrics_comparison = _metrics_comparison(details)
    normalized_equity_curves = _joined_equity_curves(details, normalize=True)
    drawdown_curves = _joined_drawdown_curves(details)
    cost_comparison = _cost_comparison(details)
    risk_return_summary = _risk_return_summary(metrics_comparison)
    winner_by_metric = _winner_by_metric(metrics_comparison)

    return {
        "metrics_comparison": metrics_comparison,
        "normalized_equity_curves": normalized_equity_curves,
        "drawdown_curves": drawdown_curves,
        "cost_comparison": cost_comparison,
        "risk_return_summary": risk_return_summary,
        "winner_by_metric": winner_by_metric,
    }


def _metrics_comparison(details: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for detail in details:
        run = detail["run"]
        metrics = run["metrics_json"]
        row = {
            "experiment_id": run["id"],
            "name": run["name"],
            "experiment_type": run["experiment_type"],
            "strategy_name": run.get("strategy_name"),
        }
        for metric in COMPARISON_METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _joined_equity_curves(details: list[dict[str, Any]], normalize: bool) -> pd.DataFrame:
    frames = []
    for detail in details:
        curve = detail["equity_curve"].copy()
        if curve.empty:
            continue
        run = detail["run"]
        label = run["name"]
        value_column = "net_equity" if "net_equity" in curve.columns else "total_equity"
        if value_column not in curve.columns:
            continue
        curve["date"] = pd.to_datetime(curve["date"])
        values = pd.to_numeric(curve[value_column], errors="coerce")
        if normalize:
            first = values.dropna().iloc[0] if not values.dropna().empty else None
            if first and first != 0:
                values = values / first
        frames.append(curve.assign(**{label: values}).loc[:, ["date", label]])
    return _outer_join_frames(frames)


def _joined_drawdown_curves(details: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for detail in details:
        curve = detail["equity_curve"].copy()
        if curve.empty or "drawdown" not in curve.columns:
            continue
        run = detail["run"]
        label = run["name"]
        curve["date"] = pd.to_datetime(curve["date"])
        frames.append(curve.assign(**{label: pd.to_numeric(curve["drawdown"], errors="coerce")}).loc[:, ["date", label]])
    return _outer_join_frames(frames)


def _outer_join_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


def _cost_comparison(details: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for detail in details:
        run = detail["run"]
        trades = detail["trades"]
        metrics = run["metrics_json"]
        rows.append(
            {
                "experiment_id": run["id"],
                "name": run["name"],
                "commission": _sum(trades, "commission"),
                "stamp_tax": _sum(trades, "stamp_tax"),
                "slippage_cost": _sum(trades, "slippage_cost"),
                "total_cost": metrics.get("total_cost", _sum(trades, "total_cost")),
                "cost_impact": metrics.get("cost_impact", 0),
            }
        )
    return pd.DataFrame(rows)


def _risk_return_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    columns = ["experiment_id", "name", "net_CAGR", "max_drawdown", "sharpe_ratio", "calmar_ratio", "cost_impact"]
    return metrics.loc[:, [column for column in columns if column in metrics.columns]].copy()


def _winner_by_metric(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for metric in COMPARISON_METRICS:
        if metric not in metrics.columns:
            continue
        series = pd.to_numeric(metrics[metric], errors="coerce")
        if series.dropna().empty:
            continue
        idx = series.idxmin() if metric in LOWER_IS_BETTER else series.idxmax()
        winners[metric] = {
            "experiment_id": str(metrics.loc[idx, "experiment_id"]),
            "name": str(metrics.loc[idx, "name"]),
            "value": float(series.loc[idx]),
        }
    return winners


def _sum(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())
