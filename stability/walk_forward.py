from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from stability.splitter import create_walk_forward_splits


BacktestRunner = Callable[..., dict[str, Any]]


class WalkForwardAnalyzer:
    def __init__(self, backtest_runner: BacktestRunner | None = None) -> None:
        self.backtest_runner = backtest_runner

    def run(
        self,
        strategy_name: str,
        parameter_grid: dict[str, list[Any]],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        train_months: int = 24,
        test_months: int = 6,
        step_months: int = 6,
        optimization_metric: str = "net_CAGR",
        **runner_kwargs: Any,
    ) -> dict[str, Any]:
        runner = self._runner()
        splits = create_walk_forward_splits(
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )
        folds: list[dict[str, Any]] = []
        for split in splits:
            train_results = []
            for params in iter_parameter_grid(parameter_grid):
                metrics = runner(
                    strategy_name=strategy_name,
                    params=params,
                    start_date=split["train_start"],
                    end_date=split["train_end"],
                    **runner_kwargs,
                )
                train_results.append({"params": params, "metrics": metrics})

            best = select_best_parameter_result(train_results, optimization_metric)
            if best is None:
                continue
            best_params = dict(best["params"])
            train_metrics = dict(best["metrics"])
            test_metrics = runner(
                strategy_name=strategy_name,
                params=best_params,
                start_date=split["test_start"],
                end_date=split["test_end"],
                **runner_kwargs,
            )
            degradation = _metric(test_metrics, optimization_metric) - _metric(train_metrics, optimization_metric)
            folds.append(
                {
                    "fold": split["fold"],
                    "train_period": {"start": split["train_start"], "end": split["train_end"]},
                    "test_period": {"start": split["test_start"], "end": split["test_end"]},
                    "best_params": best_params,
                    "train_metrics": train_metrics,
                    "test_metrics": dict(test_metrics),
                    "degradation_metric": optimization_metric,
                    "degradation": float(degradation),
                    "parameter_results": [
                        {"params": item["params"], "metrics": item["metrics"]} for item in train_results
                    ],
                }
            )

        return {"folds": folds, "summary": summarize_walk_forward(folds)}

    def _runner(self) -> BacktestRunner:
        if self.backtest_runner is None:
            raise ValueError("WalkForwardAnalyzer requires a backtest_runner")
        return self.backtest_runner


def iter_parameter_grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not parameter_grid:
        return [{}]
    keys = list(parameter_grid)
    values = [list(parameter_grid[key]) for key in keys]
    if any(len(items) == 0 for items in values):
        return []
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def select_best_parameter_result(
    results: list[dict[str, Any]],
    optimization_metric: str,
) -> dict[str, Any] | None:
    if not results:
        return None
    return max(results, key=lambda item: _metric(item.get("metrics", {}), optimization_metric))


def summarize_walk_forward(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        return {
            "fold_count": 0,
            "positive_test_fold_ratio": 0.0,
            "avg_train_net_CAGR": 0.0,
            "avg_test_net_CAGR": 0.0,
            "avg_degradation": 0.0,
            "median_test_net_CAGR": 0.0,
            "test_win_rate": 0.0,
            "param_stability_score": 0.0,
            "walk_forward_score": 0.0,
        }

    train_cagrs = [_metric(fold["train_metrics"], "net_CAGR") for fold in folds]
    test_cagrs = [_metric(fold["test_metrics"], "net_CAGR") for fold in folds]
    degradations = [float(fold.get("degradation", 0.0)) for fold in folds]
    positive_ratio = float(np.mean([value > 0 for value in test_cagrs]))
    best_param_keys = [_param_key(fold.get("best_params", {})) for fold in folds]
    most_common = Counter(best_param_keys).most_common(1)[0][1] if best_param_keys else 0
    param_stability = most_common / len(folds) * 100 if folds else 0.0
    avg_degradation = float(np.mean(degradations))
    degradation_component = max(0.0, min(30.0, 30.0 + avg_degradation * 100.0))
    walk_forward_score = max(0.0, min(100.0, positive_ratio * 70.0 + degradation_component))

    return {
        "fold_count": int(len(folds)),
        "positive_test_fold_ratio": positive_ratio,
        "avg_train_net_CAGR": float(np.mean(train_cagrs)),
        "avg_test_net_CAGR": float(np.mean(test_cagrs)),
        "avg_degradation": avg_degradation,
        "median_test_net_CAGR": float(np.median(test_cagrs)),
        "test_win_rate": positive_ratio,
        "param_stability_score": float(param_stability),
        "walk_forward_score": float(walk_forward_score),
    }


def _metric(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None and metric_name == "net_CAGR":
        value = metrics.get("CAGR", metrics.get("annual_return", 0))
    if value is None:
        value = 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _param_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in params.items()))
