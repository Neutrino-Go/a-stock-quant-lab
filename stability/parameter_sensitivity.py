from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ParameterSensitivityAnalyzer:
    strategy_name: str
    parameter_grid: dict[str, list[Any]]
    metrics_by_param: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]
    primary_metric: str = "net_CAGR"

    def analyze(self) -> dict[str, Any]:
        table = self._prepare_table()
        if table.empty or self.primary_metric not in table.columns:
            return {
                "parameter_table": [],
                "param_columns": [],
                "best_params": {},
                "top_10_params": [],
                "top_20_params": [],
                "median_metric": 0.0,
                "metric_std": 0.0,
                "best_vs_median_gap": 0.0,
                "local_neighborhood_score": 0.0,
                "sensitivity_score": 0.0,
                "instability_flags": ["no_parameter_metrics"],
            }

        table[self.primary_metric] = pd.to_numeric(table[self.primary_metric], errors="coerce")
        table = table.dropna(subset=[self.primary_metric]).reset_index(drop=True)
        if table.empty:
            return {
                "parameter_table": [],
                "param_columns": [],
                "best_params": {},
                "top_10_params": [],
                "top_20_params": [],
                "median_metric": 0.0,
                "metric_std": 0.0,
                "best_vs_median_gap": 0.0,
                "local_neighborhood_score": 0.0,
                "sensitivity_score": 0.0,
                "instability_flags": ["no_parameter_metrics"],
            }

        param_columns = [column for column in self.parameter_grid if column in table.columns]
        ranked = table.sort_values(self.primary_metric, ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        best_row = ranked.iloc[0]
        best_params = {column: _json_scalar(best_row[column]) for column in param_columns}
        best_metric = float(best_row[self.primary_metric])
        median_metric = float(ranked[self.primary_metric].median())
        metric_std = float(ranked[self.primary_metric].std(ddof=0) or 0.0)
        best_vs_median_gap = float(best_metric - median_metric)
        local_score = self._local_neighborhood_score(table, best_params, best_metric)
        sensitivity_score = _sensitivity_score(best_metric, median_metric, metric_std, local_score)
        flags = _instability_flags(sensitivity_score, best_metric, median_metric, metric_std, local_score)

        return {
            "parameter_table": _records(ranked),
            "param_columns": param_columns,
            "best_params": best_params,
            "top_10_params": _records(ranked.head(10)),
            "top_20_params": _records(ranked.head(20)),
            "median_metric": median_metric,
            "metric_std": metric_std,
            "best_vs_median_gap": best_vs_median_gap,
            "local_neighborhood_score": local_score,
            "sensitivity_score": sensitivity_score,
            "instability_flags": flags,
        }

    def _prepare_table(self) -> pd.DataFrame:
        if isinstance(self.metrics_by_param, pd.DataFrame):
            frame = self.metrics_by_param.copy()
        elif isinstance(self.metrics_by_param, list):
            frame = pd.DataFrame(self.metrics_by_param)
        elif isinstance(self.metrics_by_param, dict):
            frame = pd.DataFrame(self.metrics_by_param.get("parameter_table", []))
        else:
            frame = pd.DataFrame()

        if frame.empty:
            return frame
        if "params" in frame.columns:
            params = pd.json_normalize(frame["params"]).add_prefix("")
            frame = pd.concat([frame.drop(columns=["params"]), params], axis=1)
        if "parameters" in frame.columns:
            params = pd.json_normalize(frame["parameters"]).add_prefix("")
            frame = pd.concat([frame.drop(columns=["parameters"]), params], axis=1)
        if "metrics" in frame.columns:
            metrics = pd.json_normalize(frame["metrics"]).add_prefix("")
            frame = pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)
        return frame

    def _local_neighborhood_score(
        self,
        table: pd.DataFrame,
        best_params: dict[str, Any],
        best_metric: float,
    ) -> float:
        if not best_params:
            return 50.0
        neighbors = table[table.apply(lambda row: self._is_neighbor(row, best_params), axis=1)]
        if neighbors.empty:
            return 50.0
        neighbor_mean = float(pd.to_numeric(neighbors[self.primary_metric], errors="coerce").dropna().mean())
        if not np.isfinite(neighbor_mean):
            return 0.0
        if best_metric <= 0:
            return max(0.0, min(100.0, 100.0 + (neighbor_mean - best_metric) * 100.0))
        return max(0.0, min(100.0, neighbor_mean / best_metric * 100.0))

    def _is_neighbor(self, row: pd.Series, best_params: dict[str, Any]) -> bool:
        diffs = []
        for param, best_value in best_params.items():
            grid_values = list(self.parameter_grid.get(param, []))
            if param not in row or best_value not in grid_values or row[param] not in grid_values:
                return False
            diffs.append(abs(grid_values.index(row[param]) - grid_values.index(best_value)))
        return bool(diffs) and max(diffs) <= 1 and any(diff > 0 for diff in diffs)


def _sensitivity_score(best_metric: float, median_metric: float, metric_std: float, local_score: float) -> float:
    denominator = abs(best_metric) if abs(best_metric) > 1e-9 else 1.0
    gap_ratio = max(0.0, (best_metric - median_metric) / denominator)
    cv_ratio = max(0.0, metric_std / denominator)
    gap_component = max(0.0, 100.0 - gap_ratio * 100.0)
    dispersion_component = max(0.0, 100.0 - cv_ratio * 100.0)
    score = local_score * 0.7 + gap_component * 0.2 + dispersion_component * 0.1
    return float(max(0.0, min(100.0, score)))


def _instability_flags(
    sensitivity_score: float,
    best_metric: float,
    median_metric: float,
    metric_std: float,
    local_score: float,
) -> list[str]:
    flags: list[str] = []
    denominator = abs(best_metric) if abs(best_metric) > 1e-9 else 1.0
    if local_score < 40:
        flags.append("isolated_best_parameter")
    if best_metric - median_metric > denominator * 0.5:
        flags.append("large_best_vs_median_gap")
    if metric_std > denominator * 0.5:
        flags.append("high_metric_std")
    if sensitivity_score < 40:
        flags.append("high_parameter_sensitivity")
    elif sensitivity_score < 60:
        flags.append("parameter_sensitivity")
    return flags


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_scalar(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if pd.isna(value):
        return None
    return value
