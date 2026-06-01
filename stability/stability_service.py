from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from backtest.backtest_engine import apply_realistic_costs_to_trades, run_strategy_backtest
from data.database import get_stock_daily
from execution.cost_model import TradingCostConfig
from factor.factor_engine import compute_factors
from ranking.ranking_engine import STRATEGIES
from research.experiment_service import ExperimentService
from stability.overfit_detector import OverfitDetector
from stability.parameter_sensitivity import ParameterSensitivityAnalyzer
from stability.splitter import create_train_test_split
from stability.stability_metrics import calculate_stability_score
from stability.walk_forward import WalkForwardAnalyzer, iter_parameter_grid, select_best_parameter_result


DEFAULT_STABILITY_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "trend_strength_strategy": {
        "min_amount": [100_000_000, 200_000_000, 300_000_000, 500_000_000],
        "return_20d": [0.05, 0.10, 0.15, 0.20],
    },
    "pullback_ma20_strategy": {
        "distance_to_ma20": [0.01, 0.02, 0.03, 0.05],
        "volume_ratio_max": [0.8, 1.0, 1.2, 1.5],
    },
    "volume_breakout_strategy": {
        "volume_ratio": [1.5, 1.8, 2.0, 2.5],
        "return_min": [0.02, 0.03, 0.04],
        "return_max": [0.06, 0.08, 0.10],
    },
    "multi_factor_score_strategy": {
        "top_n": [5, 10, 20, 30],
        "score_threshold": [60, 70, 80],
    },
}

REGIMES = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]


class StrategyStabilityService:
    def __init__(
        self,
        experiment_service: ExperimentService | None = None,
        factor_data_loader=None,
        backtest_runner=None,
    ) -> None:
        self.experiment_service = experiment_service or ExperimentService()
        self.factor_data_loader = factor_data_loader or _load_factor_data
        self.backtest_runner = backtest_runner or self._default_backtest_runner

    def run_stability_analysis(
        self,
        strategy_name: str,
        parameter_grid: dict[str, list[Any]] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        train_ratio: float = 0.7,
        walk_forward_config: dict[str, int] | None = None,
        cost_config: TradingCostConfig | None = None,
        position_config: Any | None = None,
        exit_config: Any | None = None,
        market_config: dict[str, Any] | None = None,
        save_as_experiment: bool = False,
        experiment_name: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        analysis_id = str(uuid.uuid4())
        parameter_grid = parameter_grid or DEFAULT_STABILITY_PARAMETER_GRIDS.get(strategy_name, {})
        walk_forward_config = walk_forward_config or {"train_months": 24, "test_months": 6, "step_months": 6}
        cost_config = cost_config or TradingCostConfig()
        market_config = market_config or {}
        start_date = str(start_date) if start_date is not None else None
        end_date = str(end_date) if end_date is not None else None

        try:
            factor_data = self.factor_data_loader(start_date=start_date, end_date=end_date)
            if factor_data is None:
                factor_data = pd.DataFrame()
            if start_date is None or end_date is None:
                start_date, end_date = _infer_dates(factor_data)
            split = create_train_test_split(start_date, end_date, train_ratio=train_ratio)

            train_results = self._run_parameter_search(
                strategy_name=strategy_name,
                parameter_grid=parameter_grid,
                start_date=split["train_start"],
                end_date=split["train_end"],
                factor_data=factor_data,
                cost_config=cost_config,
                exit_config=exit_config,
            )
            best_train = select_best_parameter_result(train_results, "net_CAGR")
            best_params = dict(best_train["params"]) if best_train else {}
            in_sample_result = dict(best_train["metrics"]) if best_train else _empty_metrics()
            out_of_sample_result = self._run_backtest(
                strategy_name=strategy_name,
                params=best_params,
                start_date=split["test_start"],
                end_date=split["test_end"],
                factor_data=factor_data,
                cost_config=cost_config,
                exit_config=exit_config,
            )
            walk_forward_result = WalkForwardAnalyzer(backtest_runner=self._runner_for_walk_forward).run(
                strategy_name=strategy_name,
                parameter_grid=parameter_grid,
                start_date=start_date,
                end_date=end_date,
                train_months=int(walk_forward_config.get("train_months", 24)),
                test_months=int(walk_forward_config.get("test_months", 6)),
                step_months=int(walk_forward_config.get("step_months", 6)),
                optimization_metric="net_CAGR",
                factor_data=factor_data,
                cost_config=cost_config,
                exit_config=exit_config,
            )
            sensitivity_result = ParameterSensitivityAnalyzer(
                strategy_name=strategy_name,
                parameter_grid=parameter_grid,
                metrics_by_param=train_results,
                primary_metric="net_CAGR",
            ).analyze()
            trade_count = _trade_count(out_of_sample_result)
            parameter_count = len(iter_parameter_grid(parameter_grid))
            overfit_result = OverfitDetector().detect(
                in_sample_metrics=in_sample_result,
                out_of_sample_metrics=out_of_sample_result,
                walk_forward_summary=walk_forward_result["summary"],
                sensitivity_result=sensitivity_result,
                trade_count=trade_count,
                parameter_count=parameter_count,
            )
            market_regime_breakdown = self._market_regime_breakdown(out_of_sample_result)
            stability_score = calculate_stability_score(
                in_sample_metrics=in_sample_result,
                out_of_sample_metrics=out_of_sample_result,
                walk_forward_summary=walk_forward_result["summary"],
                sensitivity_result=sensitivity_result,
                market_regime_breakdown=market_regime_breakdown,
                overfit_result=overfit_result,
            )
            result = {
                "analysis_id": analysis_id,
                "strategy_name": strategy_name,
                "best_params": best_params,
                "split": split,
                "in_sample_result": _metrics_without_frames(in_sample_result),
                "out_of_sample_result": _metrics_without_frames(out_of_sample_result),
                "walk_forward_result": walk_forward_result,
                "sensitivity_result": sensitivity_result,
                "overfit_result": overfit_result,
                "stability_score": stability_score,
                "market_regime_breakdown": market_regime_breakdown,
                "recommendation": overfit_result["recommendation"],
                "parameter_search_top_20": sensitivity_result.get("top_10_params", []),
                "status": "completed",
            }
            if save_as_experiment:
                analysis_id = self.experiment_service.save_stability_analysis_result(
                    name=experiment_name or f"{strategy_name} stability analysis",
                    result=result,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=strategy_name,
                    config={
                        "strategy_config": {"parameter_grid": parameter_grid, "best_params": best_params},
                        "cost_config": _config_to_dict(cost_config),
                        "position_config": _config_to_dict(position_config),
                        "exit_config": _config_to_dict(exit_config),
                        "market_config": market_config,
                        "walk_forward_config": walk_forward_config,
                        "train_ratio": train_ratio,
                    },
                    tags=tags or [],
                    notes=notes,
                )
                result["analysis_id"] = analysis_id
            return result
        except Exception as exc:
            result = {
                "analysis_id": analysis_id,
                "strategy_name": strategy_name,
                "status": "failed",
                "error": str(exc),
                "in_sample_result": {},
                "out_of_sample_result": {},
                "walk_forward_result": {"folds": [], "summary": {}},
                "sensitivity_result": {},
                "overfit_result": {"overfit_risk": "HIGH", "overfit_score": 100, "risk_flags": ["analysis_failed"]},
                "stability_score": {"stability_score": 0, "final_rating": "D"},
                "market_regime_breakdown": _empty_regime_breakdown(),
                "recommendation": "reject_for_now",
            }
            if save_as_experiment:
                experiment_id = self.experiment_service.save_stability_analysis_result(
                    name=experiment_name or f"{strategy_name} stability analysis failed",
                    result=result,
                    start_date=start_date or "",
                    end_date=end_date or "",
                    strategy_name=strategy_name,
                    config={
                        "strategy_config": {"parameter_grid": parameter_grid},
                        "cost_config": _config_to_dict(cost_config),
                        "position_config": _config_to_dict(position_config),
                        "exit_config": _config_to_dict(exit_config),
                        "market_config": market_config,
                        "walk_forward_config": walk_forward_config,
                        "train_ratio": train_ratio,
                    },
                    tags=tags or [],
                    notes=notes,
                    status="failed",
                )
                result["analysis_id"] = experiment_id
            return result

    def _run_parameter_search(
        self,
        strategy_name: str,
        parameter_grid: dict[str, list[Any]],
        start_date: str,
        end_date: str,
        factor_data: pd.DataFrame,
        cost_config: TradingCostConfig,
        exit_config: Any | None,
    ) -> list[dict[str, Any]]:
        results = []
        for params in iter_parameter_grid(parameter_grid):
            metrics = self._run_backtest(
                strategy_name=strategy_name,
                params=params,
                start_date=start_date,
                end_date=end_date,
                factor_data=factor_data,
                cost_config=cost_config,
                exit_config=exit_config,
            )
            results.append({"params": params, "metrics": _metrics_without_frames(metrics)})
        return results

    def _runner_for_walk_forward(self, strategy_name: str, params: dict[str, Any], start_date, end_date, **kwargs) -> dict[str, Any]:
        return _metrics_without_frames(
            self._run_backtest(
                strategy_name=strategy_name,
                params=params,
                start_date=start_date,
                end_date=end_date,
                factor_data=kwargs.get("factor_data", pd.DataFrame()),
                cost_config=kwargs.get("cost_config") or TradingCostConfig(),
                exit_config=kwargs.get("exit_config"),
            )
        )

    def _run_backtest(
        self,
        strategy_name: str,
        params: dict[str, Any],
        start_date: str,
        end_date: str,
        factor_data: pd.DataFrame,
        cost_config: TradingCostConfig,
        exit_config: Any | None,
    ) -> dict[str, Any]:
        return self.backtest_runner(
            strategy_name=strategy_name,
            params=params,
            start_date=start_date,
            end_date=end_date,
            factor_data=factor_data,
            cost_config=cost_config,
            exit_config=exit_config,
        )

    def _default_backtest_runner(
        self,
        strategy_name: str,
        params: dict[str, Any],
        start_date,
        end_date,
        factor_data: pd.DataFrame,
        cost_config: TradingCostConfig,
        exit_config: Any | None = None,
    ) -> dict[str, Any]:
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        strategy_params = normalize_strategy_params(strategy_name, params)
        result = run_strategy_backtest(
            factor_df=factor_data,
            strategy_func=STRATEGIES[strategy_name],
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            holding_days=int(getattr(exit_config, "holding_days", 20) if exit_config is not None else 20),
            take_profit_pct=getattr(exit_config, "take_profit_pct", 0.10) if exit_config is not None else 0.10,
            stop_loss_pct=getattr(exit_config, "stop_loss_pct", 0.08) if exit_config is not None else 0.08,
            sell_on_ma20_break=bool(getattr(exit_config, "sell_on_ma20_break", True)) if exit_config is not None else True,
        )
        trade_details = result.get("trade_details")
        if isinstance(trade_details, pd.DataFrame) and not trade_details.empty:
            adjusted_trades, cost_metrics = apply_realistic_costs_to_trades(trade_details, cost_config=cost_config)
            merged = {key: value for key, value in result.items() if key != "trade_details"}
            merged.update({key: value for key, value in cost_metrics.items() if key != "trade_details"})
            merged["trade_details"] = adjusted_trades
            merged["gross_CAGR"] = result.get("CAGR", 0)
            merged["net_CAGR"] = cost_metrics.get("CAGR", 0)
            return merged
        result = dict(result)
        result["gross_CAGR"] = result.get("CAGR", 0)
        result["net_CAGR"] = result.get("CAGR", 0)
        result["cost_impact"] = 0.0
        return result

    @staticmethod
    def _market_regime_breakdown(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        trades = result.get("trade_details")
        if not isinstance(trades, pd.DataFrame) or trades.empty or "market_regime" not in trades.columns:
            return _empty_regime_breakdown("sample_insufficient_or_regime_unavailable")

        breakdown = _empty_regime_breakdown()
        return_col = "net_return" if "net_return" in trades.columns else "return"
        for regime, group in trades.groupby("market_regime"):
            returns = pd.to_numeric(group.get(return_col, pd.Series(dtype=float)), errors="coerce").dropna()
            breakdown[str(regime)] = {
                "trade_count": int(len(group)),
                "net_CAGR": float((1 + returns).prod() - 1) if not returns.empty else 0.0,
                "win_rate": float(returns.gt(0).mean()) if not returns.empty else 0.0,
                "max_drawdown": 0.0,
                "sample_note": "insufficient_sample" if len(group) < 10 else "",
            }
        return breakdown


def normalize_strategy_params(strategy_name: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if strategy_name == "trend_strength_strategy":
        if "return_20d" in normalized:
            normalized["min_return_20d"] = normalized.pop("return_20d")
    elif strategy_name == "pullback_ma20_strategy":
        if "distance_to_ma20" in normalized:
            normalized["max_distance"] = normalized.pop("distance_to_ma20")
        normalized.pop("volume_ratio_max", None)
    elif strategy_name == "volume_breakout_strategy":
        if "volume_ratio" in normalized:
            normalized["min_volume_ratio"] = normalized.pop("volume_ratio")
        normalized.pop("return_min", None)
        normalized.pop("return_max", None)
    elif strategy_name == "multi_factor_score_strategy":
        if "score_threshold" in normalized:
            normalized["min_score"] = normalized.pop("score_threshold")
    return normalized


def _load_factor_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    daily = get_stock_daily(start_date=start_date, end_date=end_date)
    if daily.empty:
        return daily
    return compute_factors(daily)


def _infer_dates(factor_data: pd.DataFrame) -> tuple[str, str]:
    if factor_data.empty or "date" not in factor_data.columns:
        raise ValueError("start_date and end_date are required when factor data is empty")
    dates = pd.to_datetime(factor_data["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("factor data has no valid date values")
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _empty_metrics() -> dict[str, Any]:
    return {"net_CAGR": 0.0, "CAGR": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_trades": 0}


def _trade_count(metrics: dict[str, Any]) -> int:
    if "total_trades" in metrics:
        return int(metrics.get("total_trades") or 0)
    trades = metrics.get("trade_details")
    return int(len(trades)) if isinstance(trades, pd.DataFrame) else 0


def _metrics_without_frames(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not isinstance(value, pd.DataFrame)}


def _empty_regime_breakdown(note: str = "insufficient_sample") -> dict[str, dict[str, Any]]:
    return {
        regime: {
            "trade_count": 0,
            "net_CAGR": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sample_note": note,
        }
        for regime in REGIMES
    }


def _config_to_dict(value: Any) -> Any:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _config_to_dict(item) for key, item in value.items()}
    return value
