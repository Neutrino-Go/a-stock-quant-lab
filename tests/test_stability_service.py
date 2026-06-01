from __future__ import annotations

import pandas as pd

from research.experiment_service import ExperimentService
from research.experiment_store import ExperimentStore
from stability.stability_service import StrategyStabilityService


def _price_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    rows = []
    for code in ["000001", "000002"]:
        price = 10.0 if code == "000001" else 20.0
        for idx, day in enumerate(dates):
            close = price * (1 + idx * 0.0005)
            rows.append(
                {
                    "code": code,
                    "name": f"Stock {code}",
                    "date": day,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "ma20": close * 0.98,
                    "ma60": close * 0.95,
                    "return_20d": 0.08,
                    "amount": 300_000_000,
                    "volume_ratio": 1.0,
                    "breakout_60d": False,
                    "distance_to_ma20": 0.02,
                    "max_drawdown_20d": -0.03,
                }
            )
    return pd.DataFrame(rows)


def test_stability_service_runs_and_saves_strategy_stability_experiment(tmp_path):
    service = ExperimentService(ExperimentStore(tmp_path / "experiments.db"))

    def runner(strategy_name: str, params: dict[str, object], start_date, end_date, **kwargs) -> dict[str, object]:
        score = float(params.get("min_return_20d", 0.1))
        is_train = pd.Timestamp(end_date) <= pd.Timestamp("2021-05-01")
        cagr = 0.20 - score if is_train else 0.12 - score / 2
        return {
            "net_CAGR": cagr,
            "CAGR": cagr + 0.01,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.10,
            "total_trades": 45,
            "trade_details": pd.DataFrame({"net_return": [0.02, -0.01], "signal_date": ["2021-01-01", "2021-02-01"]}),
        }

    result = StrategyStabilityService(
        experiment_service=service,
        factor_data_loader=lambda start_date=None, end_date=None: _price_data(),
        backtest_runner=runner,
    ).run_stability_analysis(
        strategy_name="trend_strength_strategy",
        parameter_grid={"min_return_20d": [0.05, 0.10]},
        start_date="2020-01-01",
        end_date="2021-12-31",
        train_ratio=0.7,
        walk_forward_config={"train_months": 6, "test_months": 3, "step_months": 3},
        save_as_experiment=True,
        experiment_name="Stability run",
        tags=["stability"],
    )

    assert result["analysis_id"]
    assert result["stability_score"]["final_rating"] in {"A", "B", "C", "D"}
    detail = service.build_experiment_detail(result["analysis_id"])
    assert detail["run"]["experiment_type"] == "strategy_stability_analysis"
    assert detail["run"]["metrics_json"]["strategy_name"] == "trend_strength_strategy"


def test_stability_service_empty_trade_result_does_not_error(tmp_path):
    result = StrategyStabilityService(
        experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
        factor_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(),
        backtest_runner=lambda *args, **kwargs: {"net_CAGR": 0, "sharpe_ratio": 0, "max_drawdown": 0, "total_trades": 0},
    ).run_stability_analysis(
        strategy_name="trend_strength_strategy",
        parameter_grid={"min_return_20d": [0.05]},
        start_date="2020-01-01",
        end_date="2020-12-31",
        walk_forward_config={"train_months": 3, "test_months": 3, "step_months": 3},
    )

    assert result["overfit_result"]["overfit_risk"] in {"MEDIUM", "HIGH"}
    assert result["market_regime_breakdown"]["BULL"]["trade_count"] == 0


def test_stability_service_records_failed_experiment_when_save_enabled(tmp_path):
    service = ExperimentService(ExperimentStore(tmp_path / "experiments.db"))

    def failing_runner(*args, **kwargs):
        raise RuntimeError("runner failed")

    result = StrategyStabilityService(
        experiment_service=service,
        factor_data_loader=lambda start_date=None, end_date=None: _price_data(),
        backtest_runner=failing_runner,
    ).run_stability_analysis(
        strategy_name="trend_strength_strategy",
        parameter_grid={"min_return_20d": [0.05]},
        start_date="2020-01-01",
        end_date="2020-12-31",
        save_as_experiment=True,
        experiment_name="Failed stability",
    )

    detail = service.build_experiment_detail(result["analysis_id"])
    assert detail["run"]["status"] == "failed"
    assert "runner failed" in detail["run"]["metrics_json"]["error"]


def test_market_regime_breakdown_uses_fallback_proxy_and_assigns_trade_regime_without_future_date(tmp_path):
    service = StrategyStabilityService(
        experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
        factor_data_loader=lambda start_date=None, end_date=None: _price_data(),
        index_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(),
        backtest_runner=lambda *args, **kwargs: {
            "net_CAGR": 0.05,
            "CAGR": 0.06,
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.05,
            "total_trades": 2,
            "trade_details": pd.DataFrame(
                {
                    "signal_date": ["2021-06-15", "2021-07-15"],
                    "buy_date": ["2021-06-16", "2021-07-16"],
                    "exit_date": ["2021-06-30", "2021-07-30"],
                    "net_return": [0.03, -0.01],
                    "return": [0.04, -0.005],
                }
            ),
        },
    )

    result = service.run_stability_analysis(
        strategy_name="trend_strength_strategy",
        parameter_grid={"return_20d": [0.05]},
        start_date="2020-01-01",
        end_date="2021-12-31",
        walk_forward_config={"train_months": 6, "test_months": 3, "step_months": 3},
    )

    breakdown = result["market_regime_breakdown"]
    assert breakdown["_meta"]["regime_source"] == "fallback_equal_weight_proxy"
    assert sum(breakdown[regime]["trade_count"] for regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]) > 0
    assert any(
        breakdown[regime]["sample_note"] == "insufficient_sample"
        for regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]
        if breakdown[regime]["trade_count"] > 0
    )


def test_market_regime_breakdown_prefers_hs300_when_available(tmp_path):
    index_prices = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=400, freq="B"),
            "close": [100 + idx * 0.2 for idx in range(400)],
        }
    )
    service = StrategyStabilityService(
        experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
        factor_data_loader=lambda start_date=None, end_date=None: _price_data(),
        index_data_loader=lambda start_date=None, end_date=None: index_prices,
        backtest_runner=lambda *args, **kwargs: {
            "net_CAGR": 0.05,
            "CAGR": 0.06,
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.05,
            "total_trades": 1,
            "trade_details": pd.DataFrame({"signal_date": ["2021-01-05"], "net_return": [0.02], "return": [0.02]}),
        },
    )

    result = service.run_stability_analysis(
        strategy_name="trend_strength_strategy",
        parameter_grid={"return_20d": [0.05]},
        start_date="2020-01-01",
        end_date="2021-12-31",
        walk_forward_config={"train_months": 6, "test_months": 3, "step_months": 3},
    )

    assert result["market_regime_breakdown"]["_meta"]["regime_source"] == "HS300"


def test_assign_market_regime_uses_backward_asof_not_future_dates(tmp_path):
    factor_data = _price_data()
    service = StrategyStabilityService(
        experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
        factor_data_loader=lambda start_date=None, end_date=None: factor_data,
        index_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-10", "2021-01-20"]),
                "close": [100.0, 80.0],
                "regime": ["BULL", "BEAR"],
            }
        ),
    )
    trades = pd.DataFrame({"signal_date": ["2021-01-15"], "net_return": [0.01], "return": [0.01]})

    breakdown = service._market_regime_breakdown(
        {"trade_details": trades},
        factor_data=factor_data,
        start_date="2021-01-01",
        end_date="2021-01-31",
    )

    assert breakdown["BULL"]["trade_count"] == 1
    assert breakdown["BEAR"]["trade_count"] == 0


def test_market_regime_breakdown_empty_trades_has_meta_and_no_error(tmp_path):
    service = StrategyStabilityService(
        experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
        index_data_loader=lambda start_date=None, end_date=None: pd.DataFrame(),
    )

    breakdown = service._market_regime_breakdown(
        {"trade_details": pd.DataFrame()},
        factor_data=pd.DataFrame(),
        start_date="2021-01-01",
        end_date="2021-01-31",
    )

    assert breakdown["_meta"]["regime_source"] == "unavailable"
    assert breakdown["BULL"]["trade_count"] == 0
