from __future__ import annotations

import pandas as pd

from research.experiment_service import ExperimentService
from research.experiment_store import ExperimentStore


def _service(tmp_path) -> ExperimentService:
    return ExperimentService(ExperimentStore(tmp_path / "experiments.db"))


def _portfolio_result() -> dict[str, object]:
    return {
        "metrics": {
            "final_equity": 1_050_000,
            "total_return": 0.05,
            "net_total_return": 0.048,
            "net_CAGR": 0.12,
            "max_drawdown": -0.04,
            "sharpe_ratio": 1.2,
            "cost_impact": 0.002,
            "total_trades": 1,
        },
        "portfolio_equity_curve": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "gross_total_equity": [1_000_000, 1_052_000],
                "total_equity": [1_000_000, 1_050_000],
                "drawdown": [0, 0],
                "cash": [1_000_000, 400_000],
                "market_value": [0, 650_000],
                "positions_count": [0, 1],
            }
        ),
        "trades": pd.DataFrame(
            {
                "code": ["000001"],
                "name": ["Sample"],
                "strategy_name": ["ensemble_strategy"],
                "buy_date": [pd.Timestamp("2024-01-02")],
                "sell_date": [pd.Timestamp("2024-01-10")],
                "net_return": [0.05],
                "total_cost": [25.0],
            }
        ),
        "orders": pd.DataFrame({"code": ["000001"], "side": ["BUY"], "status": ["FILLED"]}),
    }


def test_save_portfolio_backtest_result_records_snapshot_and_git_commit(tmp_path):
    service = _service(tmp_path)

    experiment_id = service.save_portfolio_backtest_result(
        name="Portfolio v1",
        result=_portfolio_result(),
        start_date="2024-01-01",
        end_date="2024-01-31",
        config={"position_config": {"initial_cash": 1_000_000}},
        tags=["portfolio"],
        data_snapshot_info={"stock_count": 500, "daily_rows": 1000},
    )
    detail = service.build_experiment_detail(experiment_id)

    assert detail["run"]["experiment_type"] == "portfolio_backtest"
    assert detail["run"]["strategy_name"] == "ensemble_strategy"
    assert detail["run"]["git_commit"]
    assert detail["run"]["data_snapshot_info"]["stock_count"] == 500
    assert len(detail["equity_curve"]) == 2
    assert len(detail["trades"]) == 1
    assert len(detail["orders"]) == 1


def test_save_single_strategy_backtest_result_uses_trade_details_and_equity_curve(tmp_path):
    service = _service(tmp_path)
    result = {
        "CAGR": 0.08,
        "max_drawdown": -0.05,
        "sharpe_ratio": 1.1,
        "trade_details": pd.DataFrame(
            {
                "code": ["000001"],
                "strategy_name": ["trend_strength_strategy"],
                "buy_date": [pd.Timestamp("2024-01-02")],
                "exit_date": [pd.Timestamp("2024-01-10")],
                "return": [0.04],
            }
        ),
        "equity_curve": pd.DataFrame({"date": pd.to_datetime(["2024-01-10"]), "equity": [1.04], "drawdown": [0.0]}),
    }

    experiment_id = service.save_single_strategy_backtest_result(
        name="Trend single",
        strategy_name="trend_strength_strategy",
        result=result,
        start_date="2024-01-01",
        end_date="2024-01-31",
        config={"strategy_config": {"min_amount": 200_000_000}},
    )
    detail = service.build_experiment_detail(experiment_id)

    assert detail["run"]["experiment_type"] == "single_strategy_backtest"
    assert detail["run"]["metrics_json"]["CAGR"] == 0.08
    assert len(detail["equity_curve"]) == 1
    assert len(detail["trades"]) == 1


def test_save_optimizer_and_leaderboard_results(tmp_path):
    service = _service(tmp_path)
    optimizer = pd.DataFrame({"strategy_name": ["trend_strength_strategy"], "annual_return": [0.1], "sharpe_ratio": [1.5]})
    leaderboard = pd.DataFrame({"strategy": ["趋势强势股策略"], "net_annual_return": [0.1], "rating": ["B"]})

    optimizer_id = service.save_optimizer_result(
        name="Optimizer",
        result=optimizer,
        strategy_name="trend_strength_strategy",
        start_date="2024-01-01",
        end_date="2024-01-31",
        config={"optimizer_config": {"top_n": 20}},
    )
    leaderboard_id = service.save_leaderboard_snapshot(
        name="Leaderboard",
        leaderboard=leaderboard,
        as_of="2024-01-31",
        config={"periods": ["1个月"]},
    )

    assert service.build_experiment_detail(optimizer_id)["run"]["experiment_type"] == "parameter_optimization"
    assert service.build_experiment_detail(leaderboard_id)["run"]["experiment_type"] == "strategy_leaderboard"


def test_recent_best_and_failed_experiment_queries(tmp_path):
    service = _service(tmp_path)
    first = service.store.create_experiment_run(
        name="Best",
        experiment_type="portfolio_backtest",
        metrics={"net_CAGR": 0.2},
        status="completed",
    )
    service.store.create_experiment_run(
        name="Failed",
        experiment_type="portfolio_backtest",
        metrics={},
        status="failed",
    )

    recent = service.get_recent_experiments(limit=2)
    best = service.get_best_experiments(metric="net_CAGR", limit=1)
    failed = service.get_failed_or_empty_experiments()

    assert first in set(recent["id"])
    assert best.iloc[0]["id"] == first
    assert len(failed) == 1


def test_save_portfolio_result_with_empty_tables_is_allowed(tmp_path):
    service = _service(tmp_path)
    result = {
        "metrics": {"final_equity": 1_000_000, "total_trades": 0},
        "portfolio_equity_curve": pd.DataFrame(),
        "trades": pd.DataFrame(),
        "orders": pd.DataFrame(),
    }

    experiment_id = service.save_portfolio_backtest_result(
        name="Empty portfolio",
        result=result,
        start_date="2024-01-01",
        end_date="2024-01-31",
        config={},
    )
    detail = service.build_experiment_detail(experiment_id)

    assert detail["run"]["metrics_json"]["total_trades"] == 0
    assert detail["trades"].empty
    assert detail["orders"].empty


def test_best_experiments_returns_empty_for_missing_metric(tmp_path):
    service = _service(tmp_path)
    service.store.create_experiment_run(name="No metric", experiment_type="portfolio_backtest", metrics={"net_CAGR": 0.1})

    result = service.get_best_experiments(metric="missing_metric")

    assert result.empty
