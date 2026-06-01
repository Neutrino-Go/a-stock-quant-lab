from __future__ import annotations

import json

import pandas as pd

from research.experiment_store import ExperimentStore


def _store(tmp_path) -> ExperimentStore:
    return ExperimentStore(tmp_path / "experiments.db")


def test_create_experiment_record_and_list_detail(tmp_path):
    store = _store(tmp_path)

    experiment_id = store.create_experiment_run(
        name="Portfolio smoke",
        experiment_type="portfolio_backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        strategy_name="ensemble_strategy",
        tags=["smoke", "portfolio"],
        config={"position_config": {"initial_cash": 1000000}},
        metrics={"net_CAGR": 0.12, "max_drawdown": -0.08},
        git_commit="abc123",
        data_snapshot_info={"stock_count": 10},
    )

    experiments = store.list_experiments()
    detail = store.get_experiment_detail(experiment_id)

    assert len(experiments) == 1
    assert experiments.iloc[0]["id"] == experiment_id
    assert detail["run"]["tags"] == ["smoke", "portfolio"]
    assert detail["run"]["config_json"]["position_config"]["initial_cash"] == 1000000
    assert detail["run"]["metrics_json"]["net_CAGR"] == 0.12


def test_save_metrics_equity_trades_and_orders(tmp_path):
    store = _store(tmp_path)
    experiment_id = store.create_experiment_run(name="Run", experiment_type="portfolio_backtest")

    store.save_metrics(experiment_id, {"final_equity": 1_100_000, "net_CAGR": 0.1})
    store.save_equity_curve(
        experiment_id,
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "gross_total_equity": [1_000_000, 1_105_000],
                "total_equity": [1_000_000, 1_100_000],
                "drawdown": [0.0, 0.0],
                "cash": [1_000_000, 500_000],
                "market_value": [0, 600_000],
                "positions_count": [0, 1],
            }
        ),
    )
    store.save_trades(
        experiment_id,
        pd.DataFrame(
            {
                "code": ["000001"],
                "name": ["Sample"],
                "strategy_name": ["ensemble_strategy"],
                "signal_date": [pd.Timestamp("2024-01-01")],
                "buy_date": [pd.Timestamp("2024-01-02")],
                "buy_price": [10.0],
                "sell_date": [pd.Timestamp("2024-01-10")],
                "sell_price": [11.0],
                "shares": [1000],
                "gross_return": [0.1],
                "net_return": [0.09],
                "holding_days": [7],
                "exit_reason": ["take_profit"],
                "commission": [10.0],
                "stamp_tax": [5.0],
                "slippage_cost": [11.0],
                "total_cost": [26.0],
            }
        ),
    )
    store.save_orders(
        experiment_id,
        pd.DataFrame(
            {
                "code": ["000001"],
                "name": ["Sample"],
                "side": ["BUY"],
                "signal_date": [pd.Timestamp("2024-01-01")],
                "order_date": [pd.Timestamp("2024-01-02")],
                "requested_price": [10.0],
                "fill_price": [10.01],
                "shares": [1000],
                "status": ["FILLED"],
                "reason": ["signal"],
                "gross_amount": [10010.0],
                "total_cost": [8.0],
                "net_amount": [10018.0],
            }
        ),
    )

    detail = store.get_experiment_detail(experiment_id)

    assert detail["run"]["metrics_json"]["final_equity"] == 1_100_000
    assert len(detail["equity_curve"]) == 2
    assert len(detail["trades"]) == 1
    assert len(detail["orders"]) == 1


def test_update_notes_tags_rating_and_duplicate_config(tmp_path):
    store = _store(tmp_path)
    experiment_id = store.create_experiment_run(
        name="Editable",
        experiment_type="single_strategy_backtest",
        config={"strategy_config": {"name": "trend_strength_strategy"}},
    )

    store.update_notes(experiment_id, notes="review later", conclusion="keep")
    store.update_tags(experiment_id, ["trend", "watch"])
    store.update_rating(experiment_id, "A")
    config = store.duplicate_experiment_config(experiment_id)
    detail = store.get_experiment_detail(experiment_id)

    assert detail["run"]["notes"] == "review later"
    assert detail["run"]["conclusion"] == "keep"
    assert detail["run"]["tags"] == ["trend", "watch"]
    assert detail["run"]["rating"] == "A"
    assert config["strategy_config"]["name"] == "trend_strength_strategy"


def test_delete_experiment_cascades_related_rows(tmp_path):
    store = _store(tmp_path)
    experiment_id = store.create_experiment_run(name="Delete me", experiment_type="portfolio_backtest")
    store.save_equity_curve(experiment_id, pd.DataFrame({"date": ["2024-01-01"], "total_equity": [1.0]}))
    store.save_trades(experiment_id, pd.DataFrame({"code": ["000001"], "net_return": [0.01]}))
    store.save_orders(experiment_id, pd.DataFrame({"code": ["000001"], "side": ["BUY"], "status": ["FILLED"]}))

    store.delete_experiment(experiment_id)

    assert store.list_experiments().empty
    assert store.get_experiment_detail(experiment_id)["run"] is None


def test_empty_frames_and_json_export_are_safe(tmp_path):
    store = _store(tmp_path)
    experiment_id = store.create_experiment_run(
        name="Empty",
        experiment_type="portfolio_backtest",
        config={"nested": {"value": 1}},
        metrics={},
    )

    store.save_equity_curve(experiment_id, pd.DataFrame())
    store.save_trades(experiment_id, pd.DataFrame())
    store.save_orders(experiment_id, pd.DataFrame())
    payload = store.export_experiment_json(experiment_id)

    decoded = json.loads(payload)
    assert decoded["run"]["id"] == experiment_id
    assert decoded["run"]["config_json"]["nested"]["value"] == 1
    assert decoded["equity_curve"] == []
    assert decoded["trades"] == []
    assert decoded["orders"] == []


def test_create_experiment_rejects_unknown_type_status_and_rating(tmp_path):
    store = _store(tmp_path)

    for kwargs in [
        {"experiment_type": "bad_type"},
        {"experiment_type": "portfolio_backtest", "status": "bad_status"},
        {"experiment_type": "portfolio_backtest", "rating": "Z"},
    ]:
        try:
            store.create_experiment_run(name="Bad", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")


def test_list_experiments_filters_and_artifacts(tmp_path):
    store = _store(tmp_path)
    first = store.create_experiment_run(
        name="Portfolio",
        experiment_type="portfolio_backtest",
        strategy_name="ensemble_strategy",
        rating="A",
    )
    store.create_experiment_run(
        name="Single",
        experiment_type="single_strategy_backtest",
        strategy_name="trend_strength_strategy",
        rating="D",
    )

    store.add_artifact(first, "markdown_report", tmp_path / "report.md")
    filtered = store.list_experiments(
        experiment_type="portfolio_backtest",
        strategy_name="ensemble_strategy",
        rating="A",
    )
    detail = store.get_experiment_detail(first)

    assert list(filtered["id"]) == [first]
    assert detail["artifacts"].iloc[0]["artifact_type"] == "markdown_report"


def test_list_experiments_filters_by_tag_date_and_keyword(tmp_path):
    store = _store(tmp_path)
    first = store.create_experiment_run(
        name="Bull portfolio",
        experiment_type="portfolio_backtest",
        strategy_name="ensemble_strategy",
        description="momentum research",
        notes="watch liquidity",
        conclusion="keep",
        tags=["alpha", "swing"],
        rating="A",
        metrics={"net_CAGR": 0.2},
    )
    store.create_experiment_run(
        name="Bear single",
        experiment_type="single_strategy_backtest",
        strategy_name="trend_strength_strategy",
        description="risk test",
        notes="",
        conclusion="discard",
        tags=["risk"],
        rating="D",
        metrics={"net_CAGR": -0.1},
    )

    first_created = store.get_experiment_detail(first)["run"]["created_at"][:10]
    filtered = store.list_experiments(
        experiment_type="portfolio_backtest",
        strategy_name="ensemble_strategy",
        rating="A",
        tag="alpha",
        created_start=first_created,
        created_end=first_created,
        keyword="liquidity",
    )

    assert list(filtered["id"]) == [first]


def test_delete_missing_experiment_returns_false(tmp_path):
    store = _store(tmp_path)

    assert store.delete_experiment("missing-id") is False
