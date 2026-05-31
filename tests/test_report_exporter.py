from __future__ import annotations

import json

import pandas as pd

from research.experiment_store import ExperimentStore
from research.report_exporter import export_config_json, export_experiment_csv, export_markdown_report


def _seed(store: ExperimentStore) -> str:
    experiment_id = store.create_experiment_run(
        name="Reportable",
        experiment_type="portfolio_backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        strategy_name="ensemble_strategy",
        notes="watch drawdown",
        conclusion="keep researching",
        tags=["report"],
        rating="B",
        git_commit="abc123",
        config={"strategy_config": {"weights": {"trend": 0.5}}, "cost_config": {"slippage_bps": 5}},
        metrics={
            "final_equity": 1_050_000,
            "total_return": 0.05,
            "net_total_return": 0.048,
            "net_CAGR": 0.12,
            "max_drawdown": -0.04,
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
            "cost_impact": 0.002,
            "commission": 10.0,
            "stamp_tax": 5.0,
            "slippage_cost": 3.0,
            "total_trades": 1,
        },
    )
    store.save_equity_curve(experiment_id, pd.DataFrame({"date": ["2024-01-01"], "total_equity": [1_050_000]}))
    store.save_trades(
        experiment_id,
        pd.DataFrame({"code": ["000001"], "net_return": [0.05], "holding_days": [5], "total_cost": [18.0]}),
    )
    store.save_orders(experiment_id, pd.DataFrame({"code": ["000001"], "side": ["BUY"], "status": ["FILLED"]}))
    return experiment_id


def test_export_markdown_report_contains_required_sections_and_artifact(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = _seed(store)

    path = export_markdown_report(store, experiment_id, output_dir=tmp_path / "reports")
    content = path.read_text(encoding="utf-8")
    detail = store.get_experiment_detail(experiment_id)

    assert "# 实验报告" in content
    assert "## 参数配置" in content
    assert "## 风险提示" in content
    assert "回测结果不代表未来收益" in content
    assert len(detail["artifacts"]) == 1
    assert detail["artifacts"].iloc[0]["artifact_type"] == "markdown_report"


def test_export_csv_for_trades_orders_and_equity(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = _seed(store)

    trades_path = export_experiment_csv(store, experiment_id, "trades", output_dir=tmp_path / "reports")
    orders_path = export_experiment_csv(store, experiment_id, "orders", output_dir=tmp_path / "reports")
    equity_path = export_experiment_csv(store, experiment_id, "equity_curve", output_dir=tmp_path / "reports")

    assert pd.read_csv(trades_path).iloc[0]["code"] == 1
    assert pd.read_csv(orders_path).iloc[0]["side"] == "BUY"
    assert "total_equity" in pd.read_csv(equity_path).columns


def test_export_config_json_can_be_deserialized(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = _seed(store)

    path = export_config_json(store, experiment_id, output_dir=tmp_path / "reports")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["strategy_config"]["weights"]["trend"] == 0.5


def test_report_export_handles_no_trades(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = store.create_experiment_run(
        name="No trades",
        experiment_type="single_strategy_backtest",
        metrics={"total_trades": 0},
    )

    path = export_markdown_report(store, experiment_id, output_dir=tmp_path / "reports")

    assert "total_trades" in path.read_text(encoding="utf-8")


def test_export_csv_rejects_unknown_kind(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = _seed(store)

    try:
        export_experiment_csv(store, experiment_id, "unknown", output_dir=tmp_path / "reports")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unsupported CSV export kind" in str(exc)
    else:
        raise AssertionError("Unknown CSV export kind should fail")


def test_export_missing_experiment_raises_key_error(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")

    try:
        export_markdown_report(store, "missing", output_dir=tmp_path / "reports")
    except KeyError as exc:
        assert "Experiment not found" in str(exc)
    else:
        raise AssertionError("Missing experiment should raise")
