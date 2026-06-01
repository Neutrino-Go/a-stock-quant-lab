from __future__ import annotations

from research.experiment_store import ExperimentStore
from research.report_exporter import export_markdown_report


def test_stability_markdown_report_contains_stability_sections(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = store.create_experiment_run(
        name="Stability",
        experiment_type="strategy_stability_analysis",
        strategy_name="trend_strength_strategy",
        metrics={
            "stability_score": 42,
            "overfit_risk": "HIGH",
            "overfit_score": 82,
            "recommendation": "reject_for_now",
            "in_sample_result": {"net_CAGR": 0.30},
            "out_of_sample_result": {"net_CAGR": -0.05},
            "walk_forward_result": {"summary": {"positive_test_fold_ratio": 0.25}},
            "sensitivity_result": {"sensitivity_score": 25},
            "market_regime_breakdown": {"BULL": {"trade_count": 3}},
        },
    )

    path = export_markdown_report(store, experiment_id, output_dir=tmp_path / "reports")
    content = path.read_text(encoding="utf-8")

    assert "## 稳定性分析" in content
    assert "样本外表现比样本内表现更重要" in content
    assert "该策略存在较高过拟合风险，不建议直接用于实盘" in content
    assert "## Walk-forward 结果" in content


def test_non_stability_markdown_report_keeps_risk_disclaimer(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = store.create_experiment_run(name="Portfolio", experiment_type="portfolio_backtest")

    path = export_markdown_report(store, experiment_id, output_dir=tmp_path / "reports")
    content = path.read_text(encoding="utf-8")

    assert "## 风险提示" in content
    assert "本工具不支持自动交易" in content
