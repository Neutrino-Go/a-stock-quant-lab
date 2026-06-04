from __future__ import annotations

import pandas as pd

from data_quality.audit import DataQualityAuditor
from data_quality.data_health_service import DataHealthService


def test_health_summary_and_snapshot_info(tmp_path):
    auditor = DataQualityAuditor(tmp_path / "health.db")
    prices = pd.DataFrame(
        [{"code": "000001", "name": "A", "date": "2024-01-02", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1, "amount": 1, "turnover": 1, "adjust_type": "qfq"}]
    )
    run = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=prices)
    service = DataHealthService(tmp_path / "health.db")

    summary = service.get_database_health_summary(price_df=prices, stock_list=pd.DataFrame({"code": ["000001"]}))
    snapshot = service.build_data_snapshot_info(price_df=prices, stock_list=pd.DataFrame({"code": ["000001"]}))

    assert summary["stock_count"] == 1
    assert snapshot["latest_quality_run_id"] == run.run_id
    assert "warning_issue_count" in snapshot


def test_stale_stocks_and_adjust_summary(tmp_path):
    service = DataHealthService(tmp_path / "health.db")
    prices = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "date": ["2024-01-01", "2024-01-10"],
            "adjust_type": ["qfq", "raw"],
        }
    )

    stale = service.get_stale_stocks(max_lag_days=5, price_df=prices, latest_date="2024-01-10")
    adjust = service.get_adjust_type_summary(price_df=prices)

    assert stale["code"].tolist() == ["000001"]
    assert set(adjust["adjust_type"]) == {"qfq", "raw"}


def test_issue_summary_by_type(tmp_path):
    auditor = DataQualityAuditor(tmp_path / "health.db")
    prices = pd.DataFrame(
        [{"code": "000001", "name": "A", "date": "2024-01-02", "open": -1, "high": 11, "low": 9, "close": 10, "volume": 1, "amount": 1, "turnover": 1, "adjust_type": "qfq"}]
    )
    auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=prices, check_survivorship_bias=False)

    summary = DataHealthService(tmp_path / "health.db").get_issue_summary_by_type()

    assert "negative_price" in summary["issue_type"].tolist()


def test_coverage_by_code_counts_rows(tmp_path):
    service = DataHealthService(tmp_path / "health.db")
    prices = pd.DataFrame({"code": ["000001", "000001", "000002"], "date": ["2024-01-01", "2024-01-02", "2024-01-02"]})

    coverage = service.get_data_coverage_by_code(price_df=prices)

    assert coverage.set_index("code").loc["000001", "row_count"] == 2


def test_snapshot_warns_when_no_quality_run(tmp_path):
    service = DataHealthService(tmp_path / "health.db")

    snapshot = service.build_data_snapshot_info(price_df=pd.DataFrame({"date": ["2024-01-01"], "code": ["000001"]}))

    assert "no data_quality_run available" in snapshot["universe_warnings"]


def test_snapshot_does_not_default_legacy_daily_price_to_qfq(tmp_path):
    service = DataHealthService(tmp_path / "health.db")

    snapshot = service.build_data_snapshot_info(
        adjust_type="qfq",
        price_df=pd.DataFrame({"date": ["2024-01-01"], "code": ["000001"], "close": [10.0]}),
    )

    assert snapshot["adjust_type"] == "unknown"
    assert snapshot["actual_adjust_type"] == "unknown"
    assert snapshot["requested_adjust_type"] == "qfq"
    assert snapshot["assumed_adjust_type"] == "qfq"
    assert "legacy daily_price has no adjust_type; backtest adjust type is an assumption" in snapshot["universe_warnings"]
