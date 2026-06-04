from __future__ import annotations

import pandas as pd

from data.database import get_engine, upsert_stock_list
from data_quality.audit import DataQualityAuditor


def test_audit_run_saves_issues_and_summary(tmp_path):
    auditor = DataQualityAuditor(tmp_path / "audit.db")
    prices = pd.DataFrame(
        [
            {"code": "000001", "name": "A", "date": "2024-01-02", "open": 10, "high": 9, "low": 8, "close": 10, "volume": 1, "amount": 1, "turnover": 1, "adjust_type": "qfq"}
        ]
    )

    result = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=prices)

    assert result.status == "completed"
    assert result.summary["total_issues"] >= 1
    assert not auditor.list_issues(run_id=result.run_id).empty


def test_failed_audit_run_records_error(tmp_path):
    class FailingAuditor(DataQualityAuditor):
        def _load_price_data(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("load failed")

    auditor = FailingAuditor(tmp_path / "audit.db")

    result = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03")

    assert result.status == "failed"
    assert "load failed" in result.summary["error"]


def test_audit_issue_filters_by_severity_and_keyword(tmp_path):
    auditor = DataQualityAuditor(tmp_path / "audit.db")
    prices = pd.DataFrame(
        [
            {"code": "000001", "name": "A", "date": "2024-01-02", "open": -1, "high": 9, "low": 8, "close": 10, "volume": 1, "amount": 1, "turnover": 1, "adjust_type": "qfq"}
        ]
    )
    result = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=prices, check_survivorship_bias=False)

    errors = auditor.list_issues(run_id=result.run_id, severity="ERROR", keyword="positive")

    assert not errors.empty
    assert errors["severity"].eq("ERROR").all()


def test_audit_records_survivorship_bias_warning(tmp_path):
    db_path = tmp_path / "audit.db"
    upsert_stock_list(pd.DataFrame({"code": ["000001"], "name": ["A"]}), engine=get_engine(db_path))
    auditor = DataQualityAuditor(db_path)
    prices = pd.DataFrame(
        [
            {"code": "000001", "name": "A", "date": "2024-01-02", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1, "amount": 1, "turnover": 1, "adjust_type": "qfq"}
        ]
    )

    result = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=prices, check_survivorship_bias=True)

    assert any(issue.issue_type == "survivorship_bias_warning" for issue in result.issues)
