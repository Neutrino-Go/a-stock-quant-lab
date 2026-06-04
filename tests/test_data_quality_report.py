from __future__ import annotations

import pandas as pd

from data_quality.audit import DataQualityAuditor
from data_quality.report_exporter import export_data_quality_issues_csv, export_data_quality_markdown_report


def test_data_quality_report_and_csv_export(tmp_path):
    auditor = DataQualityAuditor(tmp_path / "report.db")
    result = auditor.run_audit("daily_prices", "2024-01-01", "2024-01-03", price_df=pd.DataFrame())

    md_path = export_data_quality_markdown_report(auditor, result.run_id, output_dir=tmp_path)
    csv_path = export_data_quality_issues_csv(auditor.list_issues(result.run_id), tmp_path / "issues.csv")

    assert md_path.exists()
    assert csv_path.exists()
    assert "当前股票列表 fallback 可能引入幸存者偏差" in md_path.read_text(encoding="utf-8")
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_empty_issues_csv_has_headers(tmp_path):
    path = export_data_quality_issues_csv(pd.DataFrame(), tmp_path / "empty.csv")

    content = path.read_text(encoding="utf-8-sig")

    assert "severity" in content
    assert "issue_type" in content


def test_issues_csv_preserves_six_digit_code(tmp_path):
    issues = pd.DataFrame({"code": [1], "severity": ["WARNING"], "issue_type": ["x"]})

    path = export_data_quality_issues_csv(issues, tmp_path / "issues.csv")

    assert "000001" in path.read_text(encoding="utf-8-sig")
