from __future__ import annotations

import pandas as pd

from data_quality.universe import StockUniverseBuilder


def test_min_list_days_excludes_recent_listing(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    stock_list = pd.DataFrame({"code": ["000001", "000002"], "name": ["Old", "New"], "list_date": ["2020-01-01", "2024-12-01"]})

    result = builder.build_universe("2025-01-01", stock_list=stock_list, min_list_days=250, use_current_stock_list_fallback=False)

    assert result.codes == ["000001"]


def test_current_stock_list_fallback_warns_survivorship(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    stock_list = pd.DataFrame({"code": ["000001"], "name": ["Only"]})

    result = builder.build_universe("2025-01-01", stock_list=stock_list, use_current_stock_list_fallback=True)

    assert "current_stock_list_fallback may introduce survivorship bias" in result.warnings


def test_min_amount_uses_only_as_of_or_before(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    stock_list = pd.DataFrame({"code": ["000001", "000002"], "name": ["A", "B"], "list_date": ["2020-01-01", "2020-01-01"]})
    prices = pd.DataFrame(
        {
            "code": ["000001", "000001", "000002"],
            "date": ["2024-01-01", "2026-01-01", "2024-01-01"],
            "amount": [100, 999999, 500],
            "close": [10, 10, 10],
        }
    )

    result = builder.build_universe("2025-01-01", stock_list=stock_list, price_df=prices, min_amount=400, use_current_stock_list_fallback=False)

    assert result.codes == ["000002"]


def test_snapshot_save_and_empty_universe(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    result = builder.build_universe("2025-01-01", stock_list=pd.DataFrame(), use_current_stock_list_fallback=False)
    snapshot_id = builder.save_snapshot(result)

    assert snapshot_id
    assert result.stock_count == 0
    assert result.warnings
    assert not builder.list_snapshots().empty


def test_price_filters_use_latest_available_before_as_of(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    stock_list = pd.DataFrame({"code": ["000001", "000002"], "name": ["A", "B"], "list_date": ["2020-01-01", "2020-01-01"]})
    prices = pd.DataFrame(
        {
            "code": ["000001", "000001", "000002"],
            "date": ["2024-01-01", "2026-01-01", "2024-01-01"],
            "amount": [1000, 1000, 1000],
            "close": [5, 100, 25],
        }
    )

    result = builder.build_universe("2025-01-01", stock_list=stock_list, price_df=prices, min_price=10, max_price=30, use_current_stock_list_fallback=False)

    assert result.codes == ["000002"]


def test_market_and_industry_filters(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    stock_list = pd.DataFrame(
        {
            "code": ["000001", "600000", "300001"],
            "name": ["A", "B", "C"],
            "list_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "market": ["SZ", "SH", "SZ"],
            "industry": ["Bank", "Bank", "Tech"],
        }
    )

    result = builder.build_universe("2025-01-01", stock_list=stock_list, market="SZ", industry="Tech", use_current_stock_list_fallback=False)

    assert result.codes == ["300001"]


def test_snapshot_json_fields_round_trip(tmp_path):
    builder = StockUniverseBuilder(tmp_path / "universe.db")
    result = builder.build_universe(
        "2025-01-01",
        stock_list=pd.DataFrame({"code": ["000001"], "name": ["A"], "list_date": ["2020-01-01"]}),
        use_current_stock_list_fallback=True,
    )
    builder.save_snapshot(result)

    snapshot = builder.list_snapshots().iloc[0]

    assert snapshot["codes_json"] == ["000001"]
    assert snapshot["warnings_json"] == result.warnings
