from __future__ import annotations

import http.client

import pandas as pd
import pytest

import data.data_fetcher as fetcher
import data.data_updater as updater
from app import data_fetch_error_technical_detail
from data.data_fetcher import DataFetchError, fetch_stock_daily, normalize_sina_daily
from data.data_updater import DataUpdateService


def _eastmoney_raw(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": [10.0 + index for index, _ in enumerate(dates)],
            "最高": [11.0 + index for index, _ in enumerate(dates)],
            "最低": [9.0 + index for index, _ in enumerate(dates)],
            "收盘": [10.5 + index for index, _ in enumerate(dates)],
            "成交量": [1000 + index for index, _ in enumerate(dates)],
            "成交额": [100000.0 + index for index, _ in enumerate(dates)],
            "换手率": [1.0 + index for index, _ in enumerate(dates)],
        }
    )


def _sina_raw(dates: list[str], include_amount: bool = True, include_turnover: bool = True) -> pd.DataFrame:
    data = {
        "date": dates,
        "open": [10.0 + index for index, _ in enumerate(dates)],
        "high": [11.0 + index for index, _ in enumerate(dates)],
        "low": [9.0 + index for index, _ in enumerate(dates)],
        "close": [10.5 + index for index, _ in enumerate(dates)],
        "volume": [1000 + index for index, _ in enumerate(dates)],
    }
    if include_amount:
        data["amount"] = [100000.0 + index for index, _ in enumerate(dates)]
    if include_turnover:
        data["turnover"] = [1.0 + index for index, _ in enumerate(dates)]
    return pd.DataFrame(data)


def test_eastmoney_success_sets_provider_and_unified_columns(monkeypatch):
    class FakeAk:
        def stock_zh_a_hist(self, **kwargs):
            return _eastmoney_raw(["2020-01-02", "2020-01-03"])

    monkeypatch.setattr(fetcher, "_akshare", lambda: FakeAk())

    result = fetch_stock_daily(
        "1",
        name="平安银行",
        start_date="20200101",
        end_date="20200110",
        preferred_provider="eastmoney",
    )

    assert result.attrs["provider"] == "eastmoney"
    assert result["code"].tolist() == ["000001", "000001"]
    assert list(result.columns) == [
        "code",
        "name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ]


def test_auto_falls_back_to_sina_after_eastmoney_remote_disconnect(monkeypatch):
    class FakeAk:
        def stock_zh_a_hist(self, **kwargs):
            raise http.client.RemoteDisconnected("remote closed")

        def stock_zh_a_daily(self, **kwargs):
            return _sina_raw(["2020-01-02", "2020-01-03"])

    monkeypatch.setattr(fetcher, "_akshare", lambda: FakeAk())

    result = fetch_stock_daily(
        "000001",
        name="平安银行",
        start_date="20200101",
        end_date="20200110",
        preferred_provider="auto",
    )

    assert result.attrs["provider"] == "sina"
    assert len(result) == 2


def test_auto_raises_clear_error_when_fallback_is_missing(monkeypatch):
    class FakeAk:
        def stock_zh_a_hist(self, **kwargs):
            raise http.client.RemoteDisconnected("remote closed")

    monkeypatch.setattr(fetcher, "_akshare", lambda: FakeAk())

    with pytest.raises(DataFetchError) as exc_info:
        fetch_stock_daily(
            "000001",
            start_date="20200101",
            end_date="20200110",
            preferred_provider="auto",
        )

    assert "备用数据源" in str(exc_info.value)
    assert any("stock_zh_a_daily unavailable" in item for item in exc_info.value.provider_errors)


def test_eastmoney_yearly_chunks_merge_deduplicate_and_sort(monkeypatch):
    class FakeAk:
        def stock_zh_a_hist(self, **kwargs):
            start_date = kwargs["start_date"]
            end_date = kwargs["end_date"]
            if start_date == "20200101" and end_date == "20211231":
                raise http.client.RemoteDisconnected("remote closed")
            if start_date == "20200101" and end_date == "20201231":
                return _eastmoney_raw(["2020-01-03", "2020-01-02", "2020-01-02"])
            if start_date == "20210101" and end_date == "20211231":
                return _eastmoney_raw(["2021-01-04"])
            return pd.DataFrame()

    monkeypatch.setattr(fetcher, "_akshare", lambda: FakeAk())

    result = fetch_stock_daily(
        "000001",
        start_date="20200101",
        end_date="20211231",
        preferred_provider="eastmoney",
    )

    assert result.attrs["provider"] == "eastmoney"
    assert "tried yearly chunks" in result.attrs["warnings"][0]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-02",
        "2020-01-03",
        "2021-01-04",
    ]


def test_sina_missing_amount_and_turnover_records_warnings():
    result, warnings = normalize_sina_daily(
        _sina_raw(["2020-01-02"], include_amount=False, include_turnover=False),
        code="1",
        name="平安银行",
    )

    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "amount"] == 0.0
    assert result.loc[0, "turnover"] == 0.0
    assert "sina provider missing amount; filled with 0" in warnings
    assert "sina provider missing turnover; filled with 0" in warnings


def test_update_stock_daily_returns_provider_and_warnings(monkeypatch):
    service = object.__new__(DataUpdateService)
    service.engine = object()

    daily = _eastmoney_raw(["2020-01-02"])
    normalized = fetcher.normalize_akshare_daily(daily, code="000001", name="平安银行")
    normalized.attrs["provider"] = "sina"
    normalized.attrs["warnings"] = ["sina provider missing amount; filled with 0"]

    monkeypatch.setattr(updater, "get_latest_trade_date", lambda code, engine: None)
    monkeypatch.setattr(updater, "get_stock_name", lambda code, engine: "平安银行")
    monkeypatch.setattr(updater, "fetch_stock_daily", lambda **kwargs: normalized)
    monkeypatch.setattr(updater, "upsert_stock_daily", lambda frame, engine: len(frame))

    result = service.update_stock_daily(
        "000001",
        start_date="20200101",
        end_date="20200110",
        preferred_provider="auto",
    )

    assert result.rows == 1
    assert result.provider == "sina"
    assert result.warnings == ("sina provider missing amount; filled with 0",)


def test_app_data_fetch_error_detail_does_not_expose_traceback():
    exc = DataFetchError(
        "friendly error",
        provider_errors=["eastmoney: RemoteDisconnected: remote closed", "sina: unavailable"],
    )

    detail = data_fetch_error_technical_detail(exc)

    assert "Traceback" not in detail
    assert "eastmoney: RemoteDisconnected" in detail
    assert "sina: unavailable" in detail
