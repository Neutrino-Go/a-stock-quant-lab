from __future__ import annotations

import http.client
from datetime import date
from typing import Any

import pandas as pd
import requests

from config import DAILY_COLUMNS, DEFAULT_ADJUST, DEFAULT_START_DATE
from utils.network import no_proxy_requests_context


class DataFetchError(RuntimeError):
    """User-facing data fetch failure with the original exception attached."""

    def __init__(self, message: str, provider_errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.provider_errors = provider_errors or []


SUPPORTED_DAILY_PROVIDERS = {"auto", "eastmoney", "sina"}
RETRYABLE_DATA_ERRORS = (
    requests.exceptions.ProxyError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    http.client.RemoteDisconnected,
)


def _akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "akshare is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return ak


def _normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)


def fetch_stock_list() -> pd.DataFrame:
    ak = _akshare()
    raw = ak.stock_info_a_code_name()
    rename_map = {
        "代码": "code",
        "证券代码": "code",
        "股票代码": "code",
        "名称": "name",
        "证券简称": "name",
        "股票简称": "name",
    }
    df = raw.rename(columns=rename_map).copy()
    if not {"code", "name"}.issubset(df.columns):
        raise ValueError(f"Unexpected AKShare stock list columns: {list(raw.columns)}")

    result = df.loc[:, ["code", "name"]].dropna().copy()
    result["code"] = result["code"].map(_normalize_code)
    result["name"] = result["name"].astype(str).str.strip()
    result = result[result["code"].str.match(r"^\d{6}$")]
    return result.drop_duplicates("code").sort_values("code").reset_index(drop=True)


def fetch_stock_daily(
    code: str,
    name: str | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    adjust: str = DEFAULT_ADJUST,
    preferred_provider: str = "auto",
    eastmoney_retries: int = 2,
) -> pd.DataFrame:
    ak = _akshare()
    end_date = end_date or date.today().strftime("%Y%m%d")
    normalized_code = _normalize_code(code)
    provider = str(preferred_provider or "auto").lower()
    if provider not in SUPPORTED_DAILY_PROVIDERS:
        raise ValueError(f"Unsupported daily data provider: {preferred_provider}")

    provider_errors: list[str] = []
    if provider in {"auto", "eastmoney"}:
        try:
            return _fetch_eastmoney_daily(ak, normalized_code, name or "", start_date, end_date, adjust, retries=eastmoney_retries)
        except RETRYABLE_DATA_ERRORS as exc:
            provider_errors.append(f"eastmoney: {type(exc).__name__}: {exc}")
            if provider == "eastmoney":
                raise _data_fetch_error(provider_errors) from exc

    if provider in {"auto", "sina"}:
        try:
            return _fetch_sina_daily(ak, normalized_code, name or "", start_date, end_date, adjust)
        except DataFetchError as exc:
            provider_errors.extend(exc.provider_errors)
            raise _data_fetch_error(provider_errors) from exc
        except Exception as exc:
            provider_errors.append(f"sina: {type(exc).__name__}: {exc}")
            raise _data_fetch_error(provider_errors) from exc

    raise _data_fetch_error(provider_errors or [f"{provider}: provider unavailable"])


def is_daily_provider_available(provider: str) -> bool:
    provider = str(provider or "").lower()
    if provider == "eastmoney":
        return hasattr(_akshare(), "stock_zh_a_hist")
    if provider == "sina":
        return hasattr(_akshare(), "stock_zh_a_daily")
    if provider == "auto":
        return is_daily_provider_available("eastmoney") or is_daily_provider_available("sina")
    return False


def _fetch_eastmoney_daily(
    ak: Any,
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    adjust: str,
    retries: int,
) -> pd.DataFrame:
    warnings: list[str] = []
    try:
        raw = _eastmoney_range_with_retries(ak, code, start_date, end_date, adjust, retries)
        result = normalize_akshare_daily(raw, code, name)
        return _finalize_daily_frame(result, provider="eastmoney", adjust=adjust, warnings=warnings)
    except RETRYABLE_DATA_ERRORS as exc:
        warnings.append(f"eastmoney full-range request failed: {type(exc).__name__}: {exc}; tried yearly chunks")
        frames: list[pd.DataFrame] = []
        failed_years: list[str] = []
        for year_start, year_end in _year_ranges(start_date, end_date):
            try:
                frames.append(_eastmoney_range_with_retries(ak, code, year_start, year_end, adjust, retries))
            except RETRYABLE_DATA_ERRORS as year_exc:
                failed_years.append(f"{year_start[:4]}: {type(year_exc).__name__}: {year_exc}")
        if failed_years:
            warnings.extend(f"eastmoney yearly chunk failed: {item}" for item in failed_years)
        if not frames:
            raise exc
        raw = pd.concat(frames, ignore_index=True)
        result = normalize_akshare_daily(raw, code, name)
        return _finalize_daily_frame(result, provider="eastmoney", adjust=adjust, warnings=warnings)


def _eastmoney_range_with_retries(
    ak: Any,
    code: str,
    start_date: str,
    end_date: str,
    adjust: str,
    retries: int,
) -> pd.DataFrame:
    last_error: BaseException | None = None
    for _ in range(max(int(retries), 0) + 1):
        try:
            with no_proxy_requests_context():
                return ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
        except RETRYABLE_DATA_ERRORS as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _fetch_sina_daily(
    ak: Any,
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame:
    if not hasattr(ak, "stock_zh_a_daily"):
        raise DataFetchError("AKShare 当前环境没有可用的新浪 A 股日线接口。", ["sina: stock_zh_a_daily unavailable"])
    raw = ak.stock_zh_a_daily(symbol=_sina_symbol(code), start_date=start_date, end_date=end_date, adjust=_sina_adjust(adjust))
    result, warnings = normalize_sina_daily(raw, code, name)
    return _finalize_daily_frame(result, provider="sina", adjust=adjust, warnings=warnings)


def normalize_sina_daily(raw: pd.DataFrame, code: str, name: str = "") -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if raw.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), warnings
    result = raw.copy()
    result["code"] = code
    result["name"] = name
    if "amount" not in result.columns:
        result["amount"] = 0.0
        warnings.append("sina provider missing amount; filled with 0")
    if "turnover" not in result.columns:
        result["turnover"] = 0.0
        warnings.append("sina provider missing turnover; filled with 0")
    missing = set(DAILY_COLUMNS) - set(result.columns)
    if missing:
        raise DataFetchError(f"新浪日线字段不完整，缺少: {sorted(missing)}", [f"sina: missing columns {sorted(missing)}"])
    result = result.loc[:, DAILY_COLUMNS].copy()
    result["code"] = result["code"].map(_normalize_code)
    result["name"] = result["name"].replace("", name).fillna(name)
    result["date"] = pd.to_datetime(result["date"])
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True), warnings


def _finalize_daily_frame(df: pd.DataFrame, provider: str, adjust: str, warnings: list[str]) -> pd.DataFrame:
    result = df.copy()
    if not result.empty:
        result = result.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    result.attrs["provider"] = provider
    result.attrs["warnings"] = list(warnings)
    result.attrs["adjust_type"] = adjust
    return result


def _data_fetch_error(provider_errors: list[str]) -> DataFetchError:
    details = "；".join(provider_errors)
    return DataFetchError(
        "A股日线数据获取失败。东方财富接口可能被当前网络断开，且备用数据源不可用或也失败。"
        "请在数据中心选择 auto 数据源，或运行 scripts/windows/check_a_stock_data_sources.bat 查看诊断。"
        f" 详情：{details}",
        provider_errors=provider_errors,
    )


def _year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    ranges = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        ranges.append((year_start.strftime("%Y%m%d"), year_end.strftime("%Y%m%d")))
    return ranges


def _sina_symbol(code: str) -> str:
    normalized = _normalize_code(code)
    if normalized.startswith(("6", "9")):
        return f"sh{normalized}"
    if normalized.startswith(("0", "2", "3")):
        return f"sz{normalized}"
    if normalized.startswith(("4", "8")):
        return f"bj{normalized}"
    return normalized


def _sina_adjust(adjust: str) -> str:
    normalized = str(adjust or "").lower()
    return "" if normalized in {"", "raw", "none", "no"} else normalized


def normalize_akshare_daily(raw: pd.DataFrame, code: str, name: str = "") -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
        "股票代码": "code",
        "代码": "code",
        "名称": "name",
    }
    df = raw.rename(columns=rename_map).copy()
    df["code"] = df.get("code", code)
    df["name"] = df.get("name", name)

    missing = set(DAILY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected AKShare daily columns, missing: {sorted(missing)}")

    result = df.loc[:, DAILY_COLUMNS].copy()
    result["code"] = result["code"].map(_normalize_code)
    result["name"] = result["name"].replace("", name).fillna(name)
    result["date"] = pd.to_datetime(result["date"])
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
