from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from config import DAILY_COLUMNS, DEFAULT_ADJUST, DEFAULT_START_DATE
from utils.network import no_proxy_requests_context


class DataFetchError(RuntimeError):
    """User-facing data fetch failure with the original exception attached."""


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
) -> pd.DataFrame:
    ak = _akshare()
    end_date = end_date or date.today().strftime("%Y%m%d")
    normalized_code = _normalize_code(code)
    try:
        with no_proxy_requests_context():
            raw = ak.stock_zh_a_hist(
                symbol=normalized_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
    except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise DataFetchError(
            "东方财富接口访问失败，可能是系统代理/VPN导致。请关闭代理或使用 "
            "scripts/windows/start_streamlit_no_proxy.bat 启动。"
        ) from exc
    return normalize_akshare_daily(raw, normalized_code, name or "")


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
