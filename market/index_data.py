from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.database import get_engine
from market.regime_engine import classify_market_regime


HS300_INDEX_CODE = "000300"
HS300_AK_SYMBOL = "sh000300"


def fetch_hs300_index_daily(
    start_date: str = "20100101",
    end_date: str | None = None,
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run `pip install -r requirements.txt`.") from exc

    end_date = end_date or date.today().strftime("%Y%m%d")
    raw = ak.stock_zh_index_daily(symbol=HS300_AK_SYMBOL)
    normalized = normalize_index_prices(raw, index_code=HS300_INDEX_CODE)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return normalized[normalized["date"].between(start_ts, end_ts)].reset_index(drop=True)


def normalize_index_prices(raw: pd.DataFrame, index_code: str = HS300_INDEX_CODE) -> pd.DataFrame:
    if raw.empty:
        return _empty_index_frame()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw.rename(columns=rename_map).copy()
    df["index_code"] = index_code
    required = {"index_code", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected AKShare index columns, missing: {sorted(missing)}")
    if "amount" not in df.columns:
        df["amount"] = 0.0
    result = df.loc[:, ["index_code", "date", "open", "high", "low", "close", "volume", "amount"]].copy()
    result["date"] = pd.to_datetime(result["date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def init_index_table(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS index_prices (
                    index_code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (index_code, date)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_index_prices_date
                ON index_prices(date)
                """
            )
        )


def upsert_index_prices(df: pd.DataFrame, engine: Engine | None = None) -> int:
    normalized = normalize_index_prices(df) if "index_code" not in df.columns else df.copy()
    if normalized.empty:
        return 0
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    engine = engine or get_engine()
    init_index_table(engine)
    sql = text(
        """
        INSERT INTO index_prices(
            index_code, date, open, high, low, close, volume, amount, updated_at
        )
        VALUES (
            :index_code, :date, :open, :high, :low, :close, :volume, :amount, CURRENT_TIMESTAMP
        )
        ON CONFLICT(index_code, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            amount = excluded.amount,
            updated_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, normalized.to_dict("records"))
    return len(normalized)


def get_index_prices(
    index_code: str = HS300_INDEX_CODE,
    start_date: str | None = None,
    end_date: str | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    engine = engine or get_engine()
    init_index_table(engine)
    conditions = ["index_code = :index_code"]
    params: dict[str, Any] = {"index_code": index_code}
    if start_date:
        conditions.append("date >= :start_date")
        params["start_date"] = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    if end_date:
        conditions.append("date <= :end_date")
        params["end_date"] = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    query = text(
        f"""
        SELECT index_code, date, open, high, low, close, volume, amount
        FROM index_prices
        WHERE {' AND '.join(conditions)}
        ORDER BY date
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=params, parse_dates=["date"])


def update_hs300_index(
    start_date: str = "20100101",
    end_date: str | None = None,
    engine: Engine | None = None,
) -> int:
    prices = fetch_hs300_index_daily(start_date=start_date, end_date=end_date)
    return upsert_index_prices(prices, engine=engine)


def calculate_index_factors(index_prices: pd.DataFrame) -> pd.DataFrame:
    return classify_market_regime(index_prices)


def _empty_index_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["index_code", "date", "open", "high", "low", "close", "volume", "amount"])
