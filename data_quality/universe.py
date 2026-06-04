from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH
from data.database import get_engine, get_stock_daily, get_stock_list
from data_quality.models import StockUniverseResult, connect, date_iso, json_dumps, json_loads, normalize_code, now_iso


SURVIVORSHIP_WARNING = "current_stock_list_fallback may introduce survivorship bias"


class StockUniverseBuilder:
    def __init__(self, db_path: str | Path = DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path).close()

    def build_universe(
        self,
        as_of_date: str,
        universe_name: str = "default",
        min_list_days: int = 250,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        exclude_delisted: bool = True,
        min_amount: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        market: str | None = None,
        industry: str | None = None,
        use_current_stock_list_fallback: bool = True,
        stock_list: pd.DataFrame | None = None,
        price_df: pd.DataFrame | None = None,
    ) -> StockUniverseResult:
        as_of = pd.to_datetime(as_of_date).date()
        warnings: list[str] = []
        bias_flags: list[str] = []
        stocks = stock_list.copy() if isinstance(stock_list, pd.DataFrame) else self._load_stock_list()
        if stocks.empty:
            warnings.append("empty stock universe input")
        elif use_current_stock_list_fallback:
            warnings.append(SURVIVORSHIP_WARNING)
            bias_flags.append("survivorship_bias_warning")
        if "list_date" not in stocks.columns:
            warnings.append("missing historical list_date; min_list_days filter may be incomplete")
            stocks["list_date"] = "1900-01-01"
        warnings.append("historical ST status may be incomplete; fallback uses available metadata")
        warnings.append("historical suspension/delisting status may be incomplete; fallback uses available metadata")
        if not stocks.empty:
            stocks = stocks.copy()
            stocks["code"] = stocks["code"].map(normalize_code)
            stocks["list_date"] = pd.to_datetime(stocks["list_date"], errors="coerce")
            stocks = stocks.loc[stocks["list_date"].le(pd.Timestamp(as_of) - pd.Timedelta(days=int(min_list_days)))]
            if exclude_st and "is_st" in stocks.columns:
                stocks = stocks.loc[~stocks["is_st"].fillna(False).astype(bool)]
            if exclude_delisted and "is_delisted" in stocks.columns:
                stocks = stocks.loc[~stocks["is_delisted"].fillna(False).astype(bool)]
            if exclude_suspended and "is_suspended" in stocks.columns:
                stocks = stocks.loc[~stocks["is_suspended"].fillna(False).astype(bool)]
            if market and "market" in stocks.columns:
                stocks = stocks.loc[stocks["market"].eq(market)]
            if industry and "industry" in stocks.columns:
                stocks = stocks.loc[stocks["industry"].eq(industry)]
            stocks = self._apply_price_filters(stocks, price_df, as_of_date, min_amount, min_price, max_price)
        codes = sorted(stocks["code"].dropna().astype(str).map(normalize_code).unique().tolist()) if not stocks.empty and "code" in stocks.columns else []
        if not codes:
            warnings.append("universe is empty after filters")
        config = {
            "min_list_days": min_list_days,
            "exclude_st": exclude_st,
            "exclude_suspended": exclude_suspended,
            "exclude_delisted": exclude_delisted,
            "min_amount": min_amount,
            "min_price": min_price,
            "max_price": max_price,
            "market": market,
            "industry": industry,
            "use_current_stock_list_fallback": use_current_stock_list_fallback,
        }
        return StockUniverseResult(date_iso(as_of_date) or str(as_of), universe_name, codes, len(codes), config, warnings, bias_flags, {"source": "current_stock_list_fallback" if use_current_stock_list_fallback else "provided"})

    def save_snapshot(self, result: StockUniverseResult) -> str:
        snapshot_id = str(uuid.uuid4())
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO stock_universe_snapshots(
                    id, snapshot_date, universe_name, as_of_date, config_json,
                    stock_count, codes_json, warnings_json, bias_flags_json,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    now_iso(),
                    result.universe_name,
                    result.as_of_date,
                    json_dumps(result.config),
                    result.stock_count,
                    json_dumps(result.codes),
                    json_dumps(result.warnings),
                    json_dumps(result.bias_flags),
                    json_dumps(result.metadata),
                    now_iso(),
                ),
            )
        return snapshot_id

    def save_snapshot_from_dict(self, result: dict) -> str:
        return self.save_snapshot(StockUniverseResult.from_dict(result))

    def list_snapshots(self) -> pd.DataFrame:
        with connect(self.db_path) as conn:
            frame = pd.read_sql_query("SELECT * FROM stock_universe_snapshots ORDER BY created_at DESC", conn)
        if not frame.empty:
            frame["codes_json"] = frame["codes_json"].map(lambda value: json_loads(value, []))
            frame["warnings_json"] = frame["warnings_json"].map(lambda value: json_loads(value, []))
            frame["config_json"] = frame["config_json"].map(lambda value: json_loads(value, {}))
            if "bias_flags_json" in frame.columns:
                frame["bias_flags_json"] = frame["bias_flags_json"].map(lambda value: json_loads(value, []))
            if "metadata_json" in frame.columns:
                frame["metadata_json"] = frame["metadata_json"].map(lambda value: json_loads(value, {}))
        return frame

    def _load_stock_list(self) -> pd.DataFrame:
        try:
            return get_stock_list(engine=get_engine(self.db_path))
        except Exception:
            return pd.DataFrame()

    def _apply_price_filters(
        self,
        stocks: pd.DataFrame,
        price_df: pd.DataFrame | None,
        as_of_date: str,
        min_amount: float | None,
        min_price: float | None,
        max_price: float | None,
    ) -> pd.DataFrame:
        if min_amount is None and min_price is None and max_price is None:
            return stocks
        prices = price_df.copy() if isinstance(price_df, pd.DataFrame) else pd.DataFrame()
        if prices.empty:
            try:
                prices = get_stock_daily(end_date=as_of_date, engine=get_engine(self.db_path))
            except Exception:
                return stocks
        if prices.empty:
            return stocks.iloc[0:0]
        prices["code"] = prices["code"].map(normalize_code)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        latest = prices.loc[prices["date"].le(pd.to_datetime(as_of_date))].sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
        keep = set(latest["code"])
        if min_amount is not None and "amount" in latest.columns:
            keep &= set(latest.loc[pd.to_numeric(latest["amount"], errors="coerce").ge(float(min_amount)), "code"])
        if min_price is not None and "close" in latest.columns:
            keep &= set(latest.loc[pd.to_numeric(latest["close"], errors="coerce").ge(float(min_price)), "code"])
        if max_price is not None and "close" in latest.columns:
            keep &= set(latest.loc[pd.to_numeric(latest["close"], errors="coerce").le(float(max_price)), "code"])
        return stocks.loc[stocks["code"].isin(keep)]
