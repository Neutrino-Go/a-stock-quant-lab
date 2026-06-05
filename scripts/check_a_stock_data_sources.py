from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_result(title: str, result: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def test_eastmoney_kline_endpoint() -> dict[str, Any]:
    from utils.network import no_proxy_requests_context

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": "0.000001",
        "beg": "20200101",
        "end": "20200601",
    }
    try:
        with no_proxy_requests_context():
            response = requests.get(url, params=params, timeout=15)
        return {
            "ok": response.status_code in {200, 403, 404},
            "status_code": response.status_code,
            "text_prefix": response.text[:120],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def test_provider(provider: str) -> dict[str, Any]:
    from data.data_fetcher import fetch_stock_daily

    try:
        frame = fetch_stock_daily(
            "000001",
            name="平安银行",
            start_date="20200101",
            end_date="20200601",
            preferred_provider=provider,
        )
        dates = frame["date"] if not frame.empty else []
        return {
            "ok": True,
            "requested_provider": provider,
            "actual_provider": frame.attrs.get("provider", ""),
            "row_count": int(len(frame)),
            "first_date": str(dates.min().date()) if len(frame) else "",
            "last_date": str(dates.max().date()) if len(frame) else "",
            "warnings": frame.attrs.get("warnings", []),
        }
    except Exception as exc:
        provider_errors = getattr(exc, "provider_errors", [])
        return {
            "ok": False,
            "requested_provider": provider,
            "error": f"{type(exc).__name__}: {exc}",
            "provider_errors": provider_errors,
        }


def main() -> int:
    from data.data_fetcher import is_daily_provider_available

    _print_result("EastMoney K-line endpoint", test_eastmoney_kline_endpoint())
    _print_result("AKShare provider: eastmoney", test_provider("eastmoney"))
    _print_result("AKShare provider: auto", test_provider("auto"))
    if is_daily_provider_available("sina"):
        _print_result("AKShare provider: sina", test_provider("sina"))
    else:
        _print_result("AKShare provider: sina", {"ok": False, "error": "stock_zh_a_daily unavailable"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
