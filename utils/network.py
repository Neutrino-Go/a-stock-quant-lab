from __future__ import annotations

import os
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator

import requests


PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")
EASTMONEY_HEALTHCHECK_URL = "https://push2his.eastmoney.com"
REACHABLE_STATUS_CODES = {200, 403, 404}


def disable_proxy_env_for_process() -> None:
    """Disable common proxy env vars for the current Python process only."""
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)
    for name in NO_PROXY_ENV_VARS:
        os.environ[name] = "*"


def get_proxy_diagnostics() -> dict[str, Any]:
    """Return proxy-related diagnostics without raising on Windows registry quirks."""
    env_values = {name: os.environ.get(name, "") for name in (*PROXY_ENV_VARS, *NO_PROXY_ENV_VARS)}
    try:
        urllib_proxies = urllib.request.getproxies()
    except Exception as exc:  # pragma: no cover - platform-specific registry behavior
        urllib_proxies = {"error": f"{type(exc).__name__}: {exc}"}
    proxy_keys = {str(key).lower() for key in urllib_proxies if str(key).lower() not in {"no", "no_proxy"}}
    return {
        "proxy_env": env_values,
        "urllib_proxies": urllib_proxies,
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
        "no_proxy": os.environ.get("no_proxy", ""),
        "system_proxy_detected": bool(proxy_keys),
    }


@contextmanager
def no_proxy_requests_context() -> Iterator[None]:
    """Temporarily force requests/AKShare calls to skip environment proxies.

    AKShare uses requests internally for EastMoney endpoints. On Windows, requests can
    still merge proxy settings from the environment during a session call. This context
    clears proxy env vars and temporarily patches Session.merge_environment_settings so
    the effective proxies dict is empty. Everything is restored on exit.
    """
    tracked_names = (*PROXY_ENV_VARS, *NO_PROXY_ENV_VARS)
    saved_env = {name: os.environ.get(name) for name in tracked_names}
    original_merge = requests.sessions.Session.merge_environment_settings

    def merge_without_proxy(self, url, proxies, stream, verify, cert):  # noqa: ANN001
        settings = original_merge(self, url, {}, stream, verify, cert)
        settings["proxies"] = {}
        return settings

    try:
        disable_proxy_env_for_process()
        requests.sessions.Session.merge_environment_settings = merge_without_proxy
        yield
    finally:
        requests.sessions.Session.merge_environment_settings = original_merge
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_eastmoney_connectivity(timeout: int = 10) -> dict[str, Any]:
    """Check whether EastMoney is reachable without proxies.

    EastMoney may return 403/404 for the root host, but those statuses still prove the
    network path can reach the server. ProxyError is treated separately because that is
    the novice-user failure mode we want to diagnose.
    """
    try:
        with no_proxy_requests_context():
            response = requests.get(EASTMONEY_HEALTHCHECK_URL, timeout=timeout)
        status_code = int(response.status_code)
        return {
            "ok": status_code in REACHABLE_STATUS_CODES,
            "status_code": status_code,
            "error_type": "",
            "message": "EastMoney is reachable.",
        }
    except requests.exceptions.ProxyError as exc:
        return {
            "ok": False,
            "status_code": None,
            "error_type": "ProxyError",
            "message": str(exc),
        }
    except requests.exceptions.Timeout as exc:
        return {
            "ok": False,
            "status_code": None,
            "error_type": "Timeout",
            "message": str(exc),
        }
    except requests.exceptions.ConnectionError as exc:
        return {
            "ok": False,
            "status_code": None,
            "error_type": "ConnectionError",
            "message": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {
            "ok": False,
            "status_code": None,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
