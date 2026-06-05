from __future__ import annotations

import os

import requests

from utils.network import disable_proxy_env_for_process, get_proxy_diagnostics, no_proxy_requests_context


def test_disable_proxy_env_for_process_clears_proxy_env(monkeypatch):
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        monkeypatch.setenv(name, "http://127.0.0.1:7890")

    disable_proxy_env_for_process()

    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        assert name not in os.environ
    assert os.environ["NO_PROXY"] == "*"
    assert os.environ["no_proxy"] == "*"


def test_no_proxy_requests_context_restores_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "localhost")

    with no_proxy_requests_context():
        assert "HTTP_PROXY" not in os.environ
        assert os.environ["NO_PROXY"] == "*"
        assert os.environ["no_proxy"] == "*"

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["NO_PROXY"] == "localhost"


def test_no_proxy_requests_context_restores_merge_patch():
    original_merge = requests.sessions.Session.merge_environment_settings

    with no_proxy_requests_context():
        assert requests.sessions.Session.merge_environment_settings is not original_merge

    assert requests.sessions.Session.merge_environment_settings is original_merge


def test_get_proxy_diagnostics_does_not_crash(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

    diagnostics = get_proxy_diagnostics()

    assert "proxy_env" in diagnostics
    assert diagnostics["proxy_env"]["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert "urllib_proxies" in diagnostics
    assert "system_proxy_detected" in diagnostics
