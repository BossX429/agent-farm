"""Tests that api_swarm is GET-only by default.

SSRF checks stop private URLs, but they do not stop a model from
making a destructive POST/PUT/DELETE to a public service. Mutating
methods therefore require an explicit per-call opt-in
(allow_mutating=True). These tests use a public URL and check the
gate, not the network: a blocked mutating call must fail before any
request is sent, so its error must be the gate message, not a network
error, and it must not depend on the URL being reachable.
"""

import asyncio
import json

from agent_farm.server import api_swarm


def run(coro):
    return asyncio.run(coro)


def test_api_swarm_blocks_mutating_by_default():
    result = run(api_swarm(json.dumps([
        {"url": "https://example.com/data", "method": "POST", "body": "{}"},
        {"url": "https://example.com/data", "method": "PUT", "body": "{}"},
        {"url": "https://example.com/data", "method": "DELETE"},
    ])))
    for r in result["results"]:
        assert r["status"] == "blocked"
        assert "allow_mutating" in r["error"]


def test_api_swarm_mutating_blocked_even_for_private_url():
    # The mutating gate runs first: a POST to a private URL is still
    # reported as blocked, and no request is attempted.
    result = run(api_swarm(json.dumps([
        {"url": "http://127.0.0.1:8765/ask", "method": "POST", "body": "{}"},
    ])))
    assert result["results"][0]["status"] == "blocked"
    assert "allow_mutating" in result["results"][0]["error"]


def test_api_swarm_get_still_not_gated():
    # A GET to a private URL must still be blocked by the SSRF guard,
    # not by the mutating gate, proving GET is not caught by the gate.
    result = run(api_swarm(json.dumps([
        {"url": "http://127.0.0.1:8765/ask"},
    ])))
    assert result["results"][0]["status"] == "blocked"
    assert "Blocked" in result["results"][0]["error"]
    assert "allow_mutating" not in result["results"][0]["error"]


def test_api_swarm_mutating_allowed_with_opt_in(monkeypatch):
    # With allow_mutating=True, the gate must not block: the request
    # reaches _http_request, which we fake here so no network is used.
    # api_swarm imports _http_request from tools inside the function,
    # so the fake must be patched on tools, not on swarm_tools.
    import agent_farm.tools as tools_module

    calls = []

    def fake_http(method, url, body="", headers=None):
        calls.append((method, url))
        return {"url": url, "status": 200, "body": "ok"}

    monkeypatch.setattr(tools_module, "_http_request", fake_http)
    result = run(api_swarm(json.dumps([
        {"url": "https://example.com/data", "method": "POST", "body": "{}"},
    ]), allow_mutating=True))
    assert result["results"][0]["status"] == "success"
    assert calls == [("POST", "https://example.com/data")]
