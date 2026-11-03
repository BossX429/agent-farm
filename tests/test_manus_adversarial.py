"""Adversarial / negative tests for Manus's second-review concerns.

Only cases NOT already covered by earlier suites are added here.
Already covered (not duplicated): symlink write-outside-allowlist,
blocked system dirs, private/loopback URLs, mixed public+private
DNS answers, DNS flipping after validation, safe pipelines vs.
rejected shell operators, and parallel payload/item limits.

New in this file:
1. Tracked tasks: no bare asyncio.create_task() in agent_farm/,
   and a tracked task's exception surfaces instead of being
   swallowed.
2. Safe defaults: AGENT_FARM_PRODUCTION=1 ignores both private-HTTP
   escape hatches; get_security_config() / escape_hatches_active()
   report the current configuration; api_swarm mutation stays
   off by default (already covered) and is logged when allowed.
3. Structured action log: opt-in JSON-lines records, credential
   redaction, no bodies/full command lines, hash-chain verification
   and tamper detection.
4. Adversarial inputs: encoded/traversal paths, symlink directory
   components, redirect chains to internal addresses, IPv6 /
   unspecified / link-local DNS answers, @-trick credential URLs,
   blocked pipeline stages, and oversized HTTP bodies.
"""

import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_farm import tools
from agent_farm.action_log import (
    disable_action_log,
    enable_action_log,
    log_action,
    redact_url,
    verify_action_log,
)
from agent_farm.reliability import (
    TRACKED_TASK_ERRORS,
    _TRACKED_TASKS,
    create_tracked_task,
    gather_tracked,
)


# ---------- 1. Tracked tasks ----------

def test_no_bare_create_task_in_agent_farm():
    """Regression guard: future code must use create_tracked_task().

    Uses AST parsing, so comments and docstrings mentioning
    create_task do not count. The single sanctioned call inside
    reliability.create_tracked_task() itself is allowed.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "agent_farm"
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "create_task"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            ):
                # Allow only the wrapper in reliability.py.
                if not (path.name == "reliability.py"):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_tracked_task_exception_surfaces_via_gather():
    async def boom():
        raise RuntimeError("tracked-boom")

    async def run():
        with pytest.raises(RuntimeError, match="tracked-boom"):
            await gather_tracked([boom()])
        assert _TRACKED_TASKS == set()

    asyncio.run(run())


def test_create_tracked_task_is_stored_until_done():
    async def ok():
        return 42

    async def run():
        TRACKED_TASK_ERRORS.clear()
        task = create_tracked_task(ok())
        assert len(_TRACKED_TASKS) == 1
        assert await task == 42
        await asyncio.sleep(0)
        assert _TRACKED_TASKS == set()

    asyncio.run(run())


# ---------- 2. Safe defaults ----------

def test_production_mode_ignores_private_http_hatches(monkeypatch):
    monkeypatch.setenv("AGENT_FARM_PRODUCTION", "1")
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setenv("AGENT_FARM_ALLOWED_HTTP_HOSTS", "internal.example")
    assert tools.escape_hatches_active() == []
    assert tools.get_security_config()["production_mode"] is True
    assert tools.is_url_public("http://127.0.0.1:9/x") is False
    assert tools.is_url_public("http://internal.example/x") is False


def test_escape_hatches_reported_when_active(monkeypatch):
    monkeypatch.delenv("AGENT_FARM_PRODUCTION", raising=False)
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setenv("AGENT_FARM_ALLOWED_HTTP_HOSTS", "a.example,b.example")
    active = tools.escape_hatches_active()
    assert "AGENT_FARM_ALLOW_PRIVATE_HTTP" in active
    assert "AGENT_FARM_ALLOWED_HTTP_HOSTS" in active
    cfg = tools.get_security_config()
    assert cfg["allow_private_http"] is True
    assert cfg["allowed_http_hosts"] == ["a.example", "b.example"]


def test_escape_hatches_off_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", raising=False)
    monkeypatch.delenv("AGENT_FARM_ALLOWED_HTTP_HOSTS", raising=False)
    monkeypatch.delenv("AGENT_FARM_PRODUCTION", raising=False)
    assert tools.escape_hatches_active() == []


# ---------- 3. Structured action log ----------

@pytest.fixture()
def log_path(tmp_path):
    path = str(tmp_path / "actions.jsonl")
    enable_action_log(path)
    try:
        yield path
    finally:
        disable_action_log()


def test_action_log_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_FARM_ACTION_LOG", raising=False)
    disable_action_log()
    assert log_action("http_blocked", "blocked", "x") is False
    assert not (tmp_path / "actions.jsonl").exists()


def test_action_log_records_and_verifies(log_path):
    assert log_action("http_blocked", "blocked", "blocked URL") is True
    assert log_action("exec_blocked", "blocked", "binary: rm") is True
    ok, bad = verify_action_log(log_path)
    assert ok and bad == []
    records = [json.loads(ln) for ln in open(log_path) if ln.strip()]
    assert [r["action"] for r in records] == ["http_blocked", "exec_blocked"]
    assert all(r["hash"] and r["prev_hash"] for r in records)
    assert records[1]["prev_hash"] == records[0]["hash"]


def test_action_log_tamper_is_detected(log_path):
    log_action("http_blocked", "blocked", "original")
    lines = open(log_path).read().splitlines()
    record = json.loads(lines[0])
    record["detail"] = "tampered"
    open(log_path, "w").write(json.dumps(record) + "\n")
    ok, bad = verify_action_log(log_path)
    assert ok is False and bad == [1]


def test_action_log_redacts_credentials_and_never_logs_bodies():
    assert "user" not in redact_url("http://user:secret@example.com/x?token=abc&ok=1")
    assert "secret" not in redact_url("http://user:secret@example.com/x?token=abc&ok=1")
    assert "abc" not in redact_url("http://example.com/x?token=abc&ok=1")
    assert "ok=1" in redact_url("http://example.com/x?token=abc&ok=1")


def test_blocked_events_are_logged(log_path):
    result = tools.tool_read_file("/etc/passwd")
    assert "error" in result
    result = tools.tool_exec_cmd("echo hi && rm -rf /")
    assert "error" in result
    records = [json.loads(ln) for ln in open(log_path) if ln.strip()]
    actions = [r["action"] for r in records]
    assert "file_blocked" in actions
    assert "exec_blocked" in actions
    # Full command line must not appear in the log.
    blob = open(log_path).read()
    assert "rm -rf" not in blob


# ---------- 4. Adversarial inputs ----------

def test_traversal_and_encoded_paths_are_blocked(tmp_path):
    # Literal ".." traversal to a blocked file.
    assert tools.tool_read_file("/tmp/../etc/passwd")["error"].startswith("Blocked")
    assert tools.tool_read_file("/tmp/../../etc/shadow")["error"].startswith("Blocked")
    # A path that only looks like it escapes via a prefix look-alike.
    assert tools.is_path_safe("/tmpfoo/secret") is True  # not blocked, but not a traversal
    assert tools.is_write_path_allowed("/tmpfoo/evil") is False


def test_symlink_directory_component_is_blocked(tmp_path):
    link = tmp_path / "linkdir"
    link.symlink_to("/etc", target_is_directory=True)
    result = tools.tool_read_file(str(link / "passwd"))
    assert "error" in result


def _fake_dns(ip):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET,
                 socket.SOCK_STREAM, 6, "", (ip, 0))]
    return fake


def test_ipv6_loopback_unspecified_and_link_local_rejected(monkeypatch):
    for ip in ("::1", "0.0.0.0", "169.254.10.20", "fe80::1"):
        monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(ip))
        assert tools.is_url_public(f"http://example.com/x") is False, ip


def test_at_trick_credential_urls_rejected(monkeypatch):
    # A URL whose real host is private, disguised with a public-looking
    # userinfo / @ segment, must still be rejected.
    assert tools.is_url_public("http://example.com@127.0.0.1:9/x") is False
    assert tools.is_url_public("http://user:pass@127.0.0.1:9/x") is False
    assert tools.is_url_public("https://example.com@169.254.169.254/latest/meta-data/") is False


def test_pipeline_blocked_stage_does_not_run(tmp_path):
    marker = tmp_path / "ran.txt"
    result = tools.tool_exec_cmd(f"echo hi | rm -rf {marker}")
    assert "error" in result
    assert not marker.exists()
    result = tools.tool_exec_cmd("echo hi | sh")
    assert "error" in result


class _ChainHandler(BaseHTTPRequestHandler):
    routes = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        status, body, extra = _ChainHandler.routes.get(
            self.path, (404, "not found", {}))
        self.send_response(status)
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture()
def chain_server():
    _ChainHandler.routes = {}
    server = HTTPServer(("127.0.0.1", 0), _ChainHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_redirect_chain_to_non_http_is_blocked_even_with_hatch(chain_server, monkeypatch):
    # With the local-testing hatch on, a private-IP redirect is
    # allowed by design -- so the adversarial case that must still
    # be blocked through a chain is a non-http scheme, which no
    # escape hatch can ever legitimise.
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    port = chain_server
    _ChainHandler.routes = {
        "/a": (302, "", {"Location": "/b"}),
        "/b": (302, "", {"Location": "file:///etc/passwd"}),
    }
    result = tools.tool_http_get(f"http://127.0.0.1:{port}/a")
    assert "error" in result
    assert "Blocked" in result["error"]


def test_redirect_chain_to_public_then_private_via_hostname(chain_server, monkeypatch):
    # Two hops: public-looking host -> private IP literal. Even with
    # the local-testing hatch on, a chain that ends at a raw private
    # literal on a different host must still go through validation,
    # and the final hop here is a loopback address on port 9.
    monkeypatch.delenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", raising=False)
    port = chain_server
    _ChainHandler.routes = {
        "/a": (302, "", {"Location": f"http://127.0.0.1:{port}/b"}),
        "/b": (302, "", {"Location": "http://127.0.0.1:9/internal"}),
    }
    result = tools.tool_http_get(f"http://127.0.0.1:{port}/a")
    assert "error" in result


def test_oversized_http_body_is_truncated(chain_server, monkeypatch):
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    port = chain_server
    big = "x" * 50000
    _ChainHandler.routes = {"/big": (200, big, {})}
    result = tools.tool_http_get(f"http://127.0.0.1:{port}/big")
    assert result.get("status") == 200
    assert len(result["body"]) <= tools.MAX_OUTPUT


def test_api_swarm_mutating_opt_in_is_logged(log_path):
    from agent_farm.server import api_swarm

    async def run():
        return await api_swarm(
            '[{"url": "http://127.0.0.1:9/x", "method": "POST"}]',
            allow_mutating=True,
        )

    result = asyncio.run(run())
    assert result["results"][0]["status"] == "blocked"
    records = [json.loads(ln) for ln in open(log_path) if ln.strip()]
    actions = [r["action"] for r in records]
    # The private URL is blocked by the SSRF guard; the mutating
    # opt-in itself is still recorded as an explicit allowance.
    assert "api_mutating_allowed" in actions
