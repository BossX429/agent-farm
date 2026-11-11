"""Tests for the ChatGPT-review top security fixes (PR: fix/chatgpt-top-security).

1. Private-HTTP override bypass: AGENT_FARM_ALLOW_PRIVATE_HTTP and
   AGENT_FARM_ALLOWED_HTTP_HOSTS relax ONLY the private-IP rule.
   Scheme validation and credential rejection stay mandatory.
2. DNS rebinding: URLs are validated once and then connected to the
   exact validated IP, with the original hostname kept for Host/TLS.
   Redirects are re-validated and re-pinned on every hop.
3. Shell execution: no shell is used at all. Only simple commands and
   safe pipelines (argument-array stages joined by a single '|') run;
   chaining, redirection, substitution, and backgrounding are rejected.

Local HTTP servers are used for the network tests, so these run
offline and never touch a real host.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_farm import tools

# The running Python, quoted and with forward slashes: tool_exec_cmd
# parses commands with shlex (POSIX mode), which would eat the
# backslashes in a Windows path, and an unquoted path with spaces
# would split into fragments. Forward slashes work for CreateProcess.
PY = '"' + sys.executable.replace("\\", "/") + '"'


# ---------- helpers ----------

def _fake_getaddrinfo_for(ip):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
    return fake


def _fake_getaddrinfo_mixed(ips):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]
    return fake


class _Handler(BaseHTTPRequestHandler):
    # Captured by the test server, per request path.
    routes = {}
    seen_hosts = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):
        _Handler.seen_hosts.append(self.headers.get("Host"))
        status, body, extra = _Handler.routes.get(
            self.path, (404, "not found", {}))
        self.send_response(status)
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture()
def local_server():
    _Handler.routes = {
        "/hello": (200, "hello-pinned", {}),
        "/redirect-to-file": (302, "", {"Location": "file:///etc/passwd"}),
        "/redirect-to-private": (302, "", {"Location": "http://127.0.0.1:9/nope"}),
    }
    _Handler.seen_hosts = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


# ---------- 1. Private-HTTP override bypass ----------

def test_private_override_still_blocks_scheme_and_credentials(monkeypatch):
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("93.184.216.34"))
    # Private addresses ARE allowed under the flag -- that is its job.
    assert tools.is_url_public("http://127.0.0.1/admin") is True
    assert tools.is_url_public("http://10.0.0.5/admin") is True
    # But scheme and credential checks are NOT relaxed by the flag.
    assert tools.is_url_public("file:///etc/passwd") is False
    assert tools.is_url_public("ftp://example.com/file") is False
    assert tools.is_url_public("https://user:pass@example.com/") is False
    assert tools.is_url_public("http://user:pass@127.0.0.1/admin") is False


def test_allowed_hosts_still_block_scheme_and_credentials(monkeypatch):
    monkeypatch.setenv("AGENT_FARM_ALLOWED_HTTP_HOSTS", "example.com")
    # Even though example.com is allowed, and even though it resolves
    # privately here, scheme/credential checks still apply.
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("10.0.0.5"))
    assert tools.is_url_public("https://example.com/data") is True
    assert tools.is_url_public("file:///etc/passwd") is False
    assert tools.is_url_public("https://user:pass@example.com/") is False


def test_allowed_hosts_must_resolve(monkeypatch):
    monkeypatch.setenv("AGENT_FARM_ALLOWED_HTTP_HOSTS", "example.com")

    def failing(host, port, *args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", failing)
    # Previously an allowed host was accepted without any DNS check;
    # now a host that cannot resolve fails closed (pinning needs an IP).
    assert tools.is_url_public("https://example.com/data") is False


def test_redirect_to_non_http_is_blocked(local_server, monkeypatch):
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("127.0.0.1"))
    result = tools.tool_http_get(
        f"http://mytest.local:{local_server}/redirect-to-file")
    assert "error" in result
    assert "Blocked" in result["error"]


# ---------- 2. DNS rebinding / IP pinning ----------

def test_validation_rejects_mixed_public_and_private_answers(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        _fake_getaddrinfo_mixed(["93.184.216.34", "10.0.0.5"]))
    # A rotating answer that includes a private address must not pass,
    # even though the first address is public.
    assert tools.is_url_public("https://example.com/data") is False
    validated = tools._validate_url("https://example.com/data")
    assert validated is None


def test_validation_pins_first_resolved_ip(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        _fake_getaddrinfo_mixed(["93.184.216.34", "93.184.216.35"]))
    validated = tools._validate_url("https://example.com/data")
    assert validated is not None
    assert validated["pinned_ip"] == "93.184.216.34"
    assert validated["host"] == "example.com"


def test_pinned_fetch_uses_validated_ip_not_dns_name(local_server, monkeypatch):
    # "mytest.local" does not exist in real DNS. It only works because
    # validation resolves it (mocked) to the local server, and the
    # connection then goes to that IP while keeping the hostname for
    # the Host header.
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("127.0.0.1"))
    result = tools.tool_http_get(f"http://mytest.local:{local_server}/hello")
    assert result.get("status") == 200
    assert result.get("body") == "hello-pinned"
    assert _Handler.seen_hosts[-1].startswith("mytest.local:")


def test_pinned_fetch_survives_dns_flipping_after_validation(local_server, monkeypatch):
    # Simulate rebinding: the first lookup (validation) returns the
    # local server, every later lookup returns a different private IP.
    # A client that re-resolved would go to the wrong address and fail;
    # the pinned connection must still reach the local server.
    monkeypatch.setenv("AGENT_FARM_ALLOW_PRIVATE_HTTP", "1")
    calls = {"n": 0}

    def flipping(host, port, *args, **kwargs):
        calls["n"] += 1
        ip = "127.0.0.1" if calls["n"] == 1 else "10.0.0.5"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", flipping)
    result = tools.tool_http_get(f"http://mytest.local:{local_server}/hello")
    assert result.get("status") == 200
    assert result.get("body") == "hello-pinned"


def test_https_pinning_keeps_hostname_for_tls(monkeypatch):
    # The HTTPS pinned connection must connect to the IP but verify
    # the certificate against the hostname, not the IP.
    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self.connected_to = None
            self.timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, address):
            self.connected_to = address

    class FakeContext:
        def __init__(self):
            self.wrapped = None
            self.post_handshake_auth = None

        def set_alpn_protocols(self, protocols):
            pass

        def wrap_socket(self, sock, server_hostname=None):
            self.wrapped = (sock, server_hostname)
            return sock

    fake_sock = FakeSocket()
    fake_ctx = FakeContext()
    monkeypatch.setattr(socket, "socket", lambda *a, **k: fake_sock)
    monkeypatch.setattr("ssl._create_default_https_context", lambda: fake_ctx)

    conn = tools._PinnedHTTPSConnection("example.com", 443, timeout=30.0)
    conn.pinned_ip = "93.184.216.34"
    conn.connect()
    assert fake_sock.connected_to == ("93.184.216.34", 443)
    assert fake_ctx.wrapped[1] == "example.com"


# ---------- 3. Shell execution: no shell, safe pipelines only ----------

def test_safe_pipeline_runs_without_shell():
    # Pipelines run through the running Python on every platform:
    # the Unix tools in the old version (printf, sort, head, tr)
    # do not exist as Windows executables, so a shell-free run
    # could never find them. The properties under test are the
    # same: stages run with no shell, and output pipes between them.
    result = tools.tool_exec_cmd(
        f"{PY} -c \"print('c\\nb\\na')\" | {PY} -c "
        "\"print(*sorted(__import__('sys').stdin), sep='', end='')\""
    )
    assert result.get("returncode") == 0
    assert result["stdout"] == "a\nb\nc\n"
    result = tools.tool_exec_cmd(
        f"{PY} -c \"print('hello')\" | {PY} -c "
        "\"print(__import__('sys').stdin.read().upper().strip())\""
    )
    assert result.get("returncode") == 0
    assert result["stdout"].strip() == "HELLO"


def test_exec_cmd_never_uses_a_shell(monkeypatch):
    # The pipeline test above would still pass if tool_exec_cmd
    # regressed to shell=True (a shell runs these Python pipelines
    # fine), so this test watches the launcher itself: every process
    # the exec path spawns must be launched with shell=False.
    # This test must FAIL if the code is changed to shell=True --
    # that is its whole job, so it is checked that way, not just by
    # watching it pass.
    calls = []
    real_popen = subprocess.Popen
    real_run = subprocess.run

    def recording_popen(*args, **kwargs):
        calls.append(("Popen", kwargs))
        return real_popen(*args, **kwargs)

    def recording_run(*args, **kwargs):
        calls.append(("run", kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(subprocess, "run", recording_run)
    assert tools.tool_exec_cmd(f"{PY} -c \"print('hello')\"")["returncode"] == 0
    assert tools.tool_exec_cmd(
        f"{PY} -c \"print('hello')\" | {PY} -c "
        "\"print(__import__('sys').stdin.read().upper().strip())\""
    )["returncode"] == 0
    assert calls, "expected the exec path to launch processes"
    assert all(kwargs.get("shell") is False for _, kwargs in calls)


def test_pipeline_blocked_stage_does_not_run(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    result = tools.tool_exec_cmd(f"echo hi | rm -rf {marker}")
    assert "error" in result
    assert marker.exists()


def test_shell_operators_are_rejected(tmp_path):
    marker = tmp_path / "should_not_exist.txt"
    cases = [
        "echo hi && echo bye",
        "echo hi || echo bye",
        f"echo hi > {marker}",
        "echo hi; echo bye",
        "echo $(whoami)",
        "echo `whoami`",
        "echo $HOME",
        "echo hi & echo bye",
    ]
    for cmd in cases:
        result = tools.tool_exec_cmd(cmd)
        assert "error" in result, cmd
    assert not marker.exists()


def test_redirection_and_chaining_do_not_create_files(tmp_path):
    marker = tmp_path / "nope.txt"
    result = tools.tool_exec_cmd(f"echo hello > {marker}")
    assert "error" in result
    assert not marker.exists()
    result = tools.tool_exec_cmd(f"echo hello && echo bye > {marker}")
    assert "error" in result
    assert not marker.exists()


def test_pipeline_stage_limit():
    too_many = " | ".join(["echo x"] * (tools.MAX_PIPELINE_STAGES + 1))
    result = tools.tool_exec_cmd(too_many)
    assert "error" in result


def test_exec_swarm_pipeline_and_rejection():
    from agent_farm.server import exec_swarm

    result = asyncio.run(exec_swarm(json.dumps([
        f"{PY} -c \"print('hello')\" | {PY} -c "
        "\"print(__import__('sys').stdin.read().upper().strip())\""
    ])))
    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["stdout"].strip() == "HELLO"
    result = asyncio.run(exec_swarm(json.dumps(["echo hi && echo bye"])))
    assert result["results"][0]["status"] == "blocked"
