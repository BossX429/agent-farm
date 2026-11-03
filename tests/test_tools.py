"""Tests for agent_farm.tools safety guards and tool-call parsing.

These cover the fixes from PR #5 plus the SSRF / shell hardening in
this PR. They use only local files and mocked DNS, so they run offline
and do not touch the network, Ollama, or the real filesystem outside
tmp_path.
"""

import os
import socket

import pytest

from agent_farm import tools
from agent_farm.server import _unique_id


# ---------- path safety ----------

def test_blocked_paths():
    assert not tools.is_path_safe("/etc/passwd")
    assert not tools.is_path_safe("/etc/shadow")
    assert not tools.is_path_safe("/root/.bashrc")
    assert not tools.is_path_safe("/proc/kcore")


def test_write_allowlist_prefix_lookalike(tmp_path):
    # "/tmpfoo" must not pass as "/tmp", and a path outside the
    # allowlist must not be writable even if it shares a prefix.
    assert not tools.is_write_path_allowed("/tmpfoo/evil.txt")
    assert not tools.is_write_path_allowed("/home/kyle/reposs/evil.txt")


def test_write_allowlist_and_symlink(tmp_path):
    # A symlink inside /tmp pointing outside the allowlist must not pass.
    # The target must itself be outside /tmp and outside the other
    # allowed dirs, otherwise the canonical path is legitimately allowed.
    outside = os.path.expanduser("~/af_outside_test_xyz.txt")
    with open(outside, "w") as f:
        f.write("secret")
    link = os.path.join("/tmp", "af_test_link_xyz")
    try:
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(outside, link)
        assert not tools.is_write_path_allowed(link)
    finally:
        if os.path.lexists(link):
            os.unlink(link)
        if os.path.exists(outside):
            os.unlink(outside)


def test_read_file_capped_and_blocked(tmp_path):
    result = tools.tool_read_file("/etc/passwd")
    assert "error" in result
    f = tmp_path / "big.txt"
    f.write_text("x" * 200000)
    result = tools.tool_read_file(str(f))
    assert result["truncated"] is True
    assert len(result["content"]) <= tools.MAX_FILE_SIZE


# ---------- exec safety ----------

def test_command_blocklist_variants():
    assert not tools.is_command_safe("rm  -rf /tmp/x")  # whitespace collapse
    assert not tools.is_command_safe("rm\t-r /tmp/x")
    assert not tools.is_command_safe("echo hi; rm -rf /")
    assert not tools.is_command_safe("echo $(rm -rf /)")
    assert not tools.is_command_safe("echo `rm -rf /`")
    assert not tools.is_command_safe("sudo ls")
    assert tools.is_command_safe("echo hello")


def test_exec_simple_runs_without_shell():
    result = tools.tool_exec_cmd("echo hello")
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


def test_exec_blocked_does_not_run(tmp_path):
    marker = tmp_path / "should_not_exist.txt"
    marker.write_text("keep me")
    result = tools.tool_exec_cmd(f"echo hi; rm -rf {marker}")
    # The blocked pattern must return an error and must not execute,
    # so the marker file must still exist.
    assert "error" in result
    assert marker.exists()
    result = tools.tool_exec_cmd("rm -rf /tmp/af_nope_xyz")
    assert "error" in result


def test_exec_multiline_rejected():
    result = tools.tool_exec_cmd("echo hi\necho bye")
    assert "error" in result


def test_check_service_rejects_injection():
    result = tools.tool_check_service("x; rm -rf /")
    assert "error" in result


# ---------- HTTP / SSRF ----------

def _fake_getaddrinfo(public=True):
    def fake(host, port, *args, **kwargs):
        ip = "93.184.216.34" if public else "10.0.0.5"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
    return fake


def test_url_public_and_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(public=True))
    assert tools.is_url_public("https://example.com/data") is True
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(public=False))
    assert tools.is_url_public("https://example.com/data") is False


def test_url_rejected_schemes_and_hosts(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(public=True))
    assert tools.is_url_public("file:///etc/passwd") is False
    assert tools.is_url_public("http://localhost:8765/search") is False
    assert tools.is_url_public("http://127.0.0.1/admin") is False
    assert tools.is_url_public("http://10.0.0.5/admin") is False
    assert tools.is_url_public("https://user:pass@example.com/") is False
    assert tools.is_url_public("https://169.254.169.254/latest/meta-data/") is False


def test_http_get_blocks_private(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(public=False))
    result = tools.tool_http_get("https://example.com/data")
    assert "error" in result


def test_http_get_blocks_loopback():
    result = tools.tool_http_get("http://127.0.0.1:8765/search")
    assert "error" in result


# ---------- tool-call parsing ----------

def test_parse_json_args():
    name, kwargs = tools.parse_tool_call(
        'TOOL: write_file({"path": "/tmp/x.txt", "content": "hi"})')
    assert name == "write_file"
    assert kwargs == {"path": "/tmp/x.txt", "content": "hi"}


def test_parse_quoted_commas_and_parens():
    name, kwargs = tools.parse_tool_call(
        'TOOL: write_file(path="/tmp/x.txt", content="Hello, World! (again)")')
    assert name == "write_file"
    assert kwargs["content"] == "Hello, World! (again)"


def test_parse_positional():
    name, kwargs = tools.parse_tool_call("TOOL: read_file(/tmp/x.txt)")
    assert name == "read_file"
    assert kwargs == {"path": "/tmp/x.txt"}


def test_parse_no_tool_call():
    assert tools.parse_tool_call("just some text") == (None, None)


# ---------- IDs ----------

def test_unique_ids_do_not_collide():
    ids = {_unique_id("colony") for _ in range(1000)}
    assert len(ids) == 1000
    assert all(i.startswith("colony-") for i in ids)
