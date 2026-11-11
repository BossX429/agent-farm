"""Tests that exec_swarm and api_swarm use the same guards as tools.py.

Previously both swarm tools called subprocess/httpx directly with their
own (older or missing) checks, bypassing the hardening in tools.py.
These tests fail if that bypass comes back.
"""

import asyncio
import json
import sys

from agent_farm.server import exec_swarm, api_swarm

# The running Python, with forward slashes: tool_exec_cmd parses
# commands with shlex (POSIX mode), which would eat the backslashes
# in a Windows path. Forward slashes work for CreateProcess too.
PY = sys.executable.replace("\\", "/")


def run(coro):
    return asyncio.run(coro)


def test_exec_swarm_simple_command():
    result = run(exec_swarm(json.dumps([f"{PY} -c \"print('hello')\""])))
    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["stdout"].strip() == "hello"


def test_exec_swarm_blocks_dangerous_and_substitution(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    result = run(exec_swarm(json.dumps([
        f"echo hi; rm -rf {marker}",
        "echo $(whoami)",
        "echo `whoami`",
    ])))
    for r in result["results"]:
        assert r["status"] == "blocked"
    assert marker.exists()


def test_exec_swarm_blocks_multiline():
    result = run(exec_swarm(json.dumps(["echo hi\necho bye"])))
    assert result["results"][0]["status"] == "blocked"


def test_api_swarm_blocks_private_and_loopback():
    result = run(api_swarm(json.dumps([
        {"url": "http://127.0.0.1:8765/ask"},
        {"url": "http://169.254.169.254/latest/meta-data/"},
        {"url": "file:///etc/passwd"},
    ])))
    for r in result["results"]:
        assert r["status"] == "blocked"
        assert "Blocked" in r["error"]


def test_api_swarm_rejects_unsupported_method():
    result = run(api_swarm(json.dumps([
        {"url": "https://example.com/data", "method": "PATCH"},
    ])))
    assert result["results"][0]["status"] in ("blocked", "failed")
    assert "Unsupported method" in result["results"][0]["error"]
