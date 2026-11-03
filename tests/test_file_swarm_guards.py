"""Tests for the Manus-review fixes to file_swarm and code_review_swarm.

Previously file_swarm allowed write/append/delete anywhere outside a
system-directory blocklist (including the whole home directory), while
write_file only allowed /tmp, /home/kyle/repos, and /home/kyle/Desktop.
And code_review_swarm opened a filepath directly, with no blocked-path
check and no byte limit.

These tests fail if those bypasses come back. They do not call Ollama:
code_review tests only check the filepath guard, which runs before
any model call.
"""

import asyncio
import json
import os

from agent_farm.swarm_tools import file_swarm, code_review_swarm


def run(coro):
    return asyncio.run(coro)


def test_file_swarm_write_outside_allowlist_blocked(tmp_path):
    # tmp_path is under /tmp, so use a path that is definitely outside
    # the write allowlist: the user's home is not in it on this VM
    # unless it happens to be /home/kyle, and /tmp is allowed, so use
    # a sibling of tmp that is not allowed.
    outside = "/var/tmp/agent_farm_test_outside.txt"
    result = run(file_swarm(json.dumps([
        {"action": "write", "path": outside, "content": "hello"},
        {"action": "append", "path": outside, "content": "hello"},
        {"action": "delete", "path": outside},
    ])))
    for r in result["results"]:
        assert "error" in r
        assert "Write only allowed" in r["error"]
    assert not os.path.exists(outside)


def test_file_swarm_write_inside_tmp_works(tmp_path):
    # file_swarm runs operations in parallel, so dependent steps must
    # be in separate calls -- otherwise a read can run before an append.
    path = str(tmp_path / "hello.txt")
    result = run(file_swarm(json.dumps([
        {"action": "write", "path": path, "content": "hello"},
    ])))
    assert result["results"][0]["status"] == "success"
    result = run(file_swarm(json.dumps([
        {"action": "append", "path": path, "content": " world"},
    ])))
    assert result["results"][0]["status"] == "success"
    result = run(file_swarm(json.dumps([
        {"action": "read", "path": path},
        {"action": "exists", "path": path},
    ])))
    assert result["results"][0]["content"] == "hello world"
    assert result["results"][1]["exists"] is True
    result = run(file_swarm(json.dumps([
        {"action": "delete", "path": path},
    ])))
    assert result["results"][0]["status"] == "success"
    result = run(file_swarm(json.dumps([
        {"action": "exists", "path": path},
    ])))
    assert result["results"][0]["exists"] is False


def test_file_swarm_read_blocked_dirs(tmp_path):
    result = run(file_swarm(json.dumps([
        {"action": "read", "path": "/etc/passwd"},
        {"action": "read", "path": "/etc/hostname"},
        {"action": "exists", "path": "/etc/passwd"},
    ])))
    for r in result["results"]:
        assert "error" in r


def test_file_swarm_write_via_symlink_outside_blocked(tmp_path):
    # A symlink inside /tmp pointing outside the allowlist must not pass.
    target = tmp_path / "outside.txt"
    target.write_text("original")
    # Use a directory outside /tmp as the real target: /var/tmp is
    # outside the allowlist, but may not exist on all systems, so
    # instead link to /etc/passwd's directory via a file that is
    # blocked for writes anyway. The key check is the symlink case:
    # link in /tmp -> a file in the home directory (outside allowlist
    # unless home is /home/kyle).
    home_target = os.path.expanduser("~/agent_farm_symlink_test.txt")
    if os.path.realpath(home_target).startswith("/tmp"):
        return  # home is inside /tmp here; test would not be meaningful
    link = tmp_path / "link.txt"
    try:
        os.symlink(home_target, link)
    except OSError:
        return
    result = run(file_swarm(json.dumps([
        {"action": "write", "path": str(link), "content": "pwned"},
    ])))
    assert "error" in result["results"][0]
    assert not os.path.exists(home_target)


def test_code_review_filepath_blocked():
    # This must fail before any model call, so no Ollama is needed.
    result = run(code_review_swarm("", filepath="/etc/passwd"))
    assert "error" in result
    assert "Blocked" in result["error"]


def test_code_review_filepath_system_dir_blocked():
    result = run(code_review_swarm("", filepath="/etc/hostname"))
    assert "error" in result
    assert "Blocked" in result["error"]
