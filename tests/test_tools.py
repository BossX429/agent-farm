"""Tests for agent_farm.tools safety guards and tool-call parsing.

These cover the fixes from PR #5 plus the SSRF / shell hardening in
this PR. They use only local files and mocked DNS, so they run offline
and do not touch the network, Ollama, or the real filesystem outside
tmp_path.
"""

import os
import socket
import sys
import tempfile

import pytest

from agent_farm import tools
from agent_farm.server import _unique_id

# The running Python, quoted and with forward slashes: tool_exec_cmd
# parses commands with shlex (POSIX mode), which would eat the
# backslashes in a Windows path, and an unquoted path with spaces
# would split into fragments. Forward slashes work for CreateProcess.
PY = '"' + sys.executable.replace("\\", "/") + '"'


# ---------- path safety ----------

def test_blocked_paths():
    assert not tools.is_path_safe("/etc/passwd")
    assert not tools.is_path_safe("/etc/shadow")
    assert not tools.is_path_safe("/root/.bashrc")
    assert not tools.is_path_safe("/proc/kcore")


# ---------- H1: unified read boundary (audit 2026-11-17) ----------
# The normal read tools used to check only BLOCKED_PATHS (8 Linux
# files), while the swarm tools also blocked whole system dirs.
# The boundary is now one deny list (READ_DENY_ROOTS) enforced by
# is_path_safe, so a normal tool can no longer read what a swarm
# tool refuses -- including home secrets and Windows paths.

def test_read_boundary_blocks_home_secrets():
    home = os.path.expanduser("~")
    for secret in (".ssh", ".ssh/id_ed25519", ".aws/credentials",
                   ".gnupg", ".config", ".mozilla", ".netrc",
                   ".docker", ".docker/config.json", ".kube", ".kube/config"):
        assert not tools.is_path_safe(os.path.join(home, secret)), secret
    # The tools refuse the path instead of probing it: file_exists
    # reports a block, not an existence answer.
    assert "error" in tools.tool_read_file(os.path.join(home, ".ssh/id_ed25519"))
    assert "error" in tools.tool_list_dir(os.path.join(home, ".ssh"))
    result = tools.tool_file_exists(os.path.join(home, ".aws/credentials"))
    assert "error" in result and "exists" not in result


def test_read_boundary_blocks_system_dirs():
    # The swarm guard already blocked these; the normal tools did
    # not (the H1 gap). /etc/hostname is the audit's example: a
    # benign file in a denied directory is still denied.
    assert not tools.is_path_safe("/etc/hostname")
    assert not tools.is_path_safe("/usr/bin/python3")
    assert not tools.is_path_safe("/proc/self/status")
    assert not tools.is_path_safe("/sys/kernel")
    assert "error" in tools.tool_read_file("/etc/hostname")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only paths")
def test_read_boundary_blocks_windows_roots():
    profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    assert not tools.is_path_safe(r"C:\Windows\System32\config\SAM")
    assert not tools.is_path_safe(
        os.path.join(profile, "AppData", "Local", "Google", "Chrome", "User Data")
    )
    assert not tools.is_path_safe(r"C:\ProgramData\some-app\config.json")
    # Case and separator mangling must not slip past a deny root
    # that does not exist on disk yet (realpath alone only folds
    # the case of components that exist).
    assert not tools.is_path_safe("c:/windows/system32/config/sam")
    assert not tools.is_path_safe(r"C:\WINDOWS\System32")


def test_read_boundary_allows_project_and_tmp(tmp_path):
    good = tmp_path / "notes.txt"
    good.write_text("hello")
    assert tools.is_path_safe(str(good))
    assert tools.tool_read_file(str(good))["content"] == "hello"
    assert tools.tool_list_dir(str(tmp_path))["count"] == 1
    assert tools.tool_file_exists(str(good))["exists"] is True
    # The swarm check is the same boundary now, not a looser one.
    assert tools.is_swarm_read_allowed(str(good))
    assert not tools.is_swarm_read_allowed("/etc/hostname")


def test_read_boundary_temp_dir_carve_out(monkeypatch, tmp_path):
    # The temp dir stays readable even when a deny root covers its
    # parent: on Windows the real temp dir sits under AppData, a
    # deny root, and without this carve-out every tmp_path test
    # fails there. The deny root added here is the temp dir's
    # parent -- an ancestor of Temp, like AppData is -- so the test
    # pins the carve-out on Linux too, where /tmp is not denied.
    # A path under that parent but outside Temp is still denied,
    # so the exception is narrow, not a blanket allow.
    temp_parent = os.path.dirname(os.path.realpath(tempfile.gettempdir()))
    monkeypatch.setattr(
        tools, "READ_DENY_ROOTS",
        list(tools.READ_DENY_ROOTS) + [temp_parent],
    )
    good = tmp_path / "scratch.txt"
    good.write_text("scratch")
    assert tools.is_path_safe(str(good))
    assert tools.tool_read_file(str(good))["content"] == "scratch"
    assert not tools.is_path_safe(os.path.join(temp_parent, "not-temp-marker"))


def test_read_boundary_temp_symlink_escape_blocked(monkeypatch, tmp_path):
    # A symlink inside the temp dir pointing at a denied location
    # does not inherit the carve-out: the canonical path is denied.
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "secret.txt").write_text("secret")
    link = tmp_path / "escape.txt"
    try:
        link.symlink_to(denied / "secret.txt")
    except OSError:
        pytest.skip("cannot create symlink here")
    monkeypatch.setattr(
        tools, "READ_DENY_ROOTS", list(tools.READ_DENY_ROOTS) + [str(denied)]
    )
    assert not tools.is_path_safe(str(link))
    assert "error" in tools.tool_read_file(str(link))


def test_search_code_skips_denied_subdir(monkeypatch, tmp_path):
    # A denied directory inside the search root must be pruned from
    # the walk, not searched. The deny list is monkeypatched so the
    # test does not depend on the real home or system dirs.
    root = tmp_path / "root"
    denied = root / "denied"
    denied.mkdir(parents=True)
    (root / "good.py").write_text("MARKER_ONE = 1\n")
    (denied / "secret.py").write_text("MARKER_ONE = 2\n")
    monkeypatch.setattr(
        tools, "READ_DENY_ROOTS", list(tools.READ_DENY_ROOTS) + [str(denied)]
    )
    result = tools.tool_search_code("MARKER_ONE", str(root))
    found = {m["file"] for m in result["matches"]}
    assert str(root / "good.py") in found
    assert str(denied / "secret.py") not in found


def test_search_code_skips_symlinked_secret_file(monkeypatch, tmp_path):
    # A symlinked file inside an allowed tree pointing at a denied
    # location must not be read: the canonical path is denied.
    root = tmp_path / "root"
    denied = tmp_path / "denied"
    denied.mkdir(parents=True)
    root.mkdir()
    (denied / "secret.py").write_text("MARKER_TWO = 1\n")
    (root / "good.py").write_text("MARKER_TWO = 2\n")
    link = root / "link.py"
    try:
        link.symlink_to(denied / "secret.py")
    except OSError:
        pytest.skip("cannot create symlink here")
    monkeypatch.setattr(
        tools, "READ_DENY_ROOTS", list(tools.READ_DENY_ROOTS) + [str(denied)]
    )
    result = tools.tool_search_code("MARKER_TWO", str(root))
    found = {m["file"] for m in result["matches"]}
    assert str(root / "good.py") in found
    assert str(link) not in found


def test_write_allowlist_prefix_lookalike(tmp_path):
    # "/tmpfoo" must not pass as "/tmp", and a path outside the
    # allowlist must not be writable even if it shares a prefix.
    assert not tools.is_write_path_allowed("/tmpfoo/evil.txt")
    assert not tools.is_write_path_allowed("/home/kyle/reposs/evil.txt")


def test_write_allowlist_and_symlink(tmp_path):
    # A symlink inside an allowed dir pointing outside it must not pass.
    # The link lives under tmp_path (an allowed dir), so the test checks
    # the symlink escape itself -- the old version put the link under
    # "/tmp", which on Windows is C:\tmp, not an allowed dir at all,
    # so it passed for the wrong reason. Skip when the platform does
    # not permit creating symlinks. The target must itself be outside
    # every allowed dir, otherwise the canonical path is legitimately
    # allowed.
    outside = os.path.expanduser("~/af_outside_test_xyz.txt")
    with open(outside, "w") as f:
        f.write("secret")
    link = tmp_path / "af_test_link_xyz"
    try:
        if os.path.lexists(link):
            os.unlink(link)
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this platform")
        assert not tools.is_write_path_allowed(str(link))
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


def test_command_blocklist_allows_ordinary_commands():
    # Audit M1: the old substring match blocked ordinary commands
    # because a blocked fragment sat inside another word or path.
    # "dd " is inside "add ", "rm -f /" is inside "rm -f /tmp/x",
    # and "mv /" is inside "mv /tmp/a /tmp/b". All four must pass.
    assert tools.is_command_safe("git add .")
    assert tools.is_command_safe('git commit -m "add feature"')
    assert tools.is_command_safe("git status")
    assert tools.is_command_safe("mv /tmp/a.txt /tmp/b.txt")
    assert tools.is_command_safe("rm -f /tmp/leftover.txt")
    assert tools.is_command_safe("echo add some text")


def test_command_blocklist_still_blocks_dangerous_commands():
    # The boundary fix must not let the dangerous forms through:
    # destructive removal, moving the root, dd, unlink, chmod 777,
    # and find -exec all stay blocked.
    assert not tools.is_command_safe("rm -rf /")
    assert not tools.is_command_safe("rm -f /")
    assert not tools.is_command_safe("mv / /tmp")
    assert not tools.is_command_safe("dd if=/dev/zero of=/dev/null")
    assert not tools.is_command_safe("unlink /tmp/x")
    assert not tools.is_command_safe("chmod 777 x")
    assert not tools.is_command_safe("find . -exec ls {} +")


def test_exec_simple_runs_without_shell():
    result = tools.tool_exec_cmd(f"{PY} -c \"print('hello')\"")
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


def test_write_path_allowed_in_system_temp_dir(tmp_path):
    # The system temp dir is in the write allowlist on every
    # platform. Only require tmp_path to be writable when it actually
    # lives inside an allowed root: with --basetemp set outside the
    # allowlist, the allowlist correctly rejects it and the test must
    # not fail for that.
    assert tools.is_write_path_allowed(
        os.path.join(tempfile.gettempdir(), "x.txt")
    )
    if tools.is_write_path_allowed(str(tmp_path)):
        assert tools.is_write_path_allowed(str(tmp_path / "x.txt"))


def test_dedicated_temp_dir_validation(tmp_path):
    # The temp-dir guard: a real dedicated temp dir passes, but a
    # home dir, a filesystem root, or a missing dir must never be
    # added to the write allowlist, or the whole home folder or
    # drive becomes writable. The passing case uses a directory
    # created under tmp_path (which itself lives under a known
    # temp location), not the configured temp dir: the configured
    # dir may legitimately point somewhere the guard must reject,
    # and the test must not fail for a TEMP setting the guard is
    # right to refuse.
    dedicated = tmp_path / "dedicated-temp"
    dedicated.mkdir()
    assert tools._is_dedicated_temp_dir(str(dedicated))
    assert not tools._is_dedicated_temp_dir(os.path.expanduser("~"))
    assert not tools._is_dedicated_temp_dir(os.path.abspath(os.sep))
    assert not tools._is_dedicated_temp_dir("/nonexistent-xyz-12345")

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


# ---------- search_code (native search, no grep) ----------

def test_search_code_finds_pattern(tmp_path):
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "b.py").write_text("def sub(a, b):\n    return a - b\n")
    result = tools.tool_search_code("def add", path=str(tmp_path))
    assert result["total_matches"] == 1
    assert result["matches"][0]["file"].endswith("a.py")
    assert result["matches"][0]["line"] == 1
    assert "def add" in result["matches"][0]["text"]
    assert result["files_searched"] == 2


def test_search_code_respects_file_pattern(tmp_path):
    (tmp_path / "a.py").write_text("needle here\n")
    (tmp_path / "a.txt").write_text("needle here\n")
    result = tools.tool_search_code("needle", path=str(tmp_path), file_pattern="*.py")
    assert result["total_matches"] == 1
    assert result["matches"][0]["file"].endswith("a.py")


def test_search_code_invalid_regex_falls_back_to_substring(tmp_path):
    (tmp_path / "a.py").write_text("price is [100] dollars\n")
    result = tools.tool_search_code("[100]", path=str(tmp_path))
    assert result["total_matches"] == 1


def test_search_code_missing_directory():
    result = tools.tool_search_code("x", path="/no/such/dir/af_xyz")
    assert "error" in result


def test_search_code_skips_pycache(tmp_path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "a.py").write_text("needle here\n")
    (tmp_path / "a.py").write_text("nothing\n")
    result = tools.tool_search_code("needle", path=str(tmp_path))
    assert result["total_matches"] == 0


# ---------- _clean_path (mangled model paths) ----------

def test_clean_path_strips_trailing_quote_and_separator():
    # The PR #30 live proof: the model wrote a Windows path ending in
    # backslash-quote, which JSON reads as a literal trailing quote and
    # Windows rejects with WinError 123.
    assert tools._clean_path('C:\\Users\\kyleh\\agent-farm\\"') == "C:\\Users\\kyleh\\agent-farm"
    assert tools._clean_path('"C:/Users/kyleh/agent-farm/"') == "C:/Users/kyleh/agent-farm"
    assert tools._clean_path("  C:/Users/kyleh/agent-farm/  ") == "C:/Users/kyleh/agent-farm"


def test_clean_path_keeps_drive_root():
    assert tools._clean_path("C:\\") == "C:\\"
    assert tools._clean_path("C:") == "C:\\"
    assert tools._clean_path("/") == "/"


def test_canonical_path_sanitizes_mangled_windows_path(tmp_path):
    # A mangled path must resolve to the real directory, not error.
    mangled = '"' + str(tmp_path) + '/"'
    assert tools._canonical_path(mangled) == str(tmp_path)
    result = tools.tool_list_dir(mangled)
    assert "error" not in result
    assert result["count"] == 0


# ---------- git resolution in exec_cmd ----------

def test_resolve_git_uses_env_path(monkeypatch):
    monkeypatch.setenv("AGENT_FARM_GIT", "/usr/bin/git")
    assert tools._resolve_git(["git", "branch", "-a"]) == ["/usr/bin/git", "branch", "-a"]


def test_resolve_git_leaves_git_alone_without_env(monkeypatch):
    monkeypatch.delenv("AGENT_FARM_GIT", raising=False)
    assert tools._resolve_git(["git", "branch", "-a"]) == ["git", "branch", "-a"]
    assert tools._resolve_git(["python", "-c", "x"]) == ["python", "-c", "x"]
