"""Tests for the PR reviewer (agent_farm/pr_review.py).

The model is always faked here: the tests prove the diff parsing,
the finding parsing, and the wiring, not the model's judgment.
The live proof on the PC is where the real model is exercised.
"""

from agent_farm.pr_review import (
    parse_findings,
    parse_unified_diff,
    review_branch,
    review_diff,
    review_file_diff,
)

_DIFF = """diff --git a/agent_farm/profiles.py b/agent_farm/profiles.py
index 1111111..2222222 100644
--- a/agent_farm/profiles.py
+++ b/agent_farm/profiles.py
@@ -10,6 +10,8 @@
 import time
 
+def new_helper(x):
+    return x / 0
+
 def old_function():
     return 1
diff --git a/README.md b/README.md
index 3333333..4444444 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
 # Title
+New line here.
 Old line.
"""


def test_parse_unified_diff_splits_files():
    files = parse_unified_diff(_DIFF)
    assert [f["path"] for f in files] == [
        "agent_farm/profiles.py",
        "README.md",
    ]
    # The added line "def new_helper(x):" lands on new-file line 12
    # (hunk starts at 10, two context lines, then the addition).
    assert files[0]["added_lines"][12] == "def new_helper(x):"
    assert files[0]["added_lines"][13] == "    return x / 0"


def test_parse_unified_diff_empty_returns_empty():
    assert parse_unified_diff("") == []
    assert parse_unified_diff("not a diff") == []


def test_parse_findings_structured():
    text = """[HIGH] Division by zero
File: agent_farm/profiles.py
Line: 14
Body: new_helper divides by zero with no guard. Any call raises ZeroDivisionError.

[LOW] Missing docstring
File: agent_farm/profiles.py
Line: 13
Body: new_helper has no docstring.
"""
    findings = parse_findings(text)
    assert len(findings) == 2
    assert findings[0]["severity"] == "HIGH"
    assert findings[0]["title"] == "Division by zero"
    assert findings[0]["file"] == "agent_farm/profiles.py"
    assert findings[0]["line"] == 14
    assert "ZeroDivisionError" in findings[0]["body"]


def test_parse_findings_no_issues_returns_empty():
    assert parse_findings("No issues found") == []


def test_parse_findings_unstructured_output_is_kept():
    # The model did not follow the format. Its words must not be
    # silently dropped: they come back as one LOW finding.
    findings = parse_findings("This looks wrong because the divisor is zero.")
    assert len(findings) == 1
    assert findings[0]["severity"] == "LOW"
    assert findings[0]["title"] == "Unstructured review output"
    assert "divisor is zero" in findings[0]["body"]


def _fake_finding(model, prompt):
    return """[MEDIUM] Division by zero
File: agent_farm/profiles.py
Line: 14
Body: new_helper divides by zero with no guard.
"""


def test_review_file_diff_returns_structured_findings():
    findings = review_file_diff(
        "agent_farm/profiles.py", _DIFF, query_fn=_fake_finding
    )
    assert len(findings) == 1
    assert findings[0]["severity"] == "MEDIUM"
    assert findings[0]["line"] == 14


def test_review_diff_sorts_by_severity():
    def _two_findings(model, prompt):
        if "profiles.py" in prompt:
            return """[LOW] Minor nit
File: agent_farm/profiles.py
Line: 13
Body: Nit.
"""
        return """[HIGH] Bad change
File: README.md
Line: 2
Body: Bad.
"""

    result = review_diff(_DIFF, query_fn=_two_findings)
    assert result["files"] == ["agent_farm/profiles.py", "README.md"]
    assert result["findings"][0]["severity"] == "HIGH"
    assert result["findings"][1]["severity"] == "LOW"


def test_review_diff_empty_returns_error():
    result = review_diff("", query_fn=_fake_finding)
    assert result["findings"] == []
    assert "error" in result


def test_review_branch_bad_repo_returns_error():
    result = review_branch("/no/such/repo", "main", query_fn=_fake_finding)
    assert result["findings"] == []
    assert "error" in result


def test_review_branch_uses_git_diff(tmp_path):
    # A real local repo: one commit, then an uncommitted change.
    # review_branch must diff base...head and review the change.
    # Skipped where git cannot run from a subprocess (e.g. a
    # Windows install where git is a package-manager shim that
    # fails outside an interactive shell).
    import shutil
    import subprocess

    import pytest

    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    probe = subprocess.run(
        ["git", "--version"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("git does not run from a subprocess here")

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@t.t"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "t"],
        check=True,
        capture_output=True,
    )
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "second"],
        check=True,
        capture_output=True,
    )
    result = review_branch(str(tmp_path), "HEAD~1", "HEAD", query_fn=_fake_finding)
    assert result["files"] == ["a.py"]
    assert len(result["findings"]) == 1
