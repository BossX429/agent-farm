"""Tests for the PR reviewer (agent_farm/pr_review.py).

The model is always faked here: the tests prove the diff parsing,
the finding parsing, and the wiring, not the model's judgment.
The live proof on the PC is where the real model is exercised.
"""

from agent_farm.pr_review import (
    cross_check_findings,
    parse_findings,
    parse_unified_diff,
    review_branch,
    review_diff,
    review_file_diff,
)

_FINDING = {
    "severity": "HIGH",
    "title": "Bad change",
    "file": "agent_farm/profiles.py",
    "line": 13,
    "body": "Bad.",
}


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
        # Route on the diff header, not on a bare filename: the
        # review examples mention colony_profiles.py, which
        # contains "profiles.py" as a substring.
        if "diff --git a/agent_farm/profiles.py" in prompt:
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


def test_review_prompt_includes_examples_and_file():
    # The reviewer must see the full file and the Sourcery
    # examples: without the file it invents findings about things
    # the rest of the file already handles, and without the
    # examples it has no standard for what a good finding is.
    seen = {}

    def _capture(model, prompt):
        seen["prompt"] = prompt
        return "No issues found"

    review_file_diff(
        "agent_farm/profiles.py",
        "diff --git a/agent_farm/profiles.py b/agent_farm/profiles.py\n+new line",
        query_fn=_capture,
        file_content="def normalize():\n    return 0\n",
    )
    prompt = seen["prompt"]
    assert "EXAMPLE 1" in prompt
    assert "EXAMPLE 2" in prompt
    assert "Failed answers inflate correctness" in prompt
    assert "def normalize():" in prompt


def test_review_diff_passes_file_contents():
    # file_contents reach the per-file review: the fake model
    # records the prompt for each file, and the file content must
    # be in it.
    prompts = {}

    def _capture(model, prompt):
        prompts[model] = prompt
        return "No issues found"

    seen_prompts = []

    def _per_file(model, prompt):
        seen_prompts.append(prompt)
        return "No issues found"

    result = review_diff(
        _DIFF,
        query_fn=_per_file,
        file_contents={"agent_farm/profiles.py": "FULL PROFILES CONTENT"},
    )
    assert result["files"] == ["agent_farm/profiles.py", "README.md"]
    assert any("FULL PROFILES CONTENT" in p for p in seen_prompts)
    # The README prompt must NOT carry the profiles content.
    readme_prompts = [
        p for p in seen_prompts if "diff --git a/README.md" in p
    ]
    assert readme_prompts
    assert all("FULL PROFILES CONTENT" not in p for p in readme_prompts)


def test_review_branch_uses_git_exe_env(monkeypatch, tmp_path):
    # AGENT_FARM_GIT must be honored: on the PC bare "git" is a
    # scoop shim that cannot spawn from a Python subprocess, so
    # the caller points at the direct git.exe. A fake git records
    # its arguments; if the code ignored GIT_EXE and ran bare
    # git, the fake would never be called and no record appears.
    import os
    import stat

    import agent_farm.pr_review as pr

    record = tmp_path / "calls.txt"
    if os.name == "nt":
        # Windows has no sh: the fake git is a batch file.
        fake_git = tmp_path / "fake-git.bat"
        fake_git.write_text(f"@echo off\r\necho %* >> {record}\r\nexit /b 0\r\n")
    else:
        fake_git = tmp_path / "fake-git"
        fake_git.write_text(
            "#!/bin/sh\n"
            f"echo \"$@\" >> {record}\n"
            "exit 0\n"
        )
        fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(pr, "GIT_EXE", str(fake_git))
    result = pr.review_branch(str(tmp_path), "main", "HEAD")
    assert result["files"] == []
    assert result["findings"] == []
    calls = record.read_text()
    assert "diff main...HEAD" in calls


def test_cross_check_keeps_judge_confirmed_finding():
    # The judge says YES: the finding is kept.
    def _yes(model, prompt):
        assert model == "gemma4:26b-a4b-qat"
        return "VERDICT: YES\nReason: The code really does that."

    kept, rejected = cross_check_findings(
        [_FINDING], {"agent_farm/profiles.py": "x = 1\n"}, query_fn=_yes
    )
    assert kept == [_FINDING]
    assert rejected == []


def test_cross_check_drops_judge_rejected_finding():
    # The judge says NO: the finding is dropped with the reason.
    def _no(model, prompt):
        return "VERDICT: NO\nReason: The code does not do that."

    kept, rejected = cross_check_findings(
        [_FINDING], {"agent_farm/profiles.py": "x = 1\n"}, query_fn=_no
    )
    assert kept == []
    assert len(rejected) == 1
    assert "cross-check judge said NO" in rejected[0][1]
    assert "The code does not do that" in rejected[0][1]


def test_cross_check_keeps_when_judge_does_not_answer():
    # A judge that does not answer in the required format does
    # not reject: the finding is kept.
    def _unclear(model, prompt):
        return "This looks bad, maybe."

    kept, rejected = cross_check_findings(
        [_FINDING], {"agent_farm/profiles.py": "x = 1\n"}, query_fn=_unclear
    )
    assert kept == [_FINDING]
    assert rejected == []


def test_cross_check_keeps_when_judge_fails():
    # A judge that raises (model not pulled, Ollama down) does
    # not silence the reviewer: the finding is kept.
    def _boom(model, prompt):
        raise RuntimeError("Ollama is down")

    kept, rejected = cross_check_findings(
        [_FINDING], {"agent_farm/profiles.py": "x = 1\n"}, query_fn=_boom
    )
    assert kept == [_FINDING]
    assert rejected == []


def test_review_diff_skips_cross_check_when_asked():
    # cross_check=False means the judge is never called, even
    # though the fake would answer. The finding is reported as
    # the reviewer wrote it.
    calls = []

    def _fake(model, prompt):
        calls.append(model)
        return """[HIGH] Bad change
File: agent_farm/profiles.py
Line: 13
Body: Bad.
"""

    result = review_diff(_DIFF, query_fn=_fake, cross_check=False)
    # The fake returns one finding per file, and _DIFF has two
    # files. Only the review calls happened (one per file); no
    # judge call.
    assert len(result["findings"]) == 2
    assert calls == ["qwen3:30b-a3b-q3k", "qwen3:30b-a3b-q3k"]


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
