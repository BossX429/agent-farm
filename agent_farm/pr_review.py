"""Review a pull-request diff the way Sourcery does: line by line,
looking for what breaks, not for style.

The diff comes from local git (``git diff base...head``), so no
GitHub token is needed on the machine running the review. The
reviewer is the synthesizer model (the 30b reasoning tier), the
same model ``code_review_swarm`` uses. Each changed file's diff
is sent to the model with the pre-push checklist as the review
prompt: what inputs can this receive, would the test fail if the
code were wrong, did a documented behaviour change without a doc
update, did a security boundary change.

Findings are returned structured (severity, file, line, title,
body) so the caller can present them, and separately decide
whether to post them as review comments on GitHub. Posting is
deliberately not part of this module: a comment on a public PR
speaks as the user, so it needs the user's go-ahead for that
specific PR.
"""

import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional

from .ollama import sync_query_raw
from .review_verify import verify_findings

# The git executable review_branch shells out to. Bare "git"
# works everywhere except the PC, where the scoop shim fails when
# spawned from a Python subprocess -- there the caller sets
# AGENT_FARM_GIT to the direct git.exe path.
GIT_EXE = os.environ.get("AGENT_FARM_GIT", "git")
from .state import MODEL_TIERS

REVIEW_MODEL = MODEL_TIERS["synthesizer"]

# A hunk header looks like: @@ -12,7 +12,9 @@ optional context
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# A finding block the model is asked to emit:
#   [SEVERITY] Title
#   File: path
#   Line: 123
#   Body: ...
_FINDING_RE = re.compile(
    r"\[(HIGH|MEDIUM|LOW)\]\s*(.+?)\nFile:\s*(.+?)\nLine:\s*(\d+)\nBody:\s*(.+?)(?=\n\[HIGH\]|\n\[MEDIUM\]|\n\[LOW\]|\Z)",
    re.DOTALL,
)

# Real review findings from Sourcery on this repo, shown to the
# reviewer as examples of what a good finding looks like. Each
# one names a concrete bug in the diff, with the file and line.
# These are the standard the reviewer is held to: a finding must
# be this specific, this grounded in the changed lines, and this
# actionable. A vague or invented finding is worse than none.
REVIEW_EXAMPLES = """EXAMPLE 1 -- a real finding (PR #24, Sourcery):
Diff:
```diff
@@ -428,6 +428,10 @@
             card["total_seconds"] += max(0.0, elapsed)
+            if correct is not None:
+                card["checked"] += 1
+                if correct:
+                    card["correct"] += 1
             entry = card["per_task_type"][task_type]
```
Reviewer found:
[MEDIUM] Failed answers inflate correctness
File: agent_farm/colony_profiles.py
Line: 431
Body: When a caller passes a non-None correct value together with success=False, record_colony_run increments checked and possibly correct whenever correct is non-None, even when success is false. A caller that supplies both values records a failed response as a checked answer, skewing the correctness score used for routing. Only record correctness when success is true, or reject inconsistent arguments.

EXAMPLE 2 -- a real finding (PR #23, Sourcery):
Diff:
```diff
@@ -38,6 +38,15 @@
+    if task_type == "write":
+        norm = _norm(response)
+        return re.search(
+            r"return\\(?[a-z_][a-z0-9_]*\\+[a-z_][a-z0-9_]*\\)?", norm
+        ) is not None
```
Reviewer found:
[MEDIUM] Unrelated code passes write check
File: agent_farm/probes.py
Line: 41
Body: The write checker accepts any code containing a return of a sum, so an unrelated function that merely contains a sum passes. The context names the function "add", so the answer must define add. Require the normalized text to contain "defadd" as well as the return-sum pattern.

"""

REVIEW_PROMPT = """You are a senior code reviewer. Review the diff below the way a careful human reviewer would: read it line by line and ask what breaks.

CHECKLIST -- ask each of these about every changed line:
- What inputs can this actually receive? Not just the ones on this machine: weird env vars, paths with spaces, custom flags, a different OS.
- Would the test fail if this code were wrong? If the test would pass anyway, the test is not proving anything -- say so.
- Did a documented behaviour change without a doc update?
- Did a security boundary change? If so, say so plainly.

RULES:
- Report ONLY concrete issues in THIS diff. Do not give generic advice, do not praise the code, do not restate what the diff does.
- Every finding must cite the file and the NEW-file line number of the changed line it is about.
- Severity: HIGH for something that breaks, corrupts data, or opens a security hole. MEDIUM for a real bug, a gap in test coverage, or a wrong assumption. LOW for a minor issue worth fixing.
- If you find no issues, output exactly: No issues found
- Output one block per finding, in exactly this format:

[HIGH] Short title
File: path/to/file.py
Line: 123
Body: What is wrong, why it matters, and what to do about it.

{examples}
DIFF TO REVIEW:
```diff
{diff}
```

FULL FILE (the changed file after this diff, so you can see what the diff does not show -- how a key is initialized, whether a function is called elsewhere, what a test covers):
```
{file_content}
```
"""

MAX_DIFF_CHARS = 60000
MAX_FILE_CHARS = 40000
MAX_FILES = 20


def parse_unified_diff(diff_text: str) -> List[Dict[str, Any]]:
    """Split a unified diff into one entry per changed file.

    Each entry has the path, the raw diff text for that file, and
    the new-file line number each added line lands on, so a
    finding can cite a real line. A malformed diff returns an
    empty list instead of raising: the caller reports no files.
    """
    if not diff_text or not isinstance(diff_text, str):
        return []
    files: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    new_line = 0
    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)
            current = {"path": "", "diff": [line], "added_lines": {}}
            new_line = 0
            continue
        if current is None:
            continue
        current["diff"].append(line)
        if line.startswith("+++ "):
            # +++ b/path/to/file -- the path, without the b/ prefix.
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current["path"] = path
            continue
        match = _HUNK_RE.match(line)
        if match:
            new_line = int(match.group(2))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current["added_lines"][new_line] = line[1:]
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Removed line: the new-file line number does not move.
            continue
        elif line.startswith(" ") or line == "":
            new_line += 1
    if current is not None:
        files.append(current)
    # Drop entries with no path (a diff that never had a +++ line).
    return [f for f in files if f["path"]]


def parse_findings(text: str, default_path: str = "") -> List[Dict[str, Any]]:
    """Parse the model's finding blocks into structured findings.

    A block that does not match the format is returned as one
    LOW finding carrying the raw text, so nothing the model said
    is silently dropped. "No issues found" returns an empty list.
    """
    if not text or not isinstance(text, str):
        return []
    if "No issues found" in text:
        return []
    findings: List[Dict[str, Any]] = []
    for match in _FINDING_RE.finditer(text):
        severity, title, path, line, body = match.groups()
        findings.append(
            {
                "severity": severity,
                "title": title.strip(),
                "file": (path or default_path).strip(),
                "line": int(line),
                "body": body.strip(),
            }
        )
    if not findings:
        # The model did not follow the format. Keep its words
        # rather than dropping them.
        cleaned = text.strip()
        if cleaned and not cleaned.startswith("Error:"):
            findings.append(
                {
                    "severity": "LOW",
                    "title": "Unstructured review output",
                    "file": default_path,
                    "line": 0,
                    "body": cleaned,
                }
            )
    return findings


def review_file_diff(
    path: str,
    diff_text: str,
    query_fn: Callable[[str, str], str] = sync_query_raw,
    model: str = REVIEW_MODEL,
    file_content: str = "",
) -> List[Dict[str, Any]]:
    """Review one file's diff. Returns structured findings.

    file_content is the file after the diff, so the reviewer can
    see what the diff alone does not show (how a key is
    initialized, whether a function is called elsewhere). Without
    it the reviewer can only guess, and it will invent findings
    about things the rest of the file already handles.
    """
    if not diff_text:
        return []
    prompt = REVIEW_PROMPT.format(
        examples=REVIEW_EXAMPLES,
        diff=diff_text[:MAX_DIFF_CHARS],
        file_content=file_content[:MAX_FILE_CHARS],
    )
    try:
        response = query_fn(model, prompt)
    except Exception as exc:  # noqa: BLE001 - a review must not raise
        return [
            {
                "severity": "LOW",
                "title": "Review failed",
                "file": path,
                "line": 0,
                "body": f"The reviewer could not be reached: {exc}",
            }
        ]
    if isinstance(response, dict):
        return [
            {
                "severity": "LOW",
                "title": "Review failed",
                "file": path,
                "line": 0,
                "body": f"The reviewer returned an error: {response.get('error', 'unknown')}",
            }
        ]
    return parse_findings(response, default_path=path)


def review_diff(
    diff_text: str,
    query_fn: Callable[[str, str], str] = sync_query_raw,
    model: str = REVIEW_MODEL,
    max_files: int = MAX_FILES,
    file_contents: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Review a whole unified diff. Returns files reviewed and findings.

    file_contents maps a path to the file's content after the
    diff, so the reviewer sees the full file, not just the
    changed lines.
    """
    files = parse_unified_diff(diff_text)
    if not files:
        return {"files": [], "findings": [], "error": "No changed files found in the diff."}
    contents = file_contents or {}
    findings: List[Dict[str, Any]] = []
    reviewed: List[str] = []
    for entry in files[:max_files]:
        file_diff = "\n".join(entry["diff"])
        findings.extend(
            review_file_diff(
                entry["path"],
                file_diff,
                query_fn,
                model,
                contents.get(entry["path"], ""),
            )
        )
        reviewed.append(entry["path"])
    # Highest severity first, then file, then line.
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: (order.get(f["severity"], 3), f["file"], f["line"]))
    # Verify every finding against the real files before
    # reporting it. A finding the checks prove wrong is dropped;
    # the drop is reported, not hidden, so a model that
    # hallucinates is visible in the result.
    kept, rejected = verify_findings(findings, contents)
    return {
        "files": reviewed,
        "findings": kept,
        "model": model,
        "rejected_findings": [
            {"finding": finding, "reason": reason}
            for finding, reason in rejected
        ],
    }


def review_branch(
    repo_path: str,
    base: str,
    head: str = "HEAD",
    query_fn: Callable[[str, str], str] = sync_query_raw,
    model: str = REVIEW_MODEL,
) -> Dict[str, Any]:
    """Review the diff between two refs in a local git repo.

    Returns the review result, or an error dict when git fails.
    The diff is capped so a huge PR does not blow up the prompt.
    """
    if not repo_path or not base:
        return {"files": [], "findings": [], "error": "repo_path and base are required."}
    try:
        proc = subprocess.run(
            [GIT_EXE, "-C", repo_path, "diff", f"{base}...{head}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"files": [], "findings": [], "error": f"git diff failed: {exc}"}
    if proc.returncode != 0:
        return {
            "files": [],
            "findings": [],
            "error": f"git diff failed: {proc.stderr.strip() or proc.stdout.strip()}",
        }
    # Fetch each changed file's content at head, so the reviewer
    # sees the full file. A file that cannot be read (deleted,
    # binary) is reviewed from the diff alone.
    files = parse_unified_diff(proc.stdout)
    contents: Dict[str, str] = {}
    for entry in files:
        show = subprocess.run(
            [GIT_EXE, "-C", repo_path, "show", f"{head}:{entry['path']}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if show.returncode == 0:
            contents[entry["path"]] = show.stdout
    return review_diff(proc.stdout, query_fn, model, file_contents=contents)
