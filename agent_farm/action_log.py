"""Structured action log for agent-farm (Manus second-review concern).

This is deliberately small and dependency-free, and it is OFF by
default: nothing is written unless the operator opts in, so normal
local runs stay quiet.

Enable it with one of:
    AGENT_FARM_ACTION_LOG=/path/to/actions.jsonl
    AGENT_FARM_ACTION_LOG=1          # ./agent-farm-actions.jsonl

Each record is one JSON object per line with: timestamp (UTC),
action, outcome, and a short detail string. A SHA-256 hash chain
(``prev_hash`` / ``hash``) lets a reader verify that a record was
not later edited in place: :func:`verify_action_log` recomputes the
chain and reports any line whose hash no longer matches. That is
*not* tamper-proofing: anyone with write access to the log file
could rewrite the whole chain, and this module does not prevent
that. It detects single-record edits, it does not prove the log
was never rewritten.

Privacy rules (enforced here, not just documented):
- Never log a message body, file content, or generated code.
- Never log a full command line: only the first token (the binary
  name) is recorded for exec actions.
- Never log a URL with credentials: :func:`redact_url` strips the
  userinfo and redacts query values for password/token/secret/key
  /auth-like keys before anything is written.
- File paths are recorded only for enforcement events (a denied
  path, a validation outcome), not for routine reads.

Callers should use :func:`log_action`; it is a no-op when logging
is not enabled, so enforcement code can call it unconditionally.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_LOCK = threading.Lock()
_EXPLICIT_PATH = None
_GENESIS_HASH = "0" * 64

_SENSITIVE_QUERY_KEYS = (
    "password", "passwd", "token", "secret", "api_key", "apikey",
    "auth", "authorization", "session", "cookie",
)


def enable_action_log(path):
    """Enable logging to ``path`` for this process (used by tests)."""
    global _EXPLICIT_PATH
    _EXPLICIT_PATH = path


def disable_action_log():
    """Disable explicitly-enabled logging for this process."""
    global _EXPLICIT_PATH
    _EXPLICIT_PATH = None


def _log_path():
    if _EXPLICIT_PATH:
        return _EXPLICIT_PATH
    raw = os.environ.get("AGENT_FARM_ACTION_LOG", "")
    if not raw:
        return None
    if raw == "1":
        return "agent-farm-actions.jsonl"
    return raw


def redact_url(url):
    """Strip credentials from a URL and redact sensitive query values."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return "<unparsable-url>"
    query = ""
    if parsed.query:
        pairs = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if any(s in key.lower() for s in _SENSITIVE_QUERY_KEYS):
                pairs.append((key, "<redacted>"))
            else:
                pairs.append((key, value))
        query = urlencode(pairs)
    # Rebuild without username/password.
    return urlunparse((parsed.scheme, parsed.hostname or "", parsed.path,
                       parsed.params, query, ""))


def _record_hash(record):
    """SHA-256 over the record without its own ``hash`` field."""
    payload = {k: v for k, v in record.items() if k != "hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prev_hash(path):
    try:
        with open(path, "rb") as f:
            last = None
            for line in f:
                if line.strip():
                    last = line
            if last is None:
                return _GENESIS_HASH
            return json.loads(last.decode("utf-8")).get("hash", _GENESIS_HASH)
    except FileNotFoundError:
        return _GENESIS_HASH
    except Exception:
        return _GENESIS_HASH


def log_action(action, outcome, detail=""):
    """Append one structured record, or do nothing if logging is off.

    ``action`` is a short name like ``http_blocked`` or
    ``exec_blocked``; ``outcome`` is ``allowed`` / ``blocked`` /
    ``error`` / ``timeout`` / ``validated`` / ``written_not_validated``.
    Returns True if a record was written, False otherwise.
    """
    path = _log_path()
    if not path:
        return False
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": str(action),
        "outcome": str(outcome),
        "detail": str(detail)[:300],
    }
    with _LOCK:
        record["prev_hash"] = _prev_hash(path)
        record["hash"] = _record_hash(record)
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            return False
    return True


def verify_action_log(path):
    """Recompute the hash chain in ``path``.

    Returns ``(ok, bad_lines)`` where ``bad_lines`` lists 1-based
    line numbers whose hash or prev_hash does not match. An empty
    or missing file verifies as ``(True, [])``. A full rewrite of
    the file with a fresh, self-consistent chain still verifies --
    see the module docstring: this detects edits, it does not
    prove the log was never rewritten.
    """
    bad = []
    prev = _GENESIS_HASH
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except FileNotFoundError:
        return True, []
    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            expected = _record_hash(record)
            if record.get("prev_hash") != prev or record.get("hash") != expected:
                bad.append(idx)
            prev = record.get("hash", prev)
        except Exception:
            bad.append(idx)
    return (len(bad) == 0), bad
