"""Profile cards for the agent profiler factory (Phase 2).

A profile card is one JSON file per role + model, stored in the
repo's ``profiles/`` folder. It summarizes what real runs measured
for that role + model: how many runs, how often they produced a
usable answer, how long they took, and which task types went best
and worst. Cards are built from probe runs (see probes.py) and
from real swarm runs (recorded in colony.execute_parallel_swarm).

A card is machine-specific: the machine that measured it is part
of its filename and its contents, so a card built on one machine
never counts toward trust on another. Cards written before machine
scoping are adopted by the machine that first loads them and
re-saved under the new name.

Rules carried over from Phase 1 and from metrics.py:
- Never invent a token cost or a quality score. A card says
  "fast and usually produces a usable answer", never "good at
  code" unless a real check proves it.
- Durations accumulate unrounded; only outputs are rounded.
- Recording never raises: if the card cannot be written, the
  swarm still runs and the failure is reported, not thrown.
- A card is trusted only when every probe task type has at
  least MIN_SAMPLES_PER_TYPE samples. One fast run is luck,
  not a profile.
- A card counts a run as a success only when it produced a
  usable answer. A model call that completed but answered
  "UNCERTAIN" is recorded as a failure of kind "uncertain":
  the call worked, the answer did not. (Phase 1 per-bug
  metrics count completed calls instead; the two stores
  measure different things, and both say so.)
"""

import contextlib
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .state import MODEL_TIERS

# Locked by Kyle 2026-11-10: 5 samples per task type before a
# profile card is trusted.
MIN_SAMPLES_PER_TYPE = 5

# The task types the probe harness measures, one job each.
PROBE_TASK_TYPES = ["write", "analyze", "search", "check", "list"]

# Every task type, including runs that match no known type.
TASK_TYPES = PROBE_TASK_TYPES + ["general"]

# One keyword table shared by routing (colony.get_best_bug_for_task)
# and by profile recording (task_type_for_prompt). A prompt must land
# in the same bucket in both places, or real-run statistics mix with
# the wrong task type. Each task type names the role that does it.
TASK_TYPE_KEYWORDS = {
    "write": (
        ["write", "create file", "save to", "output to", "append to",
         "http", "api", "fetch", "request", "post to", "get from"],
        "worker",
    ),
    "analyze": (["analyze code", "review code", "code quality"], "learner"),
    "search": (
        ["kmkb", "knowledge", "search knowledge", "ask knowledge", "search"],
        "memory",
    ),
    "check": (
        ["monitor", "health check", "system status", "check service"],
        "guardian",
    ),
    "list": (["list", "find", "explore", "scan", "read file", "check if"], "scout"),
}
TASK_TYPE_ROLES = {
    task_type: role for task_type, (_, role) in TASK_TYPE_KEYWORDS.items()
}

_LOCK = threading.Lock()


def machine_slug() -> str:
    """Short stable id for this machine, used to scope cards.

    Cards are machine-specific measurements: a card built on one
    machine must never count toward trust on another, so the
    machine is part of the card's filename and its contents.
    """
    try:
        name = socket.gethostname() or "unknown"
    except Exception:
        name = "unknown"
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.lower())
    slug = "-".join(part for part in slug.split("-") if part)[:32]
    return slug or "unknown"


def profiles_dir() -> Path:
    """Where profile cards live.

    The repo's ``profiles/`` folder by default; tests point it
    elsewhere with AGENT_FARM_PROFILES_DIR.
    """
    override = os.environ.get("AGENT_FARM_PROFILES_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "profiles"


def _card_path(role: str, model: str) -> Path:
    """Card file for a role + model on this machine.

    The model hash keeps two model names that normalize to the
    same string (for example "qwen3:4b" and "qwen3-4b") in
    separate files, and the machine slug keeps one machine's
    measurements out of another machine's cards.
    """
    safe_model = model.replace(":", "-").replace("/", "-")
    digest = hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]
    return profiles_dir() / f"{role}-{safe_model}-{digest}-{machine_slug()}.json"


def _legacy_card_path(role: str, model: str) -> Path:
    """The pre-machine-scope card filename, for one-time migration."""
    safe_model = model.replace(":", "-").replace("/", "-")
    return profiles_dir() / f"{role}-{safe_model}.json"


def create_profile(role: str, model: str) -> Dict[str, Any]:
    """Build a fresh, empty profile card for a role + model.

    This only builds the card in memory; nothing is written
    until a run is recorded for it.
    """
    per_type = {
        task_type: {
            "samples": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "total_seconds": 0.0,
            "avg_seconds": 0.0,
            "trusted": False,
            # Correctness: how many usable answers were checked
            # against the job's known answer, and how many were
            # right. A failed response is not checked, so checked
            # is usually smaller than samples.
            "checked": 0,
            "correct": 0,
            "correctness_rate": 0.0,
        }
        for task_type in TASK_TYPES
    }
    return {
        "role": role,
        "model": model,
        "machine": machine_slug(),
        "samples": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "total_seconds": 0.0,
        "avg_seconds": 0.0,
        "checked": 0,
        "correct": 0,
        "correctness_rate": 0.0,
        "per_task_type": per_type,
        "failure_modes": {},
        "best_task_types": [],
        "worst_task_types": [],
        "trusted": False,
        "stale": False,
        "stale_reason": None,
        "last_updated": None,
    }


def _refresh_card(card: Dict[str, Any]) -> None:
    """Recompute the derived fields of a card from its counts."""
    samples = card["samples"]
    card["success_rate"] = round(card["successes"] / samples, 2) if samples else 0.0
    card["avg_seconds"] = round(card["total_seconds"] / samples, 2) if samples else 0.0
    checked = card.get("checked", 0)
    card["correctness_rate"] = (
        round(card.get("correct", 0) / checked, 2) if checked else 0.0
    )
    for task_type, entry in card["per_task_type"].items():
        type_samples = entry["samples"]
        entry["success_rate"] = (
            round(entry["successes"] / type_samples, 2) if type_samples else 0.0
        )
        entry["avg_seconds"] = (
            round(entry["total_seconds"] / type_samples, 2) if type_samples else 0.0
        )
        entry["trusted"] = type_samples >= MIN_SAMPLES_PER_TYPE
        type_checked = entry.get("checked", 0)
        entry["correctness_rate"] = (
            round(entry.get("correct", 0) / type_checked, 2) if type_checked else 0.0
        )
    # A card is trusted only when every probe task type has
    # enough samples, and only while it is not stale. A retired
    # card stays untrusted even if more runs land on it: its
    # measurements are no longer the ones to route on.
    # "general" runs do not count toward trust: they are
    # unclassified work, not a measured task type.
    card["trusted"] = (
        not card["stale"]
        and all(
            card["per_task_type"][task_type]["samples"] >= MIN_SAMPLES_PER_TYPE
            for task_type in PROBE_TASK_TYPES
        )
    )
    ranked = [
        (task_type, entry)
        for task_type, entry in card["per_task_type"].items()
        if entry["samples"] > 0
    ]
    # Best first: highest success rate, then fastest.
    ranked.sort(key=lambda item: (-item[1]["success_rate"], item[1]["avg_seconds"]))
    card["best_task_types"] = [task_type for task_type, _ in ranked]
    card["worst_task_types"] = [task_type for task_type, _ in reversed(ranked)]
    card["last_updated"] = datetime.now(timezone.utc).isoformat()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_card(card: Any, role: str, model: str) -> Optional[Dict[str, Any]]:
    """Validate a loaded card and fill or replace bad fields.

    A card written by an older version, or hand-edited into a bad
    shape, must not crash profiling: missing or mistyped fields
    are replaced with fresh defaults. A card whose stored role or
    model does not match the file it was loaded for is rejected
    outright: it belongs to another role or model. Returns None
    when the card is not usable at all.
    """
    if not isinstance(card, dict):
        return None
    if card.get("role") != role or card.get("model") != model:
        return None
    fresh = create_profile(role, model)
    machine = card.get("machine")
    # A card from before machine scoping has no machine field; it
    # is adopted by the machine that first loads it.
    card["machine"] = machine if isinstance(machine, str) and machine else machine_slug()
    for key in ("samples", "successes", "failures"):
        if not _is_int(card.get(key)):
            card[key] = 0
    for key in ("success_rate", "total_seconds", "avg_seconds"):
        if not _is_number(card.get(key)):
            card[key] = 0.0
    # Cards written before correctness was measured have no
    # checked / correct fields; they start at zero, not as a
    # broken card.
    for key in ("checked", "correct"):
        if not _is_int(card.get(key)):
            card[key] = 0
    if not _is_number(card.get("correctness_rate")):
        card["correctness_rate"] = 0.0
    per_type = card.get("per_task_type")
    if not isinstance(per_type, dict):
        per_type = {}
        card["per_task_type"] = per_type
    for task_type in TASK_TYPES:
        entry = per_type.get(task_type)
        if not isinstance(entry, dict):
            per_type[task_type] = fresh["per_task_type"][task_type]
            continue
        for key in ("samples", "successes", "failures"):
            if not _is_int(entry.get(key)):
                entry[key] = 0
        for key in ("success_rate", "total_seconds", "avg_seconds"):
            if not _is_number(entry.get(key)):
                entry[key] = 0.0
        for key in ("checked", "correct"):
            if not _is_int(entry.get(key)):
                entry[key] = 0
        if not _is_number(entry.get("correctness_rate")):
            entry["correctness_rate"] = 0.0
        if not isinstance(entry.get("trusted"), bool):
            entry["trusted"] = False
    modes = card.get("failure_modes")
    if not isinstance(modes, dict) or not all(
        isinstance(key, str) and _is_int(value) for key, value in modes.items()
    ):
        card["failure_modes"] = {}
    for key in ("best_task_types", "worst_task_types"):
        value = card.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            card[key] = []
    for key in ("trusted", "stale"):
        if not isinstance(card.get(key), bool):
            card[key] = False
    if card.get("stale_reason") is not None and not isinstance(
        card.get("stale_reason"), str
    ):
        card["stale_reason"] = None
    if card.get("last_updated") is not None and not isinstance(
        card.get("last_updated"), str
    ):
        card["last_updated"] = None
    return card


def _load_card(role: str, model: str) -> Dict[str, Any]:
    """Load a card from disk, or a fresh one if none is stored.

    Tries the current machine-scoped filename first, then the
    legacy filename from before machine scoping: a legacy card is
    adopted by this machine and re-saved under the new name on
    the next write, so the old file does not linger as a duplicate.
    A malformed or mismatched card is replaced with a fresh one,
    never raised.
    """
    path = _card_path(role, model)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except (json.JSONDecodeError, OSError):
            return create_profile(role, model)
        normalized = _normalize_card(card, role, model)
        return normalized if normalized is not None else create_profile(role, model)
    legacy = _legacy_card_path(role, model)
    if legacy.exists():
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                card = json.load(f)
        except (json.JSONDecodeError, OSError):
            return create_profile(role, model)
        normalized = _normalize_card(card, role, model)
        if normalized is not None:
            return normalized
    return create_profile(role, model)


@contextlib.contextmanager
def _card_lock(path: Path):
    """Best-effort cross-process lock for one card's read-modify-write.

    The in-process _LOCK only coordinates threads in this process;
    two processes recording the same card would otherwise load the
    same counts and one run would be lost. The lock is a file
    created with O_EXCL: the first process wins, the other waits
    briefly, and a lock older than 30 seconds is treated as stale
    (its process died) and removed. If the lock cannot be taken
    within 10 seconds, the update still runs rather than breaking
    the swarm: profiling must never block work.
    """
    lock_path = path.with_suffix(".lock")
    fd = None
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    # getmtime is wall-clock time, so the age
                    # check must use time.time(), not the
                    # monotonic clock used for the deadline.
                    if time.time() - os.path.getmtime(lock_path) > 30.0:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    pass
                time.sleep(0.1)
            except OSError:
                # The lock file cannot be created at all (the
                # profiles dir is missing or not writable):
                # proceed without it rather than breaking
                # the swarm. The save will report its own
                # failure.
                break
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def _save_card(card: Dict[str, Any]) -> bool:
    """Write a card to disk. Returns False instead of raising.

    The write goes to a unique temp file (no two processes share
    a name) and is moved into place, so a reader never sees a
    half-written card.
    """
    try:
        directory = profiles_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = _card_path(card["role"], card["model"])
        fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix=path.name + ".", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        # A legacy card adopted by this machine is gone once the
        # new file is in place; leaving it would list the same
        # role + model twice.
        legacy = _legacy_card_path(card["role"], card["model"])
        if legacy != path:
            try:
                legacy.unlink()
            except OSError:
                pass
        return True
    except (OSError, ValueError, TypeError):
        return False


def prune_stale_profiles() -> List[str]:
    """Delete profile cards that can never be used again.

    Removes two kinds of stale file, and nothing else:
    - A legacy card file when the machine-scoped card for the
      same role + model exists: the legacy is a duplicate, and
      the save that created the scoped card should have removed
      it, but a card that is never saved again (see below)
      leaves it behind.
    - A card for a role + model combination no current bug can
      use: its model is no longer assigned to that role in
      MODEL_TIERS, so no bug will ever match it, it will never
      gain samples, and routing will never pick it. This is what
      happens when a role's model changes (the worker moving
      from qwen3:4b to qwen2.5-coder leaves a worker-qwen3:4b
      card behind, trusted but useless).
    Only this machine's cards are removed: a file whose name
    ends with another machine's slug is left alone, and a legacy
    file is this machine's by adoption. Returns the removed
    filenames, so a caller can report what was cleaned up.
    Never raises: a read or delete failure just skips that file.
    """
    removed: List[str] = []
    try:
        files = sorted(profiles_dir().glob("*.json"))
    except OSError:
        return removed
    current = set(MODEL_TIERS.items())
    slug = machine_slug()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(card, dict):
            continue
        role = card.get("role")
        model = card.get("model")
        if not isinstance(role, str) or not isinstance(model, str):
            continue
        is_legacy = path == _legacy_card_path(role, model)
        is_mine = is_legacy or path.name.endswith(f"-{slug}.json")
        if not is_mine:
            continue
        # Legacy duplicate: the scoped card exists, so the
        # legacy file is redundant.
        if is_legacy and _card_path(role, model).exists():
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                pass
            continue
        # Orphaned combo: no bug can have this role + model.
        if (role, model) not in current:
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                pass
    return removed


def get_profile(role: str, model: str) -> Optional[Dict[str, Any]]:
    """Return a stored card, or None if none exists yet."""
    path = _card_path(role, model)
    legacy = _legacy_card_path(role, model)
    if not path.exists() and not legacy.exists():
        return None
    with _LOCK:
        return _load_card(role, model)


def list_profiles() -> List[Dict[str, Any]]:
    """Return every stored card, newest behaviour first by role."""
    directory = profiles_dir()
    cards = []
    try:
        files = sorted(directory.glob("*.json"))
    except OSError:
        return []
    with _LOCK:
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    card = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(card, dict):
                continue
            role = card.get("role")
            model = card.get("model")
            if not isinstance(role, str) or not isinstance(model, str):
                continue
            normalized = _normalize_card(card, role, model)
            if normalized is not None:
                cards.append(normalized)
    cards.sort(key=lambda c: (c["role"], c["model"]))
    return cards


def profiles_summary() -> List[Dict[str, Any]]:
    """Short summary of every card, for farm_stats()."""
    return [
        {
            "role": card["role"],
            "model": card["model"],
            "machine": card.get("machine"),
            "samples": card["samples"],
            "trusted": card["trusted"],
            "success_rate": card["success_rate"],
            "avg_seconds": card["avg_seconds"],
            "stale": card["stale"],
            "checked": card.get("checked", 0),
            "correct": card.get("correct", 0),
            "correctness_rate": card.get("correctness_rate", 0.0),
        }
        for card in list_profiles()
    ]


def retire_profile(role: str, model: str, reason: str = "model changed") -> bool:
    """Mark a card stale so routing never trusts it again.

    A card goes stale when its model changes or its samples are
    too old to trust. The card is kept on disk, not deleted, so
    the history stays reviewable. Returns False if no card exists.
    """
    with _LOCK, _card_lock(_card_path(role, model)):
        card = _load_card(role, model)
        if card["samples"] == 0 and card["last_updated"] is None:
            return False
        card["stale"] = True
        card["stale_reason"] = reason
        card["trusted"] = False
        card["last_updated"] = datetime.now(timezone.utc).isoformat()
        return _save_card(card)


def classify_result(response: Any, error: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Classify one model call as usable or not, and why not.

    Success means the call produced a usable answer. The failure
    kinds are the ones the probe sketch names: a call that raised
    or came back as an "Error: ..." string, an empty response, and
    an "UNCERTAIN" answer (the grounded model refusing for lack
    of context). The timeout kind is split out of the error kind
    because a timeout is a different problem than a bad response:
    the wording is checked in both the error argument and in an
    "Error: ..." response string, since a timeout can arrive
    through either one.
    """
    if error is not None:
        lowered = str(error).lower()
        if "timeout" in lowered or "timed out" in lowered:
            return False, "timeout"
        return False, "error"
    if not isinstance(response, str) or not response.strip():
        return False, "empty"
    stripped = response.strip()
    if stripped.startswith("Error:"):
        lowered = stripped.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return False, "timeout"
        return False, "error"
    if stripped == "UNCERTAIN" or stripped.startswith("UNCERTAIN"):
        return False, "uncertain"
    return True, None


def task_type_for_prompt(prompt: str) -> str:
    """Classify a task prompt into a probe task type.

    Uses the shared TASK_TYPE_KEYWORDS table, the same table
    colony.get_best_bug_for_task routes with, so a real run and
    a probe job land in the same bucket. A prompt that matches
    nothing is "general", not a guess at one of the measured types.
    """
    text = (prompt or "").lower()
    for task_type, (keywords, _role) in TASK_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return task_type
    return "general"


def _card_usable_for(card: Dict[str, Any], task_type: str) -> bool:
    """True when a card can drive a profile pick for a task type.

    The card must be for this machine, trusted, not stale, and
    its entry for the task type must be trusted too. The entry
    check matters: a card that claims to be trusted overall but
    has an untrusted entry for the requested type is not
    evidence for that type.
    """
    if card.get("machine") != machine_slug():
        return False
    if not card.get("trusted") or card.get("stale"):
        return False
    entry = card.get("per_task_type", {}).get(task_type)
    return bool(entry and entry.get("trusted"))


def profile_pick(task_type: str, bugs: List[Any]) -> Optional[Any]:
    """Pick a bug from measured profile evidence, or None.

    A bug is picked this way only when its role + model has a
    trusted, non-stale card for this machine and that card's
    entry for the task type is trusted too. Among several, the
    highest success rate wins, then the fastest. Anything less
    than that falls back to keyword routing: an untrusted card
    is a hint, not evidence. Cards are read from disk on each
    call; they are small JSON files and the OS caches them.
    Both routers (colony and agent) use this one function, so
    a profile pick means the same thing on either path.
    """
    best = None
    best_key = None
    for card in list_profiles():
        if not _card_usable_for(card, task_type):
            continue
        entry = card.get("per_task_type", {}).get(task_type)
        if entry is None:
            continue
        for bug in bugs:
            if bug.role == card["role"] and bug.model == card["model"]:
                key = (entry["success_rate"], -entry["avg_seconds"])
                if best_key is None or key > best_key:
                    best_key = key
                    best = bug
    return best


def record_profile_run(
    role: str,
    model: str,
    task_type: str,
    elapsed_seconds: float,
    success: bool,
    failure_kind: Optional[str] = None,
    correct: Optional[bool] = None,
) -> bool:
    """Record one measured run on a role + model's card.

    Loads the card, adds the run, recomputes its derived
    fields, and writes it back to profiles/. Never raises on
    bad input or a failed write: returns False instead, so a
    swarm is never broken by its own profiling. The load and
    the write happen under a cross-process file lock, so two
    processes recording the same card cannot lose each other's
    runs.
    """
    try:
        if not role or not model:
            return False
        elapsed = float(elapsed_seconds or 0.0)
    except (TypeError, ValueError):
        return False
    if task_type not in TASK_TYPES:
        task_type = "general"
    with _LOCK, _card_lock(_card_path(role, model)):
        card = _load_card(role, model)
        card["samples"] += 1
        if success:
            card["successes"] += 1
        else:
            card["failures"] += 1
            if failure_kind:
                modes = card["failure_modes"]
                modes[failure_kind] = modes.get(failure_kind, 0) + 1
        card["total_seconds"] += max(0.0, elapsed)
        # Correctness is recorded only when the answer was
        # checked (a usable answer with a checker) AND the run
        # succeeded. A failed response is a failure, not a
        # checked answer, even if a caller passes a correct
        # value alongside success=False. An uncheckable task
        # type leaves correctness untouched.
        correct = correct if success else None
        if correct is not None:
            card["checked"] += 1
            if correct:
                card["correct"] += 1
        entry = card["per_task_type"][task_type]
        entry["samples"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        if correct is not None:
            entry["checked"] += 1
            if correct:
                entry["correct"] += 1
        entry["total_seconds"] += max(0.0, elapsed)
        _refresh_card(card)
        return _save_card(card)
