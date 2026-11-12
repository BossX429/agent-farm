"""Colony-type profile cards (Phase 3: colony-type measurement).

Same pattern as the bug profiler (profiles.py), one level up:
a colony-type card is one JSON file per colony type in the
repo's ``colony_profiles/`` folder, built from real swarm runs.
A card is trusted only at MIN_SAMPLES_PER_TYPE samples per
probe task type, and only while it is not stale. Each swarm run
feeds its colony type's card once per task, so a card's
per-task-type entries say how that colony type did on that kind
of work. colony_type_pick() returns the best colony type for a
task type from trusted cards, or None: the caller falls back to
its old default, so behaviour is unchanged until cards are
trusted.

Rules carried over from the bug profiler: never invent a token
cost or a quality score; durations accumulate unrounded, only
outputs are rounded; recording never raises; a run counts as a
success only when it produced a usable answer (classify_result
from profiles.py: a call that answered "UNCERTAIN" is a failure
of kind "uncertain"); cards are machine-specific and
colony_profiles/ is gitignored, so no card is ever committed.
"""

import contextlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .profiles import (
    MIN_SAMPLES_PER_TYPE,
    PROBE_TASK_TYPES,
    TASK_TYPES,
    machine_slug,
)

# The colony types spawn_colony_internal builds. A card for a
# type not in this list is orphaned and gets pruned.
COLONY_TYPES = ["standard", "fast", "heavy", "hybrid"]

_LOCK = threading.Lock()


def colony_profiles_dir() -> Path:
    """Where colony-type cards live.

    The repo's ``colony_profiles/`` folder by default; tests
    point it elsewhere with AGENT_FARM_COLONY_PROFILES_DIR.
    """
    override = os.environ.get("AGENT_FARM_COLONY_PROFILES_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "colony_profiles"


def _colony_card_path(colony_type: str) -> Path:
    """Card file for a colony type on this machine."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in colony_type.lower())
    safe = "-".join(part for part in safe.split("-") if part) or "unknown"
    return colony_profiles_dir() / f"colony-{safe}-{machine_slug()}.json"


def create_colony_profile(colony_type: str) -> Dict[str, Any]:
    """Build a fresh, empty colony-type card.

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
        }
        for task_type in TASK_TYPES
    }
    return {
        "colony_type": colony_type,
        "machine": machine_slug(),
        "samples": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "total_seconds": 0.0,
        "avg_seconds": 0.0,
        "per_task_type": per_type,
        "failure_modes": {},
        "trusted": False,
        "stale": False,
        "stale_reason": None,
        "last_updated": None,
    }


def _refresh_colony_card(card: Dict[str, Any]) -> None:
    """Recompute the derived fields of a colony-type card."""
    samples = card["samples"]
    card["success_rate"] = round(card["successes"] / samples, 2) if samples else 0.0
    card["avg_seconds"] = round(card["total_seconds"] / samples, 2) if samples else 0.0
    for task_type, entry in card["per_task_type"].items():
        type_samples = entry["samples"]
        entry["success_rate"] = (
            round(entry["successes"] / type_samples, 2) if type_samples else 0.0
        )
        entry["avg_seconds"] = (
            round(entry["total_seconds"] / type_samples, 2) if type_samples else 0.0
        )
        entry["trusted"] = type_samples >= MIN_SAMPLES_PER_TYPE
    # Trusted only when every probe task type has enough samples,
    # and only while it is not stale. "general" runs do not count
    # toward trust: they are unclassified work, not a measured
    # task type.
    card["trusted"] = (
        not card["stale"]
        and all(
            card["per_task_type"][task_type]["samples"] >= MIN_SAMPLES_PER_TYPE
            for task_type in PROBE_TASK_TYPES
        )
    )
    card["last_updated"] = datetime.now(timezone.utc).isoformat()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_colony_card(card: Any, colony_type: str) -> Optional[Dict[str, Any]]:
    """Validate a loaded card and fill or replace bad fields.

    Same contract as profiles._normalize_card. A card whose
    stored colony type does not match the file it was loaded
    for is rejected outright. Returns None when unusable.
    """
    if not isinstance(card, dict):
        return None
    if card.get("colony_type") != colony_type:
        return None
    fresh = create_colony_profile(colony_type)
    machine = card.get("machine")
    card["machine"] = machine if isinstance(machine, str) and machine else machine_slug()
    for key in ("samples", "successes", "failures"):
        if not _is_int(card.get(key)):
            card[key] = 0
    for key in ("success_rate", "total_seconds", "avg_seconds"):
        if not _is_number(card.get(key)):
            card[key] = 0.0
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
        if not isinstance(entry.get("trusted"), bool):
            entry["trusted"] = False
    modes = card.get("failure_modes")
    if not isinstance(modes, dict) or not all(
        isinstance(key, str) and _is_int(value) for key, value in modes.items()
    ):
        card["failure_modes"] = {}
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


def _load_colony_card(colony_type: str) -> Dict[str, Any]:
    """Load a card from disk, or a fresh one if none is stored."""
    path = _colony_card_path(colony_type)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except (json.JSONDecodeError, OSError):
            return create_colony_profile(colony_type)
        normalized = _normalize_colony_card(card, colony_type)
        return normalized if normalized is not None else create_colony_profile(colony_type)
    return create_colony_profile(colony_type)


@contextlib.contextmanager
def _colony_card_lock(path: Path):
    """Best-effort cross-process lock for one card's read-modify-write.

    Same contract as profiles._card_lock: a lock file created
    with O_EXCL, a lock older than 30 seconds is treated as
    stale and removed, and if the lock cannot be taken within
    10 seconds the update still runs rather than breaking the
    swarm.
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


def _save_colony_card(card: Dict[str, Any]) -> bool:
    """Write a card to disk. Returns False instead of raising."""
    try:
        directory = colony_profiles_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = _colony_card_path(card["colony_type"])
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
        return True
    except (OSError, ValueError, TypeError):
        return False


def get_colony_profile(colony_type: str) -> Optional[Dict[str, Any]]:
    """Return a stored card, or None if none exists."""
    path = _colony_card_path(colony_type)
    if not path.exists():
        return None
    with _LOCK:
        return _load_colony_card(colony_type)


def list_colony_profiles() -> List[Dict[str, Any]]:
    """Return every stored colony-type card."""
    directory = colony_profiles_dir()
    cards = []
    try:
        files = sorted(directory.glob("colony-*.json"))
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
            colony_type = card.get("colony_type")
            if not isinstance(colony_type, str):
                continue
            normalized = _normalize_colony_card(card, colony_type)
            if normalized is not None:
                cards.append(normalized)
    cards.sort(key=lambda c: c["colony_type"])
    return cards


def colony_profiles_summary() -> List[Dict[str, Any]]:
    """One summary line per card, for farm_stats()."""
    return [
        {
            "colony_type": card["colony_type"],
            "machine": card.get("machine"),
            "samples": card["samples"],
            "trusted": card["trusted"],
            "success_rate": card["success_rate"],
            "avg_seconds": card["avg_seconds"],
            "stale": card["stale"],
        }
        for card in list_colony_profiles()
    ]


def record_colony_run(
    colony_type: str,
    task_type: str,
    elapsed_seconds: float,
    success: bool,
    failure_kind: Optional[str] = None,
) -> bool:
    """Record one measured task run on a colony type's card.

    Loads the card, adds the run, recomputes its derived fields,
    and writes it back. Never raises on bad input or a failed
    write: returns False instead, so a swarm is never broken by
    its own profiling.
    """
    try:
        if not colony_type:
            return False
        elapsed = float(elapsed_seconds or 0.0)
    except (TypeError, ValueError):
        return False
    if task_type not in TASK_TYPES:
        task_type = "general"
    with _LOCK, _colony_card_lock(_colony_card_path(colony_type)):
        card = _load_colony_card(colony_type)
        card["samples"] += 1
        if success:
            card["successes"] += 1
        else:
            card["failures"] += 1
            if failure_kind:
                modes = card["failure_modes"]
                modes[failure_kind] = modes.get(failure_kind, 0) + 1
        card["total_seconds"] += max(0.0, elapsed)
        entry = card["per_task_type"][task_type]
        entry["samples"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["total_seconds"] += max(0.0, elapsed)
        _refresh_colony_card(card)
        return _save_colony_card(card)


def _colony_card_usable_for(card: Dict[str, Any], task_type: str) -> bool:
    """True when a card can drive a colony-type pick for a task type."""
    if card.get("machine") != machine_slug():
        return False
    if not card.get("trusted") or card.get("stale"):
        return False
    entry = card.get("per_task_type", {}).get(task_type)
    return bool(entry and entry.get("trusted"))


def colony_type_pick(task_type: str) -> Optional[str]:
    """Pick the best colony type for a task type from measured evidence.

    Among types with a trusted, non-stale card for this machine
    whose entry for the task type is trusted, the highest success
    rate wins, then the fastest. Returns None when no card
    qualifies: the caller falls back to its old default, so
    behaviour is unchanged until cards are trusted.
    """
    best = None
    best_key = None
    for card in list_colony_profiles():
        if not _colony_card_usable_for(card, task_type):
            continue
        entry = card.get("per_task_type", {}).get(task_type)
        if entry is None:
            continue
        key = (entry["success_rate"], -entry["avg_seconds"])
        if best_key is None or key > best_key:
            best_key = key
            best = card["colony_type"]
    return best


def dominant_task_type(tasks: List[Dict[str, str]]) -> str:
    """The most common task type among a task list.

    Used to pick a colony type for a batch of tasks: the type
    the batch mostly is. A tie goes to the type seen first.
    """
    from .profiles import task_type_for_prompt

    counts: Dict[str, int] = {}
    for task in tasks or []:
        task_type = task_type_for_prompt(task.get("prompt", ""))
        counts[task_type] = counts.get(task_type, 0) + 1
    if not counts:
        return "general"
    return max(counts, key=lambda key: counts[key])


def resolve_colony_type(requested: str, tasks: List[Dict[str, str]], default: str):
    """Resolve the colony type to spawn.

    When requested is "auto", pick the best colony type for the
    tasks' dominant task type from trusted cards; fall back to
    default when no card qualifies. Otherwise the requested type
    is used as-is. Returns (colony_type, was_profile_picked).
    """
    if requested != "auto":
        return requested, False
    picked = colony_type_pick(dominant_task_type(tasks))
    if picked is not None:
        return picked, True
    return default, False


def retire_colony_profile(colony_type: str, reason: str = "type removed") -> bool:
    """Mark a card stale so it is never picked again.

    The card is kept on disk, not deleted. Returns False if no
    card exists.
    """
    with _LOCK, _colony_card_lock(_colony_card_path(colony_type)):
        card = _load_colony_card(colony_type)
        if card["samples"] == 0 and card["last_updated"] is None:
            return False
        card["stale"] = True
        card["stale_reason"] = reason
        card["trusted"] = False
        card["last_updated"] = datetime.now(timezone.utc).isoformat()
        return _save_colony_card(card)


def prune_stale_colony_profiles() -> List[str]:
    """Delete colony-type cards that can never be used again.

    Removes cards for a type no longer built (not in
    COLONY_TYPES), this machine's only. Returns the removed
    filenames. Never raises.
    """
    removed: List[str] = []
    try:
        files = sorted(colony_profiles_dir().glob("colony-*.json"))
    except OSError:
        return removed
    slug = machine_slug()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(card, dict):
            continue
        colony_type = card.get("colony_type")
        if not isinstance(colony_type, str):
            continue
        if colony_type in COLONY_TYPES:
            continue
        if not path.name.endswith(f"-{slug}.json"):
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return removed
