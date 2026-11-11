"""Profile cards for the agent profiler factory (Phase 2).

A profile card is one JSON file per role + model, stored in the
repo's ``profiles/`` folder. It summarizes what real runs measured
for that role + model: how many runs, how often they produced a
usable answer, how long they took, and which task types went best
and worst. Cards are built from probe runs (see probes.py) and
from real swarm runs (recorded in colony.execute_parallel_swarm).

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

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Locked by Kyle 2026-11-10: 5 samples per task type before a
# profile card is trusted.
MIN_SAMPLES_PER_TYPE = 5

# The task types the probe harness measures, one job each.
PROBE_TASK_TYPES = ["write", "analyze", "search", "check", "list"]

# Every task type, including runs that match no known type.
TASK_TYPES = PROBE_TASK_TYPES + ["general"]

_LOCK = threading.Lock()


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
        }
        for task_type in TASK_TYPES
    }
    return {
        "role": role,
        "model": model,
        "samples": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "total_seconds": 0.0,
        "avg_seconds": 0.0,
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
    for task_type, entry in card["per_task_type"].items():
        type_samples = entry["samples"]
        entry["success_rate"] = (
            round(entry["successes"] / type_samples, 2) if type_samples else 0.0
        )
        entry["avg_seconds"] = (
            round(entry["total_seconds"] / type_samples, 2) if type_samples else 0.0
        )
        entry["trusted"] = type_samples >= MIN_SAMPLES_PER_TYPE
    # A card is trusted only when every probe task type has
    # enough samples. "general" runs do not count toward trust:
    # they are unclassified work, not a measured task type.
    card["trusted"] = all(
        card["per_task_type"][task_type]["samples"] >= MIN_SAMPLES_PER_TYPE
        for task_type in PROBE_TASK_TYPES
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


def _load_card(role: str, model: str) -> Dict[str, Any]:
    """Load a card from disk, or a fresh one if none is stored."""
    path = _card_path(role, model)
    try:
        with open(path, "r", encoding="utf-8") as f:
            card = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return create_profile(role, model)
    if not isinstance(card, dict) or card.get("role") != role:
        return create_profile(role, model)
    # A card written by an older version may lack fields added
    # since; fill the gaps from a fresh card instead of crashing.
    fresh = create_profile(role, model)
    for key, value in fresh.items():
        card.setdefault(key, value)
    for task_type in TASK_TYPES:
        card["per_task_type"].setdefault(task_type, fresh["per_task_type"][task_type])
    return card


def _save_card(card: Dict[str, Any]) -> bool:
    """Write a card to disk. Returns False instead of raising."""
    try:
        directory = profiles_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = _card_path(card["role"], card["model"])
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
        return True
    except (OSError, ValueError, TypeError):
        return False


def get_profile(role: str, model: str) -> Optional[Dict[str, Any]]:
    """Return a stored card, or None if none exists yet."""
    path = _card_path(role, model)
    if not path.exists():
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
            if isinstance(card, dict) and card.get("role") and card.get("model"):
                cards.append(card)
    cards.sort(key=lambda c: (c["role"], c["model"]))
    return cards


def profiles_summary() -> List[Dict[str, Any]]:
    """Short summary of every card, for farm_stats()."""
    return [
        {
            "role": card["role"],
            "model": card["model"],
            "samples": card["samples"],
            "trusted": card["trusted"],
            "success_rate": card["success_rate"],
            "avg_seconds": card["avg_seconds"],
            "stale": card["stale"],
        }
        for card in list_profiles()
    ]


def retire_profile(role: str, model: str, reason: str = "model changed") -> bool:
    """Mark a card stale so routing never trusts it again.

    A card goes stale when its model changes or its samples are
    too old to trust. The card is kept on disk, not deleted, so
    the history stays reviewable. Returns False if no card exists.
    """
    with _LOCK:
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
    because a timeout is a different problem than a bad response.
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
        return False, "error"
    if stripped == "UNCERTAIN" or stripped.startswith("UNCERTAIN"):
        return False, "uncertain"
    return True, None


def task_type_for_prompt(prompt: str) -> str:
    """Classify a task prompt into a probe task type.

    Uses the same keyword lists as colony.get_best_bug_for_task,
    so a real run and a probe job land in the same bucket. A
    prompt that matches nothing is "general", not a guess at
    one of the measured types.
    """
    text = (prompt or "").lower()
    if any(kw in text for kw in ("write", "create file", "save to", "output to", "append to")):
        return "write"
    if any(kw in text for kw in ("http", "api", "fetch", "request", "post to", "get from")):
        return "write"
    if any(kw in text for kw in ("analyze code", "review code", "code quality")):
        return "analyze"
    if any(kw in text for kw in ("kmkb", "knowledge", "search knowledge", "ask knowledge", "search")):
        return "search"
    if any(kw in text for kw in ("monitor", "health check", "system status", "check service")):
        return "check"
    if any(kw in text for kw in ("list", "find", "explore", "scan", "read file", "check if")):
        return "list"
    return "general"


def record_profile_run(
    role: str,
    model: str,
    task_type: str,
    elapsed_seconds: float,
    success: bool,
    failure_kind: Optional[str] = None,
) -> bool:
    """Record one measured run on a role + model's card.

    Loads the card, adds the run, recomputes its derived
    fields, and writes it back to profiles/. Never raises on
    bad input or a failed write: returns False instead, so a
    swarm is never broken by its own profiling.
    """
    try:
        if not role or not model:
            return False
        elapsed = float(elapsed_seconds or 0.0)
    except (TypeError, ValueError):
        return False
    if task_type not in TASK_TYPES:
        task_type = "general"
    with _LOCK:
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
        entry = card["per_task_type"][task_type]
        entry["samples"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["total_seconds"] += max(0.0, elapsed)
        _refresh_card(card)
        return _save_card(card)
