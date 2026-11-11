"""In-memory per-tool metrics for agent-farm (Copilot roadmap #3).

This is deliberately small and additive: it only counts what the
server already knows, it never calls a model, and it never logs
content, prompts, file contents, or generated code -- only counts,
durations, and validation outcomes.

What is tracked, per tool name:
- calls, successes, failures
- total_seconds and avg_seconds (wall time of the tool call)
- items: units the tool produced (sections, functions, perspectives)
- failed_items: items that came back as a failure marker
- validations: how many runs ended validated / written_not_validated
  / failed (only tools that report a validation status fill this in)

What is tracked, per bug id (profiler factory Phase 1):
- calls, successes, failures, total_seconds and avg_seconds for
  each bug in a parallel swarm run, with the bug's role and model,
  so a later phase can group timings by bug. Only elapsed time
  and success/failure are recorded: no token cost, no quality
  score, and no content of any kind.

What is NOT tracked, on purpose:
- Token cost. Ollama responses here do not report token counts in a
  uniform way, so a token-cost number would be invented. If a future
  Ollama client returns prompt/eval token counts, add them here as
  real counts, not estimates.
- Per-bug attribution for chunked tools. Chunked tools run several
  worker bugs in a thread pool and return aggregated results, so the
  honest unit there is still the tool run, not a single bug.
  Per-bug timing exists only where a swarm result already carries
  a bug id (see record_bug_run).

Metrics live only in this process and reset on restart, like the
colonies themselves. They are exposed through farm_stats(), not a
new MCP tool, so the MCP tool count does not change.
"""

import functools
import threading
import time
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_DATA: Dict[str, Dict[str, Any]] = {}
_BUG_DATA: Dict[str, Dict[str, Any]] = {}

# Bound on retained per-bug records. A long-running server creates
# new bug ids on every colony spawn and nothing removes colonies, so
# an unbounded store would grow for the process lifetime. Oldest
# entries are dropped first. 1000 entries covers far more colonies
# than a session holds (a colony has at most 8 bugs); a later phase
# can persist or age entries out instead of dropping them.
_MAX_BUG_ENTRIES = 1000


def _entry(tool: str) -> Dict[str, Any]:
    entry = _DATA.setdefault(tool, {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "total_seconds": 0.0,
        "items": 0,
        "failed_items": 0,
        "validations": {"validated": 0, "written_not_validated": 0, "failed": 0},
    })
    return entry


def record_tool_run(
    tool: str,
    elapsed_seconds: float,
    success: bool,
    items: int = 0,
    failed_items: int = 0,
    validation_status: Optional[str] = None,
) -> None:
    """Record one completed tool run. Never raises on bad input."""
    try:
        elapsed = float(elapsed_seconds or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    with _LOCK:
        entry = _entry(tool)
        entry["calls"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["total_seconds"] += max(0.0, elapsed)
        entry["items"] += max(0, int(items or 0))
        entry["failed_items"] += max(0, int(failed_items or 0))
        if validation_status in entry["validations"]:
            entry["validations"][validation_status] += 1


def get_metrics() -> Dict[str, Dict[str, Any]]:
    """Return a copy of all metrics, with avg_seconds computed."""
    with _LOCK:
        result = {}
        for tool, entry in _DATA.items():
            calls = entry["calls"]
            result[tool] = {
                "calls": calls,
                "successes": entry["successes"],
                "failures": entry["failures"],
                "total_seconds": round(entry["total_seconds"], 2),
                "avg_seconds": round(entry["total_seconds"] / calls, 2) if calls else 0.0,
                "items": entry["items"],
                "failed_items": entry["failed_items"],
                "validations": dict(entry["validations"]),
            }
        return result


def _bug_entry(bug_id: str, role: Optional[str], model: Optional[str]) -> Dict[str, Any]:
    entry = _BUG_DATA.setdefault(bug_id, {
        "role": role,
        "model": model,
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "total_seconds": 0.0,
    })
    return entry


def record_bug_run(
    bug_id: str,
    elapsed_seconds: float,
    success: bool,
    role: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Record one completed run for a single bug. Never raises on bad input.

    This is the profiler factory Phase 1 store: it only counts what a
    swarm result already carries (bug id, role, model, elapsed time,
    success/failure). It never calls a model and never logs content.
    A bug id is unique to one colony spawn, so a later phase groups
    these runs by role and model, not by bug id alone.
    Retained records are capped at _MAX_BUG_ENTRIES (oldest dropped
    first) so a long-running server cannot grow this store forever.
    """
    try:
        if not bug_id:
            return
        elapsed = float(elapsed_seconds or 0.0)
    except (TypeError, ValueError):
        return
    with _LOCK:
        entry = _bug_entry(str(bug_id), role, model)
        entry["calls"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["total_seconds"] += max(0.0, elapsed)
        while len(_BUG_DATA) > _MAX_BUG_ENTRIES:
            _BUG_DATA.pop(next(iter(_BUG_DATA)))


def get_bug_metrics() -> Dict[str, Dict[str, Any]]:
    """Return a copy of all per-bug metrics, with avg_seconds computed."""
    with _LOCK:
        result = {}
        for bug_id, entry in _BUG_DATA.items():
            calls = entry["calls"]
            result[bug_id] = {
                "role": entry["role"],
                "model": entry["model"],
                "calls": calls,
                "successes": entry["successes"],
                "failures": entry["failures"],
                "total_seconds": round(entry["total_seconds"], 2),
                "avg_seconds": round(entry["total_seconds"] / calls, 2) if calls else 0.0,
            }
        return result


def reset_bug_metrics() -> None:
    """Clear per-bug metrics only. Used by tests; not an MCP tool."""
    with _LOCK:
        _BUG_DATA.clear()


def reset_metrics() -> None:
    """Clear all metrics (tool and per-bug). Used by tests; not an MCP tool."""
    with _LOCK:
        _DATA.clear()
        _BUG_DATA.clear()


def track_tool_run(tool_name=None):
    """Decorator for async MCP tools: record one run in the metrics.

    It never changes the tool's return value. A run counts as a
    failure only if the tool raises, or returns a dict containing an
    "error" key. "items" is taken from a "results" list when the tool
    returns one, otherwise 0 -- swarm tools report their counts in
    different fields, so a missing items count is honest, not guessed.
    Chunked tools record themselves (they also report validation
    outcomes), so do not apply this decorator to them or they will
    be counted twice.
    """
    def decorator(fn):
        name = tool_name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await fn(*args, **kwargs)
            except Exception:
                record_tool_run(name, time.time() - start, False)
                raise
            success = not (
                isinstance(result, dict) and "error" in result
            )
            items = 0
            if isinstance(result, dict):
                results = result.get("results")
                if isinstance(results, list):
                    items = len(results)
            record_tool_run(name, time.time() - start, success, items=items)
            return result

        return wrapper

    return decorator
