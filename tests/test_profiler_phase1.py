"""Tests for profiler factory Phase 1: per-bug timing and status.

Phase 1 only records, for every bug in a swarm run, how long it
took and whether it finished or failed. It changes nothing about
routing: get_best_bug_for_task still picks the first bug of the
matched role, and no model tier changes.

No model calls are made here: sync_query_ollama is replaced with
a fake, so these tests run without Ollama. The fakes match how the
real helper behaves: the real sync_query_ollama returns an
"Error: ..." string on a failed model call instead of raising, so
one fake does exactly that. A test that only passes when the code
is right is the point here: earlier versions of these tests passed
even with always-zero timings and swapped per-bug outcomes.

The fakes sleep 0.05 seconds, a known delay. Timing assertions
check for at least 0.04 seconds of it, so a recording of 0.0 fails.
"""

import time

import agent_farm.colony as colony_module
from agent_farm.colony import (
    execute_parallel_swarm,
    get_best_bug_for_task,
    spawn_colony_internal,
)
from agent_farm.metrics import (
    _MAX_BUG_ENTRIES,
    get_bug_metrics,
    record_bug_run,
    reset_metrics,
)
from agent_farm.state import MODEL_TIERS

# Known delay in the fakes, and the smallest part of it a timing
# assertion will accept (sleep is a minimum, so this has margin).
_FAKE_DELAY = 0.05
_MIN_ELAPSED = 0.04


def _fake_ok(model, prompt):
    time.sleep(_FAKE_DELAY)
    return "ok"


def _fake_maybe_boom(model, prompt):
    time.sleep(_FAKE_DELAY)
    if "boom" in prompt:
        raise RuntimeError("boom")
    return "ok"


def _fake_error_string(model, prompt):
    # Matches the real sync_query_ollama: a failed model call comes
    # back as an "Error: ..." string, not an exception.
    time.sleep(_FAKE_DELAY)
    return "Error: 500"


def test_swarm_results_carry_elapsed_seconds_and_status(monkeypatch):
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)

    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([
        {"prompt": "list files in the folder"},
        {"prompt": "write a short note"},
    ], colony)

    assert len(results) == 2
    for result in results:
        assert result["status"] == "success"
        assert "bug_id" in result
        assert "elapsed_seconds" in result
        # The fake slept 0.05s: a recording of 0.0 must fail here.
        assert result["elapsed_seconds"] >= _MIN_ELAPSED

    bug_metrics = get_bug_metrics()
    assert len(bug_metrics) == 2
    for bug_id, entry in bug_metrics.items():
        assert entry["calls"] == 1
        assert entry["successes"] == 1
        assert entry["failures"] == 0
        assert entry["avg_seconds"] >= _MIN_ELAPSED
        assert entry["role"] in ("scout", "worker", "memory", "guardian", "learner")
        assert entry["model"] == MODEL_TIERS[entry["role"]]
        # No token cost is ever invented, same rule as tool metrics.
        assert "tokens" not in entry
        assert "token_cost" not in entry
    reset_metrics()


def test_swarm_failure_is_recorded_per_bug(monkeypatch):
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_maybe_boom)

    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([
        {"prompt": "list files in the folder"},
        {"prompt": "write a boom file"},
    ], colony)

    assert len(results) == 2
    by_status = {r["status"] for r in results}
    assert by_status == {"success", "failed"}
    for result in results:
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] >= _MIN_ELAPSED

    bug_metrics = get_bug_metrics()
    assert len(bug_metrics) == 2
    # Each result's outcome must sit under its own bug id, not just
    # in the aggregate sums: swapping the two bugs' outcomes must fail.
    for result in results:
        entry = bug_metrics[result["bug_id"]]
        if result["status"] == "success":
            assert entry["successes"] == 1
            assert entry["failures"] == 0
        else:
            assert entry["successes"] == 0
            assert entry["failures"] == 1
    reset_metrics()


def test_swarm_error_string_is_recorded_as_failure(monkeypatch):
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_error_string)

    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([
        {"prompt": "list files in the folder"},
    ], colony)

    assert len(results) == 1
    result = results[0]
    # The model call failed, so the result is failed even though no
    # exception was raised: the "Error: ..." string is the failure.
    assert result["status"] == "failed"
    assert result["error"].startswith("Error:")
    assert "response" not in result
    assert result["elapsed_seconds"] >= _MIN_ELAPSED

    bug_metrics = get_bug_metrics()
    entry = bug_metrics[result["bug_id"]]
    assert entry["successes"] == 0
    assert entry["failures"] == 1
    reset_metrics()


def test_bug_metrics_store_is_bounded():
    reset_metrics()
    for i in range(_MAX_BUG_ENTRIES + 5):
        record_bug_run(f"bug-{i}", 0.01, True, role="worker", model="qwen3:4b")
    bug_metrics = get_bug_metrics()
    assert len(bug_metrics) == _MAX_BUG_ENTRIES
    # Oldest entries are dropped first.
    assert "bug-0" not in bug_metrics
    assert f"bug-{_MAX_BUG_ENTRIES + 4}" in bug_metrics
    reset_metrics()


def test_routing_is_unchanged_by_phase1():
    colony = spawn_colony_internal("standard")
    workers = [b for b in colony.bugs if b.role == "worker"]
    scouts = [b for b in colony.bugs if b.role == "scout"]

    # Routing still picks the first bug of the matched role.
    assert get_best_bug_for_task({"prompt": "write a file"}, colony).id == workers[0].id
    assert get_best_bug_for_task({"prompt": "list files"}, colony).id == scouts[0].id


def test_empty_swarm_records_nothing():
    reset_metrics()
    colony = spawn_colony_internal("standard")
    assert execute_parallel_swarm([], colony) == []
    assert get_bug_metrics() == {}
    reset_metrics()
