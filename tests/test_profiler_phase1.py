"""Tests for profiler factory Phase 1: per-bug timing and status.

Phase 1 only records, for every bug in a swarm run, how long it
took and whether it finished or failed. It changes nothing about
routing: get_best_bug_for_task still picks the first bug of the
matched role, and no model tier changes.

No model calls are made here: sync_query_ollama is replaced with
a fake, so these tests run without Ollama.
"""

import time

import agent_farm.colony as colony_module
from agent_farm.colony import (
    execute_parallel_swarm,
    get_best_bug_for_task,
    spawn_colony_internal,
)
from agent_farm.metrics import get_bug_metrics, reset_metrics


def _fake_ok(model, prompt):
    time.sleep(0.01)
    return "ok"


def _fake_maybe_boom(model, prompt):
    time.sleep(0.01)
    if "boom" in prompt:
        raise RuntimeError("boom")
    return "ok"


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
        assert result["elapsed_seconds"] >= 0

    bug_metrics = get_bug_metrics()
    assert len(bug_metrics) == 2
    for bug_id, entry in bug_metrics.items():
        assert entry["calls"] == 1
        assert entry["successes"] == 1
        assert entry["failures"] == 0
        assert entry["avg_seconds"] >= 0
        assert entry["role"] in ("scout", "worker", "memory", "guardian", "learner")
        assert entry["model"] == "qwen3:4b"
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
        assert result["elapsed_seconds"] >= 0

    bug_metrics = get_bug_metrics()
    assert len(bug_metrics) == 2
    assert sum(e["successes"] for e in bug_metrics.values()) == 1
    assert sum(e["failures"] for e in bug_metrics.values()) == 1
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
