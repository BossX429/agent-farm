"""Tests for profiler factory Phase 2: profile cards + probe harness.

Phase 2 builds a measured profile card per role + model, stored
as JSON in profiles/, from probe runs and real swarm runs. A card
is trusted only at 5 samples per task type.

No model calls are made here: sync_query_ollama is replaced
with fakes that behave like the real helper -- a failed call
comes back as an "Error: ..." string, not an exception, and a
grounded call with no usable context answers "UNCERTAIN". The
one fake that raises exists only to cover the safety net for a
call that raises, which the real helper does not do.
"""

import asyncio
import json
import time

import agent_farm.colony as colony_module
import agent_farm.probes as probes_module
import agent_farm.server as server
from agent_farm.colony import execute_parallel_swarm, spawn_colony_internal
from agent_farm.metrics import get_bug_metrics, reset_metrics
from agent_farm.probes import PROBE_JOBS, run_probes, summarize_probe_results
from agent_farm.profiles import (
    MIN_SAMPLES_PER_TYPE,
    PROBE_TASK_TYPES,
    classify_result,
    create_profile,
    get_profile,
    list_profiles,
    machine_slug,
    profiles_summary,
    record_profile_run,
    retire_profile,
    task_type_for_prompt,
)

# Known delay in the fakes, and the smallest part of it a timing
# assertion will accept (sleep is a minimum, so this has margin).
_FAKE_DELAY = 0.05
_MIN_ELAPSED = 0.04


def _fake_ok(model, prompt):
    time.sleep(_FAKE_DELAY)
    return "def add(a, b):\n    return a + b"


def _fake_error(model, prompt):
    # Matches the real sync_query_ollama: a failed model call
    # comes back as an "Error: ..." string, not an exception.
    time.sleep(_FAKE_DELAY)
    return "Error: 500"


def _fake_uncertain(model, prompt):
    # Matches the real grounded helper: no usable context in
    # the prompt, so the model answers UNCERTAIN.
    time.sleep(_FAKE_DELAY)
    return "UNCERTAIN"


def _fake_empty(model, prompt):
    time.sleep(_FAKE_DELAY)
    return "   "


def _fake_boom(model, prompt):
    # Safety net only: the real helper returns an Error string
    # instead of raising, so this path should never fire live.
    time.sleep(_FAKE_DELAY)
    raise RuntimeError("boom")


import threading

_concurrency = {"active": 0, "peak": 0}
_concurrency_lock = threading.Lock()


def _fake_counting(model, prompt):
    # Tracks how many fake calls run at once, to prove the
    # probe pool never exceeds the model semaphore size.
    with _concurrency_lock:
        _concurrency["active"] += 1
        _concurrency["peak"] = max(
            _concurrency["peak"], _concurrency["active"]
        )
    try:
        time.sleep(_FAKE_DELAY)
        return "ok"
    finally:
        with _concurrency_lock:
            _concurrency["active"] -= 1


def _profile_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(tmp_path))


def test_create_profile_schema():
    card = create_profile("worker", "qwen3:4b")
    assert card["role"] == "worker"
    assert card["model"] == "qwen3:4b"
    assert card["samples"] == 0
    assert card["trusted"] is False
    assert card["stale"] is False
    assert set(card["per_task_type"]) == set(PROBE_TASK_TYPES + ["general"])


def test_record_runs_accumulate_and_trust(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    for _ in range(MIN_SAMPLES_PER_TYPE):
        assert record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    card = get_profile("worker", "qwen3:4b")
    assert card["samples"] == 5
    assert card["per_task_type"]["write"]["trusted"] is True
    # Only one task type has enough samples: the card is not trusted.
    assert card["trusted"] is False
    for task_type in PROBE_TASK_TYPES:
        if task_type == "write":
            continue
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen3:4b", task_type, 0.2, True)
    card = get_profile("worker", "qwen3:4b")
    assert card["samples"] == 25
    assert card["trusted"] is True
    assert card["success_rate"] == 1.0
    assert card["avg_seconds"] == 0.18


def test_card_survives_reload_from_disk(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    record_profile_run("scout", "qwen3:4b", "list", 0.3, True)
    record_profile_run("scout", "qwen3:4b", "list", 0.1, False, "uncertain")
    card = get_profile("scout", "qwen3:4b")
    assert card["samples"] == 2
    assert card["successes"] == 1
    assert card["failures"] == 1
    assert card["per_task_type"]["list"]["samples"] == 2
    paths = sorted(tmp_path.glob("scout-qwen3-4b-*.json"))
    assert len(paths) == 1
    path = paths[0]
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["role"] == "scout"
    assert stored["failure_modes"] == {"uncertain": 1}


def test_failure_modes_counted(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    record_profile_run("worker", "qwen3:4b", "write", 0.1, False, "error")
    record_profile_run("worker", "qwen3:4b", "write", 0.1, False, "error")
    record_profile_run("worker", "qwen3:4b", "write", 0.1, False, "uncertain")
    record_profile_run("worker", "qwen3:4b", "write", 0.1, False, "timeout")
    card = get_profile("worker", "qwen3:4b")
    assert card["failure_modes"] == {"error": 2, "uncertain": 1, "timeout": 1}
    assert card["successes"] == 0


def test_best_worst_ranking(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    for _ in range(3):
        record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    for _ in range(3):
        record_profile_run("worker", "qwen3:4b", "search", 0.5, False, "uncertain")
    card = get_profile("worker", "qwen3:4b")
    assert card["best_task_types"][0] == "write"
    assert card["worst_task_types"][0] == "search"


def test_retire_profile(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    assert retire_profile("worker", "qwen3:4b") is False
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    assert retire_profile("worker", "qwen3:4b", "model changed") is True
    card = get_profile("worker", "qwen3:4b")
    assert card["stale"] is True
    assert card["stale_reason"] == "model changed"
    assert card["trusted"] is False
    assert card["samples"] == 1


def test_classify_result():
    assert classify_result("some answer") == (True, None)
    assert classify_result("Error: 500") == (False, "error")
    assert classify_result("UNCERTAIN") == (False, "uncertain")
    assert classify_result("UNCERTAIN, no context given") == (False, "uncertain")
    assert classify_result("") == (False, "empty")
    assert classify_result("   ") == (False, "empty")
    assert classify_result(None) == (False, "empty")
    assert classify_result(None, "Error: timed out") == (False, "timeout")
    assert classify_result("Error: request timed out") == (False, "timeout")
    assert classify_result("Error: timed out after 120s") == (False, "timeout")
    assert classify_result(None, "boom") == (False, "error")
    assert classify_result("ok", "Error: 500") == (False, "error")


def test_task_type_for_prompt():
    assert task_type_for_prompt("write a short note") == "write"
    assert task_type_for_prompt("analyze code in main.py") == "analyze"
    assert task_type_for_prompt("search the knowledge base") == "search"
    assert task_type_for_prompt("check service status") == "check"
    assert task_type_for_prompt("list files in the folder") == "list"
    assert task_type_for_prompt("do something vague") == "general"


def test_run_probes_with_ok_fake(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)

    # Standard colony: scout, worker, worker, memory. Cards are
    # per role + model, so one representative per combo is probed:
    # 3 combos x 5 jobs.
    assert len(results) == 15
    assert len(PROBE_JOBS) == 5
    for result in results:
        assert result["status"] == "success"
        assert result["failure_kind"] is None
        assert result["elapsed_seconds"] >= _MIN_ELAPSED
    for role in ("scout", "worker", "memory"):
        card = get_profile(role, "qwen3:4b")
        assert card is not None, role
        assert card["samples"] == 5
        assert card["successes"] == 5
        # One sample per task type: not trusted yet.
        assert card["trusted"] is False
        for task_type in PROBE_TASK_TYPES:
            assert card["per_task_type"][task_type]["samples"] == 1


def test_run_probes_error_string_is_failure(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_error)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert len(results) == 15
    for result in results:
        assert result["status"] == "failed"
        assert result["failure_kind"] == "error"
    card = get_profile("worker", "qwen3:4b")
    assert card["failures"] == 5
    assert card["failure_modes"] == {"error": 5}


def test_run_probes_never_exceed_model_concurrency(monkeypatch, tmp_path):
    from agent_farm.reliability import MAX_MODEL_CONCURRENCY

    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_counting)
    _concurrency["active"] = 0
    _concurrency["peak"] = 0
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert len(results) == 15
    assert all(r["status"] == "success" for r in results)
    # The pool is capped at the model semaphore size, so no
    # probe waits for a slot with its clock running. The old
    # pool of 8 peaked at 8 here; the cap must hold at 4.
    assert _concurrency["peak"] <= MAX_MODEL_CONCURRENCY
    # ...and the cap must not have serialized the run: probes
    # still overlap.
    assert _concurrency["peak"] >= 2


def test_run_probes_uncertain_is_failure(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_uncertain)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert all(r["failure_kind"] == "uncertain" for r in results)
    card = get_profile("scout", "qwen3:4b")
    assert card["failure_modes"] == {"uncertain": 5}


def test_run_probes_empty_and_raising_fakes(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_empty)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert all(r["failure_kind"] == "empty" for r in results)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_boom)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert all(r["status"] == "failed" for r in results)
    assert all(r["failure_kind"] == "error" for r in results)


def test_run_probes_empty_colony(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    colony.bugs = []
    assert run_probes(colony) == []


def test_spawn_probes_by_default(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)

    async def main():
        result = await server.spawn_colony("fast")
        assert result["bugs"]
        # Fast colony: scout, scout, worker, worker. Two unique
        # combos x 5 jobs.
        assert result["probes"]["probed"] == 10
        assert result["probes"]["successes"] == 10
        assert result["probes"]["failures"] == 0
        await server.dissolve_colony(result["colony_id"])
        return result

    asyncio.run(main())
    assert get_profile("scout", "qwen3:4b")["samples"] == 5
    assert get_profile("worker", "qwen3:4b")["samples"] == 5


def test_spawn_returns_colony_when_probes_fail(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_error)

    async def main():
        result = await server.spawn_colony("fast")
        assert result["bugs"]
        assert result["probes"]["probed"] == 10
        assert result["probes"]["failures"] == 10
        await server.dissolve_colony(result["colony_id"])

    asyncio.run(main())
    card = get_profile("worker", "qwen3:4b")
    assert card["failures"] == 5


def test_spawn_probes_opt_out(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)

    async def main():
        result = await server.spawn_colony("fast", run_probes=False)
        assert result["bugs"]
        assert result["probes"] is None
        await server.dissolve_colony(result["colony_id"])

    asyncio.run(main())
    assert list_profiles() == []


def test_real_swarm_run_feeds_card(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([
        {"prompt": "write a short note"},
        {"prompt": "list files in the folder"},
    ], colony)
    assert len(results) == 2
    worker_card = get_profile("worker", "qwen3:4b")
    assert worker_card["per_task_type"]["write"]["samples"] == 1
    assert worker_card["per_task_type"]["write"]["successes"] == 1
    scout_card = get_profile("scout", "qwen3:4b")
    assert scout_card["per_task_type"]["list"]["samples"] == 1

    # A grounded call that answers UNCERTAIN counts as a
    # completed call in Phase 1 metrics but as a failure on
    # the card: the call worked, the answer did not.
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_uncertain)
    execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    worker_card = get_profile("worker", "qwen3:4b")
    assert worker_card["per_task_type"]["write"]["samples"] == 2
    assert worker_card["per_task_type"]["write"]["failures"] == 1
    assert worker_card["failure_modes"] == {"uncertain": 1}
    bug_metrics = get_bug_metrics()
    worker_bug = next(
        entry for entry in bug_metrics.values() if entry["role"] == "worker"
    )
    assert worker_bug["successes"] == 2
    assert worker_bug["failures"] == 0
    reset_metrics()


def test_farm_stats_includes_profiles(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)

    async def main():
        stats = await server.farm_stats()
        return stats

    stats = asyncio.run(main())
    assert stats["profiles"] == [{
        "role": "worker",
        "model": "qwen3:4b",
        "machine": machine_slug(),
        "samples": 1,
        "trusted": False,
        "success_rate": 1.0,
        "avg_seconds": 0.1,
        "stale": False,
    }]


def test_summarize_probe_results():
    results = [
        {"status": "success", "elapsed_seconds": 0.1},
        {"status": "failed", "elapsed_seconds": 0.2},
    ]
    assert summarize_probe_results(results) == {
        "probed": 2,
        "successes": 1,
        "failures": 1,
        "elapsed_seconds": 0.3,
    }
    assert summarize_probe_results([]) == {
        "probed": 0,
        "successes": 0,
        "failures": 0,
        "elapsed_seconds": 0.0,
    }


def test_record_never_raises_on_bad_input(monkeypatch, tmp_path):
    _profile_dir(monkeypatch, tmp_path)
    assert record_profile_run("", "", "write", 0.1, True) is False
    assert record_profile_run("worker", "qwen3:4b", "bogus-type", 0.1, True) is True
    card = get_profile("worker", "qwen3:4b")
    assert card["per_task_type"]["general"]["samples"] == 1
    # A profiles dir that cannot be created: record fails
    # quietly instead of raising.
    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(tmp_path / "a-file"))
    (tmp_path / "a-file").write_text("not a dir", encoding="utf-8")
    assert record_profile_run("worker", "qwen3:4b", "write", 0.1, True) is False
