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


def _fake_ok(model, prompt, **kwargs):
    time.sleep(_FAKE_DELAY)
    return "def add(a, b):\n    return a + b"


def _fake_error(model, prompt, **kwargs):
    # Matches the real sync_query_ollama: a failed model call
    # comes back as an "Error: ..." string, not an exception.
    time.sleep(_FAKE_DELAY)
    return "Error: 500"


def _fake_uncertain(model, prompt, **kwargs):
    # Matches the real grounded helper: no usable context in
    # the prompt, so the model answers UNCERTAIN.
    time.sleep(_FAKE_DELAY)
    return "UNCERTAIN"


def _fake_empty(model, prompt, **kwargs):
    time.sleep(_FAKE_DELAY)
    return "   "


def _fake_boom(model, prompt, **kwargs):
    # Safety net only: the real helper returns an Error string
    # instead of raising, so this path should never fire live.
    time.sleep(_FAKE_DELAY)
    raise RuntimeError("boom")


import threading

_concurrency = {"active": 0, "peak": 0}
_concurrency_lock = threading.Lock()


def _fake_counting(model, prompt, **kwargs):
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
        "incomplete": 0,
        "elapsed_seconds": 0.3,
        "total_call_seconds": 0.3,
        "profile_save_failures": 0,
    }
    assert summarize_probe_results([]) == {
        "probed": 0,
        "successes": 0,
        "failures": 0,
        "incomplete": 0,
        "elapsed_seconds": 0.0,
        "total_call_seconds": 0.0,
        "profile_save_failures": 0,
    }


def test_summarize_uses_wall_time_not_call_sum():
    # Two overlapping calls: each took 0.5s, but they ran at the
    # same time, so the run took 0.5s, not 1.0s.
    results = [
        {"status": "success", "elapsed_seconds": 0.5, "started_monotonic": 10.0, "profile_saved": True},
        {"status": "success", "elapsed_seconds": 0.5, "started_monotonic": 10.0, "profile_saved": True},
    ]
    summary = summarize_probe_results(results)
    assert summary["elapsed_seconds"] == 0.5
    assert summary["total_call_seconds"] == 1.0
    assert summary["profile_save_failures"] == 0


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


def test_malformed_card_is_normalized(monkeypatch, tmp_path):
    # A card with wrong types and missing fields must load as a
    # usable card, not crash the reader.
    from agent_farm.profiles import _card_path

    _profile_dir(monkeypatch, tmp_path)
    path = _card_path("worker", "qwen3:4b")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "role": "worker",
        "model": "qwen3:4b",
        "samples": "five",
        "successes": True,
        "total_seconds": "fast",
        "per_task_type": {"write": {"samples": "three"}},
        "failure_modes": {"error": "many"},
        "trusted": "yes",
        "stale_reason": 42,
    }), encoding="utf-8")
    card = get_profile("worker", "qwen3:4b")
    assert card is not None
    assert card["samples"] == 0
    assert card["successes"] == 0
    assert card["total_seconds"] == 0.0
    assert card["per_task_type"]["write"]["samples"] == 0
    assert card["per_task_type"]["search"]["samples"] == 0
    assert card["failure_modes"] == {}
    assert card["trusted"] is False
    assert card["stale_reason"] is None
    # And a run still records on the repaired card.
    assert record_profile_run("worker", "qwen3:4b", "write", 0.1, True) is True
    assert get_profile("worker", "qwen3:4b")["samples"] == 1


def test_mismatched_card_is_rejected(monkeypatch, tmp_path):
    # A card whose stored model is not the one the file is for
    # belongs to another model: it is replaced, not mixed in.
    from agent_farm.profiles import _card_path

    _profile_dir(monkeypatch, tmp_path)
    path = _card_path("worker", "qwen3:4b")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "role": "worker", "model": "other-model", "samples": 99,
    }), encoding="utf-8")
    assert get_profile("worker", "qwen3:4b")["samples"] == 0


def test_models_with_same_normalized_name_get_separate_cards(monkeypatch, tmp_path):
    # "qwen3:4b" and "qwen3-4b" normalize to the same string, but
    # they are different models and must not share a card.
    _profile_dir(monkeypatch, tmp_path)
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    record_profile_run("worker", "qwen3-4b", "write", 0.2, True)
    card_a = get_profile("worker", "qwen3:4b")
    card_b = get_profile("worker", "qwen3-4b")
    assert card_a["samples"] == 1
    assert card_b["samples"] == 1
    assert card_a["avg_seconds"] == 0.1
    assert card_b["avg_seconds"] == 0.2
    assert len(list_profiles()) == 2


def test_card_records_machine_and_legacy_card_is_adopted(monkeypatch, tmp_path):
    # A card records the machine that measured it, and a card
    # written before machine scoping is adopted by the machine
    # that first loads it, then re-saved under the new name.
    from agent_farm.profiles import _legacy_card_path, machine_slug

    _profile_dir(monkeypatch, tmp_path)
    legacy = _legacy_card_path("worker", "qwen3:4b")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({
        "role": "worker", "model": "qwen3:4b", "samples": 3,
        "successes": 3, "failures": 0, "success_rate": 1.0,
        "total_seconds": 0.3, "avg_seconds": 0.1,
        "per_task_type": {},
        "failure_modes": {}, "best_task_types": [], "worst_task_types": [],
        "trusted": False, "stale": False, "stale_reason": None,
        "last_updated": None,
    }), encoding="utf-8")
    card = get_profile("worker", "qwen3:4b")
    assert card["samples"] == 3
    assert card["machine"] == machine_slug()
    # Recording a run re-saves under the new name and removes
    # the legacy file, so the card is not listed twice.
    assert record_profile_run("worker", "qwen3:4b", "write", 0.1, True) is True
    assert not legacy.exists()
    assert len(list_profiles()) == 1
    assert list_profiles()[0]["samples"] == 4


def test_retired_card_stays_untrusted_after_more_runs(monkeypatch, tmp_path):
    # A retired card must not become trusted again just because
    # more runs land on it.
    _profile_dir(monkeypatch, tmp_path)
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("worker", "qwen3:4b")["trusted"] is True
    assert retire_profile("worker", "qwen3:4b", "model changed") is True
    for _ in range(MIN_SAMPLES_PER_TYPE):
        record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    card = get_profile("worker", "qwen3:4b")
    assert card["stale"] is True
    assert card["trusted"] is False


def test_probe_clock_excludes_semaphore_wait(monkeypatch, tmp_path):
    # Hold every model slot, then release them after 0.5s. A probe
    # waits for a slot with its clock stopped, so the recorded
    # time is just the fake call time, not the 0.5s wait.
    from agent_farm.reliability import MODEL_SEMAPHORE

    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    for _ in range(4):
        MODEL_SEMAPHORE.acquire()

    def release_all():
        for _ in range(4):
            try:
                MODEL_SEMAPHORE.release()
            except ValueError:
                pass

    timer = threading.Timer(0.5, release_all)
    timer.start()
    try:
        results = run_probes(colony)
    finally:
        timer.cancel()
        release_all()
    assert len(results) == 15
    for result in results:
        assert result["elapsed_seconds"] < 0.3


def test_probe_run_returns_within_budget(monkeypatch, tmp_path):
    # A model that answers slowly must not hold a spawn hostage:
    # the run returns within the wall budget and reports the
    # unfinished jobs as incomplete.
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "PROBE_WALL_BUDGET_SECONDS", 0.5)

    def _fake_slow(model, prompt, **kwargs):
        time.sleep(5.0)
        return "ok"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_slow)
    colony = spawn_colony_internal("standard")
    started = time.monotonic()
    results = run_probes(colony)
    wall = time.monotonic() - started
    assert wall < 3.0
    summary = summarize_probe_results(results)
    assert summary["incomplete"] > 0
    assert summary["probed"] + summary["incomplete"] == 15


def test_failed_profile_save_is_reported(monkeypatch, tmp_path):
    # A profiles dir that cannot be written: the swarm and probe
    # results say profile_saved False, and the summary counts it,
    # instead of the failure going unnoticed.
    blocker = tmp_path / "a-file"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(blocker))
    assert record_profile_run("worker", "qwen3:4b", "write", 0.1, True) is False
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    results = run_probes(colony)
    assert results
    assert all(r["profile_saved"] is False for r in results)
    assert summarize_probe_results(results)["profile_save_failures"] == len(results)
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    swarm_results = execute_parallel_swarm([{"prompt": "write a file"}], colony)
    assert swarm_results[0]["profile_saved"] is False


def test_tool_swarm_run_feeds_card(monkeypatch, tmp_path):
    # Real tool-swarm work lands on the bug's profile card, with
    # the elapsed time of the whole agent run.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    _profile_dir(monkeypatch, tmp_path)

    def _fake_agent(role, model, task, context="", deep=False):
        return {
            "answer": "done", "tool_calls": [],
            "iterations": 1, "status": "completed", "mode": "test",
        }

    monkeypatch.setattr(agent_module, "run_tool_agent", _fake_agent)
    colony = spawn_colony_internal("hybrid")
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["profile_saved"] is True
    assert results[0]["elapsed_seconds"] >= 0.0
    card = get_profile("scout", "qwen3:4b")
    assert card is not None
    assert card["samples"] == 1
    assert card["successes"] == 1
    assert card["per_task_type"]["list"]["samples"] == 1


def test_router_prefers_trusted_card(monkeypatch, tmp_path):
    # A trusted card for the task type picks the bug that earned
    # it, even when the keyword default would pick another bug
    # of the same role.
    from agent_farm.colony import get_best_bug_for_task

    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    workers = [b for b in colony.bugs if b.role == "worker"]
    assert len(workers) == 2
    # No cards yet: the keyword default picks the first worker.
    assert get_best_bug_for_task({"prompt": "write a file"}, colony).id == workers[0].id
    # Give the SECOND worker a trusted card for every task type.
    workers[1].model = "qwen2.5:14b"
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen2.5:14b", task_type, 0.1, True)
    assert get_profile("worker", "qwen2.5:14b")["trusted"] is True
    picked = get_best_bug_for_task({"prompt": "write a file"}, colony)
    assert picked.id == workers[1].id


def test_router_search_goes_to_memory(monkeypatch, tmp_path):
    # The shared taxonomy: a "search" prompt routes to memory,
    # the same bucket the profiler records it in.
    from agent_farm.colony import get_best_bug_for_task

    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("hybrid")
    picked = get_best_bug_for_task({"prompt": "search the knowledge base"}, colony)
    assert picked.role == "memory"


def test_cancelled_spawn_removes_colony(monkeypatch, tmp_path):
    # Cancelling a spawn while probes are running must not leave
    # an orphan colony nobody can reach by id.
    from agent_farm.state import colonies

    _profile_dir(monkeypatch, tmp_path)

    def _fake_slow(model, prompt, **kwargs):
        time.sleep(1.5)
        return "ok"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_slow)
    before = set(colonies)

    async def main():
        task = asyncio.create_task(server.spawn_colony("fast"))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Let the probe threads finish so the test does not end
        # while they are still writing to the profiles dir.
        await asyncio.sleep(2.0)

    asyncio.run(main())
    assert set(colonies) == before


def test_concurrent_processes_do_not_lose_runs(monkeypatch, tmp_path):
    # Two processes recording the same card at once must not lose
    # each other's runs: the file lock serializes them.
    import os
    import subprocess
    import sys
    from pathlib import Path

    _profile_dir(monkeypatch, tmp_path)
    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from agent_farm.profiles import record_profile_run;"
        "[record_profile_run('worker', 'qwen3:4b', 'write', 0.1, True) for _ in range(3)]"
    ) % str(root)
    env = dict(os.environ, AGENT_FARM_PROFILES_DIR=str(tmp_path))
    procs = [
        subprocess.Popen([sys.executable, "-c", code], env=env, cwd=str(root))
        for _ in range(2)
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    card = get_profile("worker", "qwen3:4b")
    assert card["samples"] == 6


def test_stale_lock_is_removed(monkeypatch, tmp_path):
    """A lock left by a dead process is removed, not waited out.

    The stale check compares the lock file's mtime (wall clock)
    against time.time(). Using the monotonic clock there made the
    check never fire: monotonic is seconds since boot, mtime is
    seconds since 1970, so their difference is always a large
    negative number and no lock ever looked stale.
    """
    import os

    from agent_farm.profiles import _card_lock, _card_path

    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(tmp_path))
    path = _card_path("worker", "qwen3:4b")
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("", encoding="utf-8")
    old = time.time() - 40.0
    os.utime(lock_path, (old, old))
    started = time.monotonic()
    with _card_lock(path):
        elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert not lock_path.exists()
    assert record_profile_run("worker", "qwen3:4b", "write", 1.0, True) is True


def test_pick_is_profiled(monkeypatch, tmp_path):
    # The marking helper agrees with profile_pick: a bug is
    # profiled only when its card is trusted, not stale, and
    # for this machine with a trusted entry for the task type.
    from agent_farm.profiles import pick_is_profiled

    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    worker = next(b for b in colony.bugs if b.role == "worker")
    # No card at all: not profiled.
    assert pick_is_profiled("write", worker) is False
    # A few runs but not enough to trust: still not profiled.
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    assert pick_is_profiled("write", worker) is False
    # A trusted card: profiled for every task type.
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("worker", "qwen3:4b")["trusted"] is True
    assert pick_is_profiled("write", worker) is True
    assert pick_is_profiled("list", worker) is True
    # A retired card is not profiled even with the samples.
    retire_profile("worker", "qwen3:4b")
    assert pick_is_profiled("write", worker) is False


def test_swarm_result_marks_profiled(monkeypatch, tmp_path):
    # A swarm result says whether its pick was profile-driven.
    # No trusted card yet: the pick is a fallback, marked False.
    # With a trusted card: the same task is marked True.
    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["profiled"] is False
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("worker", "qwen3:4b")["trusted"] is True
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["profiled"] is True
    reset_metrics()


def test_tool_swarm_result_marks_profiled(monkeypatch, tmp_path):
    # A tool-swarm result carries the same marking.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    _profile_dir(monkeypatch, tmp_path)

    def _fake_agent(role, model, task, context="", deep=False):
        return {
            "answer": "done", "tool_calls": [],
            "iterations": 1, "status": "completed", "mode": "test",
        }

    monkeypatch.setattr(agent_module, "run_tool_agent", _fake_agent)
    colony = spawn_colony_internal("hybrid")
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["profiled"] is False
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("scout", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("scout", "qwen3:4b")["trusted"] is True
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["profiled"] is True


def test_swarm_failure_result_marks_profiled(monkeypatch, tmp_path):
    # A failed swarm result carries the marking too: the flag
    # is about the routing decision, not the outcome.
    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_error)
    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["profiled"] is False
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("worker", "qwen3:4b")["trusted"] is True
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["profiled"] is True
    reset_metrics()


def test_tool_swarm_failure_result_marks_profiled(monkeypatch, tmp_path):
    # A tool-swarm result from a raised exception carries it too.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    _profile_dir(monkeypatch, tmp_path)

    def _fake_raise(role, model, task, context="", deep=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_module, "run_tool_agent", _fake_raise)
    colony = spawn_colony_internal("hybrid")
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["profiled"] is False
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("scout", "qwen3:4b", task_type, 0.1, True)
    assert get_profile("scout", "qwen3:4b")["trusted"] is True
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["profiled"] is True
