"""Tests for colony-type measurement (Phase 3).

A colony-type card measures which colony type is best for a
task type: standard, fast, heavy, or hybrid. Same contract as
the bug profiler: a card is trusted only at 5 samples per
probe task type, and colony_type_pick() returns the best
trusted card's type or None.

No model calls are made here: sync_query_ollama is replaced
with fakes that behave like the real helper -- a failed call
comes back as an "Error: ..." string, not an exception.
"""

import asyncio
import json
import time

import agent_farm.colony as colony_module
from agent_farm.colony import execute_parallel_swarm, spawn_colony_internal
from agent_farm.colony_profiles import (
    COLONY_TYPES,
    colony_type_pick,
    create_colony_profile,
    dominant_task_type,
    get_colony_profile,
    list_colony_profiles,
    prune_stale_colony_profiles,
    record_colony_run,
    resolve_colony_type,
)
from agent_farm.profiles import MIN_SAMPLES_PER_TYPE, PROBE_TASK_TYPES

_FAKE_DELAY = 0.05


def _fake_ok(model, prompt, **kwargs):
    time.sleep(_FAKE_DELAY)
    return "def add(a, b):\n    return a + b"


def _trust_card(colony_type):
    """Record enough successful runs to trust a card."""
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            assert record_colony_run(colony_type, task_type, 1.0, True)


def test_fresh_card_is_untrusted():
    card = create_colony_profile("fast")
    assert card["colony_type"] == "fast"
    assert card["samples"] == 0
    assert card["trusted"] is False
    assert card["stale"] is False
    assert all(
        entry["samples"] == 0 for entry in card["per_task_type"].values()
    )


def test_colony_type_is_stored_on_spawn():
    colony = spawn_colony_internal("heavy")
    assert colony.colony_type == "heavy"
    assert [bug.role for bug in colony.bugs] == [
        "memory",
        "memory",
        "learner",
        "guardian",
    ]
    colony = spawn_colony_internal("standard")
    assert colony.colony_type == "standard"


def test_record_runs_accumulate_and_trust():
    for _ in range(MIN_SAMPLES_PER_TYPE):
        assert record_colony_run("fast", "write", 2.0, True)
    card = get_colony_profile("fast")
    assert card["samples"] == MIN_SAMPLES_PER_TYPE
    assert card["per_task_type"]["write"]["samples"] == MIN_SAMPLES_PER_TYPE
    assert card["per_task_type"]["write"]["success_rate"] == 1.0
    assert card["per_task_type"]["write"]["avg_seconds"] == 2.0
    # Only one task type is trusted: the card is not trusted yet.
    assert card["trusted"] is False
    for task_type in ("analyze", "search", "check", "list"):
        for _ in range(MIN_SAMPLES_PER_TYPE):
            assert record_colony_run("fast", task_type, 1.0, True)
    card = get_colony_profile("fast")
    assert card["trusted"] is True
    assert card["success_rate"] == 1.0


def test_card_survives_reload_from_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    assert record_colony_run("hybrid", "write", 3.0, True)
    card = get_colony_profile("hybrid")
    assert card["samples"] == 1
    assert card["per_task_type"]["write"]["total_seconds"] == 3.0


def test_failure_kind_counted():
    assert record_colony_run("heavy", "search", 1.0, False, "uncertain")
    card = get_colony_profile("heavy")
    assert card["failures"] == 1
    assert card["failure_modes"] == {"uncertain": 1}
    assert card["success_rate"] == 0.0


def test_pick_prefers_higher_success_rate():
    _trust_card("fast")
    _trust_card("heavy")
    # Make heavy the better card for "write": all successes at
    # 1.0s, while fast has one failure in its write entry.
    assert record_colony_run("fast", "write", 1.0, False, "error")
    assert colony_type_pick("write") == "heavy"


def test_pick_prefers_faster_on_tie():
    _trust_card("fast")
    _trust_card("standard")
    # Same success rate for "write"; standard is faster.
    assert record_colony_run("fast", "write", 5.0, True)
    assert record_colony_run("standard", "write", 1.0, True)
    assert colony_type_pick("write") == "standard"


def test_pick_returns_none_without_trusted_card():
    assert record_colony_run("fast", "write", 1.0, True)
    assert colony_type_pick("write") is None
    assert colony_type_pick("analyze") is None


def test_pick_ignores_untrusted_task_type():
    _trust_card("fast")
    # The card is trusted, but its "write" entry is trusted and
    # its "analyze" entry is trusted too; record a fresh type
    # with no samples by asking about a type that has no entry
    # trusted -- "general" never counts toward trust.
    assert colony_type_pick("general") is None


def test_dominant_task_type():
    tasks = [
        {"prompt": "write a function"},
        {"prompt": "write another function"},
        {"prompt": "list files in the folder"},
    ]
    assert dominant_task_type(tasks) == "write"
    assert dominant_task_type([]) == "general"
    assert dominant_task_type([{"prompt": "hello"}]) == "general"


def test_resolve_colony_type_auto_picks_trusted():
    _trust_card("heavy")
    colony_type, profiled = resolve_colony_type(
        "auto", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "heavy"
    assert profiled is True


def test_resolve_colony_type_auto_falls_back_without_trusted_card():
    colony_type, profiled = resolve_colony_type(
        "auto", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "fast"
    assert profiled is False


def test_resolve_colony_type_explicit_is_used_as_is():
    _trust_card("heavy")
    colony_type, profiled = resolve_colony_type(
        "standard", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "standard"
    assert profiled is False


def test_prune_removes_orphaned_colony_type(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    assert record_colony_run("standard", "write", 1.0, True)
    # A card for a type that is no longer built: write it by hand.
    from agent_farm.colony_profiles import _colony_card_path, _save_colony_card

    orphan = create_colony_profile("mega")
    orphan["samples"] = 5
    assert _save_colony_card(orphan)
    removed = prune_stale_colony_profiles()
    assert removed == [_colony_card_path("mega").name]
    assert get_colony_profile("mega") is None
    assert get_colony_profile("standard") is not None


def test_prune_keeps_current_cards(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    for colony_type in COLONY_TYPES:
        assert record_colony_run(colony_type, "write", 1.0, True)
    assert prune_stale_colony_profiles() == []
    assert len(list_colony_profiles()) == len(COLONY_TYPES)


def test_record_never_raises_on_bad_input():
    assert record_colony_run("", "write", 1.0, True) is False
    assert record_colony_run("fast", "write", "not a number", True) is False
    # None elapsed is recorded as 0.0, same as the bug profiler:
    # a missing duration is not a reason to drop the run.
    assert record_colony_run("fast", "write", None, True) is True
    assert record_colony_run("fast", "not-a-type", 1.0, True) is True


def test_probes_feed_colony_card(monkeypatch):
    # Probes are the primary way a new colony type earns trust
    # quickly (5 tasks per spawn), so they must feed the colony
    # card, not just the bug cards. The live proof on the PC
    # caught this: after two heavy spawns with probes, the heavy
    # card did not exist, because run_probes bypasses
    # execute_parallel_swarm.
    import agent_farm.probes as probes_module
    from agent_farm.probes import PROBE_JOBS, run_probes

    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("heavy")
    results = run_probes(colony)
    assert len(results) == len(PROBE_JOBS) * 3  # 3 role+model combos
    for result in results:
        assert result["colony_profile_saved"] is True
    card = get_colony_profile("heavy")
    assert card is not None
    # 3 representative bugs x 5 probe jobs = 15 samples, 3 per type.
    assert card["samples"] == 15
    for task_type in PROBE_TASK_TYPES:
        assert card["per_task_type"][task_type]["samples"] == 3
    assert card["trusted"] is False


def test_probes_trust_colony_card_after_five_spawns(monkeypatch):
    # Five spawns with probes build a trusted colony card: 5
    # samples per probe task type.
    import agent_farm.probes as probes_module
    from agent_farm.probes import run_probes

    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    for _ in range(5):
        colony = spawn_colony_internal("heavy")
        run_probes(colony)
    card = get_colony_profile("heavy")
    assert card is not None
    assert card["samples"] == 75
    for task_type in PROBE_TASK_TYPES:
        assert card["per_task_type"][task_type]["samples"] == 15
    assert card["trusted"] is True


def test_swarm_run_feeds_colony_card(monkeypatch):
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("fast")
    results = execute_parallel_swarm(
        [{"prompt": "write a function that adds two numbers"}], colony
    )
    assert results and results[0]["status"] == "success"
    assert results[0]["colony_profile_saved"] is True
    card = get_colony_profile("fast")
    assert card is not None
    assert card["samples"] == 1
    assert card["per_task_type"]["write"]["samples"] == 1


def test_spawn_prunes_stale_colony_cards(monkeypatch, tmp_path):
    # The cleanup runs on a real spawn through the MCP tool
    # layer, so a stale colony card left by a removed colony
    # type is gone after the next spawn, without anyone asking.
    import agent_farm.server as server
    from agent_farm.colony_profiles import _colony_card_path, _save_colony_card

    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    orphan = create_colony_profile("mega")
    orphan["samples"] = 5
    assert _save_colony_card(orphan)
    orphan_path = _colony_card_path("mega")
    assert orphan_path.exists()

    async def main():
        return await server.spawn_colony("standard", run_probes=False)

    asyncio.run(main())
    assert not orphan_path.exists()


def test_tool_swarm_run_feeds_colony_card(monkeypatch):
    # Tool-swarm work lands on the colony type's card too, when
    # the caller passes the colony type. Without it, the colony
    # card is never fed: the re-review caught that tool_swarm
    # runs through agent.run_tool_swarm, not execute_parallel_swarm.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    def _fake_agent(role, model, task, context="", deep=False):
        return {
            "answer": "done", "tool_calls": [],
            "iterations": 1, "status": "completed", "mode": "test",
        }

    monkeypatch.setattr(agent_module, "run_tool_agent", _fake_agent)
    colony = spawn_colony_internal("hybrid")
    results = run_tool_swarm(
        [{"prompt": "list files in the folder"}],
        colony.bugs,
        colony_type=colony.colony_type,
    )
    assert len(results) == 1
    assert results[0]["colony_profile_saved"] is True
    card = get_colony_profile("hybrid")
    assert card is not None
    assert card["samples"] == 1
    assert card["per_task_type"]["list"]["samples"] == 1


def test_tool_swarm_without_colony_type_skips_colony_card(monkeypatch):
    # Callers with no colony type (health/recon/deep swarms build
    # ad-hoc bugs, not a colony type) must not invent one: no
    # colony card is recorded and the flag says so.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    def _fake_agent(role, model, task, context="", deep=False):
        return {
            "answer": "done", "tool_calls": [],
            "iterations": 1, "status": "completed", "mode": "test",
        }

    monkeypatch.setattr(agent_module, "run_tool_agent", _fake_agent)
    colony = spawn_colony_internal("hybrid")
    results = run_tool_swarm([{"prompt": "list files in the folder"}], colony.bugs)
    assert len(results) == 1
    assert results[0]["colony_profile_saved"] is False
    assert get_colony_profile("hybrid") is None


def test_quick_swarm_auto_picks_trusted_type(monkeypatch):
    import agent_farm.colony_tools as colony_tools_module
    import agent_farm.server as server

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    _trust_card("heavy")
    result = asyncio.run(
        server.quick_swarm(
            json.dumps([{"prompt": "write a function that adds two numbers"}]),
            run_probes=False,
        )
    )
    assert result["colony_type"] == "heavy"
    assert result["colony_type_profiled"] is True


def test_quick_swarm_auto_falls_back_to_fast(monkeypatch):
    import agent_farm.server as server

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    result = asyncio.run(
        server.quick_swarm(
            json.dumps([{"prompt": "write a function that adds two numbers"}]),
            run_probes=False,
        )
    )
    assert result["colony_type"] == "fast"
    assert result["colony_type_profiled"] is False


def test_quick_swarm_explicit_type_is_used(monkeypatch):
    import agent_farm.server as server

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    _trust_card("heavy")
    result = asyncio.run(
        server.quick_swarm(
            json.dumps([{"prompt": "write a function that adds two numbers"}]),
            colony_type="standard",
            run_probes=False,
        )
    )
    assert result["colony_type"] == "standard"
    assert result["colony_type_profiled"] is False
