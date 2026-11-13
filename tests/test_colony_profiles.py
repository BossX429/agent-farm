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
    # One sample per remaining type is not enough either: a
    # threshold weakened to 1 sample per type would trust the
    # card here, and the old test would not have noticed.
    for task_type in ("analyze", "search", "check", "list"):
        assert record_colony_run("fast", task_type, 1.0, True)
    card = get_colony_profile("fast")
    assert card["trusted"] is False
    for task_type in ("analyze", "search", "check", "list"):
        for _ in range(MIN_SAMPLES_PER_TYPE - 1):
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


def test_pick_prefers_higher_correctness():
    # Both types answer everything for "write" (success 1.0),
    # but heavy is right and fast is wrong. The pick must choose
    # heavy: correctness beats a success rate that hides wrong
    # answers.
    _trust_card("fast")
    _trust_card("heavy")
    for _ in range(MIN_SAMPLES_PER_TYPE):
        assert record_colony_run("heavy", "write", 1.0, True, None, True)
        assert record_colony_run("fast", "write", 1.0, True, None, False)
    assert colony_type_pick("write") == "heavy"


def test_pick_prefers_measured_correctness_over_unmeasured():
    # Heavy has been measured and is right for "write"
    # (correctness 1.0). Fast has a lower success rate and no
    # correctness data yet, so it competes on success. Heavy
    # wins on its measured correctness.
    _trust_card("fast")
    _trust_card("heavy")
    assert record_colony_run("fast", "write", 1.0, False, "error")
    for _ in range(MIN_SAMPLES_PER_TYPE):
        assert record_colony_run("heavy", "write", 1.0, True, None, True)
    assert colony_type_pick("write") == "heavy"


def test_probe_records_correctness_on_colony_card(monkeypatch):
    # A probe run records correctness on the colony type's card
    # as well as the role cards, so the pick has correctness to
    # rank on.
    import agent_farm.probes as probes_module

    def _right(model, prompt, **kwargs):
        return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _right)
    colony = spawn_colony_internal("standard")
    probes_module.run_probes(colony)
    card = get_colony_profile("standard")
    write_entry = card["per_task_type"]["write"]
    assert write_entry["checked"] >= 1
    assert write_entry["correct"] == write_entry["checked"]
    assert write_entry["correctness_rate"] == 1.0
    assert card["correctness_rate"] > 0.0


def test_old_colony_card_without_correctness_loads():
    # A colony card written before correctness was measured has
    # no checked / correct fields; loading it must fill them
    # with zero instead of breaking.
    from agent_farm.colony_profiles import _normalize_colony_card

    old = {
        "colony_type": "heavy",
        "machine": "him",
        "samples": 10,
        "successes": 9,
        "failures": 1,
        "success_rate": 0.9,
        "total_seconds": 50.0,
        "avg_seconds": 5.0,
        "per_task_type": {},
        "failure_modes": {},
        "trusted": False,
        "stale": False,
    }
    normalized = _normalize_colony_card(old, "heavy")
    assert normalized is not None
    assert normalized["checked"] == 0
    assert normalized["correct"] == 0
    assert normalized["correctness_rate"] == 0.0
    assert normalized["per_task_type"]["write"]["checked"] == 0


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


def _trust_card_for_type(colony_type, good_type):
    """Trust a card that is good at one type and bad at the rest.

    Every probe type gets 5 samples (so the card is trusted),
    but only good_type succeeds; the others fail. That makes the
    pick task-specific: a write task and an analyze task must
    choose different cards.
    """
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            assert record_colony_run(
                colony_type, task_type, 1.0, task_type == good_type
            )


def test_resolve_colony_type_auto_picks_trusted():
    _trust_card_for_type("heavy", "write")
    _trust_card_for_type("standard", "analyze")
    colony_type, profiled = resolve_colony_type(
        "auto", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "heavy"
    assert profiled is True
    colony_type, profiled = resolve_colony_type(
        "auto", [{"prompt": "analyze code for bugs"}], "fast"
    )
    assert colony_type == "standard"
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


def test_pick_uses_unrounded_metrics():
    # Two trusted cards whose rounded success rate and avg seconds
    # are identical (1.0 and 2.0) but whose unrounded averages
    # differ: fast averages 2.004s, heavy 2.001s. The pick must
    # choose heavy (faster unrounded). A pick that compared the
    # rounded values would tie and fall back to sorted card order,
    # choosing fast.
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            assert record_colony_run("fast", task_type, 2.004, True)
            assert record_colony_run("heavy", task_type, 2.001, True)
    assert colony_type_pick("write") == "heavy"


def test_malformed_extra_entry_is_dropped(monkeypatch, tmp_path):
    # A stored card with an unknown per_task_type key whose value
    # lacks the expected fields must not crash recording: the
    # normalize step drops unknown keys.
    from agent_farm.colony_profiles import _colony_card_path, _save_colony_card

    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    card = create_colony_profile("fast")
    card["per_task_type"]["bogus"] = {"wrong": "shape"}
    assert _save_colony_card(card)
    assert record_colony_run("fast", "write", 1.0, True) is True
    loaded = get_colony_profile("fast")
    assert loaded is not None
    assert "bogus" not in loaded["per_task_type"]
    assert loaded["samples"] == 1


def test_corrupt_card_bytes_do_not_break_recording(monkeypatch, tmp_path):
    # A card file with invalid UTF-8 bytes must be treated as
    # unusable, not raise UnicodeDecodeError out of recording,
    # listing, or pruning.
    from agent_farm.colony_profiles import _colony_card_path

    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    path = _colony_card_path("fast")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe not json \x80\x81")
    assert record_colony_run("fast", "write", 1.0, True) is True
    assert get_colony_profile("fast") is not None
    assert list_colony_profiles() != []
    assert prune_stale_colony_profiles() == []


def test_unsupported_explicit_type_falls_back():
    colony_type, profiled = resolve_colony_type(
        "auto", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "fast"
    colony_type, profiled = resolve_colony_type(
        "mega", [{"prompt": "write a function"}], "fast"
    )
    assert colony_type == "fast"
    assert profiled is False
    # An unsupported type never gains a card.
    assert record_colony_run("mega", "write", 1.0, True) is False
    assert get_colony_profile("mega") is None


def test_spawn_normalizes_unsupported_type():
    colony = spawn_colony_internal("mega")
    assert colony.colony_type == "standard"
    assert [bug.role for bug in colony.bugs] == [
        "scout", "worker", "worker", "memory",
    ]


def test_positional_colony_construction():
    # Colony(id, bugs, consciousness) positionally must still bind
    # consciousness correctly: colony_type is the LAST field, not
    # the third, so it cannot silently capture the consciousness
    # object.
    from agent_farm.state import Bug, Colony, SharedConsciousness

    consciousness = SharedConsciousness()
    colony = Colony("c1", [Bug(id="b1", role="worker", model="qwen3:4b")], consciousness)
    assert colony.consciousness is consciousness
    assert colony.colony_type == "standard"


def test_dominant_task_type_handles_bad_input():
    assert dominant_task_type({"prompt": "write a function"}) == "general"
    assert dominant_task_type([{"prompt": "write a function"}, "not a dict"]) == "write"
    assert dominant_task_type([None, 42]) == "general"


def test_quick_swarm_object_tasks_returns_validation_error(monkeypatch):
    # A JSON object (not a list) must return the validation error,
    # not crash in dominant_task_type before validation runs.
    import agent_farm.server as server

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    result = asyncio.run(
        server.quick_swarm('{"prompt": "write a function"}', run_probes=False)
    )
    assert "error" in result


def test_stale_lock_is_recovered(monkeypatch, tmp_path):
    # A lock file older than 30 seconds is treated as stale (its
    # process died) and removed, so recording proceeds.
    import os
    import time
    from agent_farm.colony_profiles import _colony_card_path

    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    path = _colony_card_path("fast")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    lock_path.touch()
    old = time.time() - 60.0
    os.utime(lock_path, (old, old))
    assert record_colony_run("fast", "write", 1.0, True) is True
    assert not lock_path.exists()


def test_lock_timeout_skips_update(monkeypatch, tmp_path):
    # A lock held by another process (fresh, not stale) must not
    # be yielded without ownership: recording returns False
    # instead of overwriting another recorder's samples.
    from agent_farm.colony_profiles import _colony_card_path

    monkeypatch.setenv("AGENT_FARM_COLONY_PROFILES_DIR", str(tmp_path))
    path = _colony_card_path("fast")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    lock_path.touch()
    assert record_colony_run("fast", "write", 1.0, True) is False
    assert get_colony_profile("fast") is None


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
    import agent_farm.server as server

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    _trust_card_for_type("heavy", "write")
    _trust_card_for_type("standard", "analyze")
    result = asyncio.run(
        server.quick_swarm(
            json.dumps([{"prompt": "write a function that adds two numbers"}]),
            run_probes=False,
        )
    )
    assert result["colony_type"] == "heavy"
    assert result["colony_type_profiled"] is True
    result = asyncio.run(
        server.quick_swarm(
            json.dumps([{"prompt": "analyze code for bugs"}]),
            run_probes=False,
        )
    )
    assert result["colony_type"] == "standard"
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
