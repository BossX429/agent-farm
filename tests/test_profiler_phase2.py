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
    prune_stale_profiles,
    record_profile_run,
    retire_profile,
    task_type_for_prompt,
)
from agent_farm.state import MODEL_TIERS

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


def test_prune_removes_legacy_duplicate(monkeypatch, tmp_path):
    # A legacy card file is a duplicate once the machine-scoped
    # card for the same role + model exists. The prune removes
    # the legacy file and keeps the scoped card.
    import json

    from agent_farm.profiles import _card_path, _legacy_card_path

    _profile_dir(monkeypatch, tmp_path)
    record_profile_run("scout", "qwen3:4b", "list", 0.1, True)
    scoped = _card_path("scout", "qwen3:4b")
    assert scoped.exists()
    legacy = _legacy_card_path("scout", "qwen3:4b")
    card = json.loads(scoped.read_text(encoding="utf-8"))
    legacy.write_text(json.dumps(card), encoding="utf-8")
    removed = prune_stale_profiles()
    assert legacy.name in removed
    assert not legacy.exists()
    assert scoped.exists()
    assert get_profile("scout", "qwen3:4b") is not None


def test_prune_removes_orphaned_combo(monkeypatch, tmp_path):
    # A card for a role + model no bug can use anymore (its
    # model is no longer assigned to that role) is removed.
    # This is the worker-qwen3:4b card left behind when the
    # worker moved to the coder model.
    import json

    from agent_farm.profiles import _card_path, _legacy_card_path
    from agent_farm.state import MODEL_TIERS

    _profile_dir(monkeypatch, tmp_path)
    # Build an orphaned card: worker + a model no role uses.
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    assert ("worker", "qwen3:4b") not in set(MODEL_TIERS.items())
    orphan = _card_path("worker", "qwen3:4b")
    assert orphan.exists()
    # And its legacy file, which lingers because the card is
    # never saved again once orphaned.
    legacy = _legacy_card_path("worker", "qwen3:4b")
    legacy.write_text(orphan.read_text(encoding="utf-8"), encoding="utf-8")
    removed = prune_stale_profiles()
    assert orphan.name in removed
    assert legacy.name in removed
    assert not orphan.exists()
    assert not legacy.exists()
    assert get_profile("worker", "qwen3:4b") is None


def test_prune_keeps_current_cards(monkeypatch, tmp_path):
    # Cards for role + model combos a bug can actually have
    # are never touched, however many samples they hold.
    from agent_farm.state import MODEL_TIERS

    _profile_dir(monkeypatch, tmp_path)
    for role, model in MODEL_TIERS.items():
        record_profile_run(role, model, "write", 0.1, True)
    removed = prune_stale_profiles()
    assert removed == []
    for role, model in MODEL_TIERS.items():
        assert get_profile(role, model) is not None


def test_prune_keeps_other_machines_cards(monkeypatch, tmp_path):
    # A card file from another machine is not this machine's
    # to delete, even when its combo is orphaned here.
    import json

    _profile_dir(monkeypatch, tmp_path)
    other = tmp_path / "worker-qwen3-4b-40862b30-othermachine.json"
    card = create_profile("worker", "qwen3:4b")
    card["machine"] = "othermachine"
    other.write_text(json.dumps(card), encoding="utf-8")
    removed = prune_stale_profiles()
    assert removed == []
    assert other.exists()


def test_prune_never_raises(monkeypatch, tmp_path):
    # A malformed card file, a card missing its role/model, and
    # a missing profiles dir are all skipped, not raised.
    _profile_dir(monkeypatch, tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "no-role.json").write_text('{"model": "qwen3:4b"}', encoding="utf-8")
    removed = prune_stale_profiles()
    assert removed == []
    # A profiles dir that does not exist at all.
    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(tmp_path / "missing"))
    assert prune_stale_profiles() == []


def test_spawn_prunes_stale_cards(monkeypatch, tmp_path):
    # The cleanup runs on a real spawn through the MCP tool
    # layer, so a stale card left by a model change is gone
    # after the next spawn, without anyone asking for it.
    from agent_farm.profiles import _card_path

    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    record_profile_run("worker", "qwen3:4b", "write", 0.1, True)
    orphan = _card_path("worker", "qwen3:4b")
    assert orphan.exists()

    async def main():
        return await server.spawn_colony("standard", run_probes=False)

    asyncio.run(main())
    assert not orphan.exists()


def test_mixed_model_roles_are_distinct():
    # Roles deliberately run different models: the code-facing
    # roles (worker writes code, learner analyzes it) use the
    # code-tuned model, the fast roles stay on qwen3:4b, and the
    # synthesizer keeps qwen3:30b-a3b-q3k. If every role ran the same
    # model, the profiler would measure timing noise between
    # identical workers instead of real differences.
    assert MODEL_TIERS["worker"] == "qwen2.5-coder:7b"
    assert MODEL_TIERS["learner"] == "qwen2.5-coder:7b"
    assert MODEL_TIERS["scout"] == "qwen3:4b"
    assert MODEL_TIERS["memory"] == "qwen3:4b"
    assert MODEL_TIERS["guardian"] == "qwen3:4b"
    assert MODEL_TIERS["synthesizer"] == "qwen3:30b-a3b-q3k"

def test_strip_think_blocks_removes_reasoning():
    from agent_farm.ollama import strip_think_blocks

    assert strip_think_blocks("plain answer") == "plain answer"
    assert strip_think_blocks(
        "<think>Thinking out loud</think>\n\nThe actual answer."
    ) == "The actual answer."
    # A <think> block in the middle is removed, the rest kept.
    assert strip_think_blocks(
        "Before <think>reasoning here</think> after."
    ) == "Before  after."
    # An unclosed block is dropped to the end of the string.
    assert strip_think_blocks("Answer <think>never closed") == "Answer"
    # Multiple blocks are all removed.
    assert strip_think_blocks(
        "<think>one</think>keep<think>two</think> this"
    ) == "keep this"
    # A dangling </think> (reasoning text with no opening tag)
    # drops everything up to and including the close tag.
    assert strip_think_blocks(
        "Reasoning with no opening tag.</think>\n\nThe answer."
    ) == "The answer."


def test_synthesis_strips_think_block(monkeypatch):
    # The synthesis path makes a direct httpx call with no
    # "/nothink", so a qwen3 model thinks out loud. The returned
    # summary must not contain the reasoning block.
    import httpx
    import agent_farm.ollama as ollama_module

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "response": (
                    "<think>Let me think about these findings.</think>"
                    "\n\n- Local AI keeps data private."
                )
            }

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    result = ollama_module.synthesize_swarm_results(
        [{"role": "worker", "task": "write a note", "answer": "Local AI keeps data private.", "status": "success"}],
        "local AI",
    )
    assert "<think>" not in result
    assert "Let me think" not in result
    assert "Local AI keeps data private" in result


def test_prompt_for_model_instructs_review_model(monkeypatch):
    from agent_farm.state import prompt_for_model
    import agent_farm.state as state_module

    # No role uses a review-tuned model now, so the set is empty
    # and the base coder gets no instruction. The mechanism is
    # tested by putting the review model back in the set: it must
    # get the do-the-task instruction.
    assert prompt_for_model("qwen2.5-coder:7b", "write a function") == "write a function"
    monkeypatch.setattr(
        state_module,
        "MODELS_NEEDING_TASK_INSTRUCTION",
        frozenset({"qwen2.5-coder:7b-v8-review"}),
    )
    instructed = prompt_for_model(
        "qwen2.5-coder:7b-v8-review", "write a function"
    )
    assert instructed.startswith("Do the task below")
    assert "write a function" in instructed
    assert "do not say CLEAN or BUGGY" in instructed
    # Any other model gets its prompt back unchanged.
    assert prompt_for_model("qwen3:4b", "write a function") == "write a function"
    assert prompt_for_model("qwen2.5:14b", "write a function") == "write a function"


def test_colony_does_not_instruct_base_coder(monkeypatch):
    # A worker bug (review-tuned coder model) must receive the
    # do-the-task instruction, or it answers "CLEAN" instead of
    # doing the work. A scout bug (qwen3:4b) must NOT: the
    # instruction is reserved for the review-tuned model.
    seen = {}

    def _capture(model, prompt, **kwargs):
        seen.setdefault(model, []).append(prompt)
        return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(colony_module, "sync_query_ollama", _capture)
    colony = spawn_colony_internal("standard")
    # Two tasks: one routes to a worker (code), one to a scout
    # (list), so both models are called and both are checked.
    execute_parallel_swarm(
        [
            {"prompt": "write a python function that adds two numbers"},
            {"prompt": "list the files in the folder"},
        ],
        colony,
    )
    worker_prompts = seen["qwen2.5-coder:7b"]
    assert worker_prompts, "worker was never called"
    for prompt in worker_prompts:
        assert "Do the task below" not in prompt
        assert "write a python function" in prompt
    scout_prompts = seen.get("qwen3:4b", [])
    assert scout_prompts, "scout was never called"
    for prompt in scout_prompts:
        assert "Do the task below" not in prompt


def test_probe_does_not_instruct_base_coder(monkeypatch):
    # A probe for a worker bug must NOT carry the instruction
    # now: the base coder is not review-tuned, so it does the task
    # without one. Every captured prompt is checked (not
    # just the last one per model): it must carry the instruction
    # AND the probe task and context, or a probe that sends only
    # the instruction would pass while recording garbage. A scout
    # probe (qwen3:4b) must NOT get the instruction.
    import agent_farm.probes as probes_module

    seen = {}

    def _capture(model, prompt, **kwargs):
        seen.setdefault(model, []).append(prompt)
        return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _capture)
    colony = spawn_colony_internal("standard")
    probes_module.run_probes(colony)
    worker_prompts = seen["qwen2.5-coder:7b"]
    assert worker_prompts, "worker was never probed"
    for prompt in worker_prompts:
        assert "Do the task below" not in prompt
        assert "Task:" in prompt
        assert "Context:" in prompt
    scout_prompts = seen.get("qwen3:4b", [])
    assert scout_prompts, "scout was never probed"
    for prompt in scout_prompts:
        assert "Do the task below" not in prompt
        assert "Task:" in prompt


def test_chunked_tools_do_not_instruct_base_coder(monkeypatch):
    # A chunked section write uses the worker model directly
    # (query_ollama_sync, no structured prompt). The base coder is
    # not review-tuned, so it must NOT carry the do-the-task
    # instruction.
    import agent_farm.chunked_tools as chunked_tools_module
    import agent_farm.agent as agent_module

    seen = {}

    def _capture(model, prompt, **kwargs):
        seen.setdefault(model, []).append(prompt)
        return "Section content here."

    monkeypatch.setattr(agent_module, "query_ollama_sync", _capture)
    chunked_tools_module.generate_section_content(
        {"title": "Intro", "description": "An intro"}, "A doc", 1, 3
    )
    worker_prompts = seen["qwen2.5-coder:7b"]
    assert worker_prompts, "worker was never called"
    for prompt in worker_prompts:
        assert "Do the task below" not in prompt


def test_check_correctness_right_answers():
    # Every probe job's known right answer passes, even when
    # phrased differently.
    from agent_farm.probes import check_correctness

    assert check_correctness("write", "def add(a, b):\n    return a + b") is True
    assert check_correctness("write", "Here it is: def add(x, y): return x+y") is True
    assert check_correctness("write", "def add(a, b): return (a + b)") is True
    assert check_correctness("write", "def add(a1, b2): return a1 + b2") is True
    assert check_correctness(
        "analyze", "The bug is a division by zero when the list is empty."
    ) is True
    assert check_correctness("analyze", "len(nums) is zero, so it crashes.") is True
    assert check_correctness("analyze", "The list is empty, so it fails.") is True
    assert check_correctness("search", "Colony C has the most bugs, 6.") is True
    assert check_correctness("search", "C") is True
    assert check_correctness("search", "The third colony has the most bugs.") is True
    assert check_correctness("check", "The ollama service is up.") is True
    assert check_correctness("check", "up") is True
    assert check_correctness("check", "Ollama is running.") is True
    assert check_correctness("check", "Ollama is online.") is True
    assert check_correctness("list", "The Python files are main.py and helper.py.") is True


def test_check_correctness_wrong_answers():
    # A wrong answer, a bare verdict, and a vague dodge all fail.
    from agent_farm.probes import check_correctness

    assert check_correctness("write", "def add(a, b):\n    return a - b") is False
    assert check_correctness("write", "CLEAN") is False
    assert check_correctness("write", "Here is the function.") is False
    # An unrelated function that happens to contain a sum is not
    # the requested add function.
    assert check_correctness(
        "write", "def multiply(x, y): return x * y\n# and a + b is 5"
    ) is False
    # A function with the wrong name that returns a sum is not
    # the requested add function either.
    assert check_correctness(
        "write", "def subtract(a, b):\n    return a + b"
    ) is False
    assert check_correctness("analyze", "The code looks fine.") is False
    assert check_correctness("analyze", "BUGGY") is False
    assert check_correctness("analyze", "The answer is zero.") is False
    assert check_correctness("search", "Colony A has the most bugs, 4.") is False
    assert check_correctness("search", "Colony B") is False
    assert check_correctness("check", "The ollama service is down.") is False
    assert check_correctness("list", "The files are notes.txt and README.md.") is False
    assert check_correctness("list", "main.py") is False  # helper.py missing


def test_check_correctness_no_judgment():
    # A failed response is not an incorrect answer, and an
    # unknown task type has no checker.
    from agent_farm.probes import check_correctness

    assert check_correctness("write", "") is None
    assert check_correctness("write", None) is None
    assert check_correctness("write", "Error: timeout") is None
    assert check_correctness("general", "anything") is None


def test_probe_records_correctness_on_card(monkeypatch):
    # A probe run records correctness on the card: a right
    # answer counts as correct, and the card's correctness_rate
    # reflects it. A wrong answer counts as checked but not
    # correct.
    import agent_farm.probes as probes_module
    from agent_farm.profiles import _load_card

    def _right(model, prompt, **kwargs):
        # The write probe's context asks for the add function.
        return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _right)
    colony = spawn_colony_internal("standard")
    probes_module.run_probes(colony)
    card = _load_card("worker", "qwen2.5-coder:7b")
    write_entry = card["per_task_type"]["write"]
    assert write_entry["checked"] >= 1
    assert write_entry["correct"] == write_entry["checked"]
    assert write_entry["correctness_rate"] == 1.0
    assert card["correctness_rate"] > 0.0


def test_wrong_answer_does_not_count_as_correct(monkeypatch):
    # Recording a checked WRONG answer must increment checked
    # but not correct, and correctness_rate must be 0.0. A bug
    # that increments correct for a false result would pass the
    # right-answer test while inflating the card.
    import agent_farm.probes as probes_module
    from agent_farm.profiles import _load_card

    def _wrong(model, prompt, **kwargs):
        # Every probe job gets a wrong answer: the write probe
        # gets a subtraction, the others get a wrong value.
        return "def add(a, b):\n    return a - b"

    monkeypatch.setattr(probes_module, "sync_query_ollama", _wrong)
    colony = spawn_colony_internal("standard")
    probes_module.run_probes(colony)
    card = _load_card("worker", "qwen2.5-coder:7b")
    write_entry = card["per_task_type"]["write"]
    assert write_entry["checked"] >= 1
    assert write_entry["correct"] == 0
    assert write_entry["correctness_rate"] == 0.0


def test_failed_run_does_not_count_as_checked():
    # A failed response must not count as a checked answer on a
    # role card either, even if a caller passes a correct value
    # alongside success=False.
    from agent_farm.profiles import _load_card, record_profile_run

    assert record_profile_run("worker", "qwen3:4b", "write", 1.0, False, "error", True)
    card = _load_card("worker", "qwen3:4b")
    assert card["checked"] == 0
    assert card["correct"] == 0
    assert card["per_task_type"]["write"]["checked"] == 0


def test_old_card_without_correctness_loads():
    # A card written before correctness was measured has no
    # checked / correct fields; loading it must fill them with
    # zero instead of breaking.
    from agent_farm.profiles import _normalize_card

    old = {
        "role": "worker",
        "model": "qwen2.5-coder:7b",
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
    normalized = _normalize_card(old, "worker", "qwen2.5-coder:7b")
    assert normalized is not None
    assert normalized["checked"] == 0
    assert normalized["correct"] == 0
    assert normalized["correctness_rate"] == 0.0
    assert normalized["per_task_type"]["write"]["checked"] == 0


def test_synthesizer_is_the_30b_reasoning_model():
    # The synthesizer is the heavy-reasoning role: it runs AFTER
    # colony work completes, so Ollama evicts the finished colony
    # models to make room for the 14.7GB 30b. It must be the 30b
    # (not the old 14b), and it must be distinct from every colony
    # role's model, so the profiler measures a real difference.
    assert MODEL_TIERS["synthesizer"] == "qwen3:30b-a3b-q3k"
    colony_models = {
        MODEL_TIERS[role]
        for role in ("scout", "worker", "memory", "guardian", "learner")
    }
    assert MODEL_TIERS["synthesizer"] not in colony_models
    # At least two different models are actually in use.
    assert len(set(MODEL_TIERS.values())) >= 3


def test_live_colony_bugs_carry_their_role_model(monkeypatch, tmp_path):
    # A spawned colony's bugs carry the model their role is
    # configured for, so the profiler's per-role+model cards
    # measure the model that actually ran.
    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    assert len(colony.bugs) > 0
    for bug in colony.bugs:
        assert bug.model == MODEL_TIERS[bug.role], bug.role
    # The standard colony has both models in it.
    models = {bug.model for bug in colony.bugs}
    assert "qwen3:4b" in models
    assert "qwen2.5-coder:7b" in models


def test_worker_card_is_keyed_by_coder_model(monkeypatch, tmp_path):
    # Probes feed the worker card under the coder model, not
    # under qwen3:4b: the card measures the model that ran.
    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(probes_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    run_probes(colony)
    card = get_profile("worker", MODEL_TIERS["worker"])
    assert card is not None
    assert card["model"] == "qwen2.5-coder:7b"
    assert card["samples"] == 5
    # No card exists for the worker under the old model.
    assert get_profile("worker", "qwen3:4b") is None


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
        card = get_profile(role, MODEL_TIERS[role])
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
    card = get_profile("worker", MODEL_TIERS["worker"])
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
    assert get_profile("worker", MODEL_TIERS["worker"])["samples"] == 5


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
    card = get_profile("worker", MODEL_TIERS["worker"])
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
    worker_card = get_profile("worker", MODEL_TIERS["worker"])
    assert worker_card["per_task_type"]["write"]["samples"] == 1
    assert worker_card["per_task_type"]["write"]["successes"] == 1
    scout_card = get_profile("scout", "qwen3:4b")
    assert scout_card["per_task_type"]["list"]["samples"] == 1

    # A grounded call that answers UNCERTAIN counts as a
    # completed call in Phase 1 metrics but as a failure on
    # the card: the call worked, the answer did not.
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_uncertain)
    execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    worker_card = get_profile("worker", MODEL_TIERS["worker"])
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
        "checked": 0,
        "correct": 0,
        "correctness_rate": 0.0,
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


def test_router_prefers_higher_correctness(monkeypatch, tmp_path):
    # An "analyze" prompt must go to the role that is actually
    # right at analyze, not the one that merely answers. The
    # worker (coder) is wrong on every checked analyze answer
    # (correctness 0.0); memory (qwen3:4b) is right on every one
    # (correctness 1.0). The worker is also faster, so a picker
    # that ignores correctness picks the worker on the
    # success-and-speed tie-breakers. The correctness picker
    # must pick memory.
    from agent_farm.colony import get_best_bug_for_task

    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            correct = task_type != "analyze"
            record_profile_run(
                "worker", "qwen2.5-coder:7b-v8-review", task_type, 0.1, True, None, correct
            )
            record_profile_run("memory", "qwen3:4b", task_type, 1.0, True, None, True)
    picked = get_best_bug_for_task({"prompt": "analyze code"}, colony)
    assert picked.role == "memory"


def test_router_all_wrong_role_loses(monkeypatch, tmp_path):
    # A role measured on 5 analyze answers and wrong on all of
    # them must lose to a role with no correctness data, even
    # when the all-wrong role answers successfully and the
    # unmeasured one does not. Memory's analyze runs are failures
    # (success 0.0), so its fallback quality is 0.0 and it cannot
    # win on success alone: only the all-wrong rule (worker
    # quality -1.0) makes memory win. Without the rule the pick
    # ties on quality 0.0 and the worker wins on success.
    from agent_farm.colony import get_best_bug_for_task

    _profile_dir(monkeypatch, tmp_path)
    colony = spawn_colony_internal("standard")
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            correct = task_type != "analyze"
            record_profile_run(
                "worker", "qwen2.5-coder:7b-v8-review", task_type, 0.1, True, None, correct
            )
            success = task_type != "analyze"
            record_profile_run(
                "memory", "qwen3:4b", task_type, 1.0, success, None, None
            )
    # Memory has no correctness data for analyze; worker is
    # measured all-wrong. Memory must win.
    picked = get_best_bug_for_task({"prompt": "analyze code"}, colony)
    assert picked.role == "memory"


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
            record_profile_run("worker", MODEL_TIERS["worker"], task_type, 0.1, True)
    assert get_profile("worker", MODEL_TIERS["worker"])["trusted"] is True
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
            record_profile_run("worker", MODEL_TIERS["worker"], task_type, 0.1, True)
    assert get_profile("worker", MODEL_TIERS["worker"])["trusted"] is True
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


def test_inconsistent_card_entry_not_profiled(monkeypatch, tmp_path):
    # A card that claims to be trusted overall but has an
    # untrusted entry for the requested type is not evidence
    # for that type. The entry check must be enforced, not
    # just the card-level flag.
    import json

    from agent_farm.profiles import _card_path

    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    # Build a trusted worker card, then corrupt one entry: the
    # card still says trusted, but the write entry does not.
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", MODEL_TIERS["worker"], task_type, 0.1, True)
    assert get_profile("worker", MODEL_TIERS["worker"])["trusted"] is True
    card = get_profile("worker", MODEL_TIERS["worker"])
    card["per_task_type"]["write"]["trusted"] = False
    with open(_card_path("worker", MODEL_TIERS["worker"]), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, sort_keys=True)
    colony = spawn_colony_internal("standard")
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["profiled"] is False
    reset_metrics()


def test_load_spread_swap_is_not_profiled(monkeypatch, tmp_path):
    # Two write tasks, both workers with trusted cards, the
    # first worker faster. The first task takes the profile
    # pick; the second wants the same bug, so load-spreading
    # swaps it to the other worker. That second bug has a
    # trusted card too, but it was not the profile pick, so
    # its result is not profiled.
    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    colony = spawn_colony_internal("standard")
    workers = [b for b in colony.bugs if b.role == "worker"]
    assert len(workers) == 2
    workers[1].model = "qwen2.5:14b"
    for task_type in PROBE_TASK_TYPES:
        for _ in range(MIN_SAMPLES_PER_TYPE):
            record_profile_run("worker", MODEL_TIERS["worker"], task_type, 0.1, True)
            record_profile_run("worker", "qwen2.5:14b", task_type, 0.5, True)
    assert get_profile("worker", MODEL_TIERS["worker"])["trusted"] is True
    assert get_profile("worker", "qwen2.5:14b")["trusted"] is True
    results = execute_parallel_swarm([
        {"prompt": "write a short note"},
        {"prompt": "write another note"},
    ], colony)
    assert len(results) == 2
    by_task = {r["task"]: r for r in results}
    assert by_task["write a short note"]["profiled"] is True
    assert by_task["write another note"]["profiled"] is False
    reset_metrics()


def test_profiled_captured_at_assignment(monkeypatch, tmp_path):
    # The flag is fixed when the router picks, not when the
    # result lands. A card that becomes trusted during the
    # model call must not rewrite a fallback pick as profiled.
    _profile_dir(monkeypatch, tmp_path)
    reset_metrics()

    def _fake_trusts_card(model, prompt, **kwargs):
        time.sleep(_FAKE_DELAY)
        for task_type in PROBE_TASK_TYPES:
            for _ in range(MIN_SAMPLES_PER_TYPE):
                record_profile_run("worker", MODEL_TIERS["worker"], task_type, 0.1, True)
        return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_trusts_card)
    colony = spawn_colony_internal("standard")
    # No card at assignment time: fallback, even though the
    # fake trusts the card before the call returns.
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["profiled"] is False
    assert get_profile("worker", MODEL_TIERS["worker"])["trusted"] is True
    # The card is trusted now, so the next pick is profiled.
    monkeypatch.setattr(colony_module, "sync_query_ollama", _fake_ok)
    results = execute_parallel_swarm([{"prompt": "write a short note"}], colony)
    assert len(results) == 1
    assert results[0]["profiled"] is True
    reset_metrics()


def test_direct_write_and_no_bug_results_have_profiled(monkeypatch, tmp_path):
    # Every task result carries the flag, including the ones
    # that never reach a bug: a direct write and a task with
    # no bugs available. Neither was profile-driven.
    import agent_farm.agent as agent_module
    from agent_farm.agent import run_tool_swarm

    _profile_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(agent_module, "execute_tool", lambda *a, **k: {})
    results = run_tool_swarm(
        [{"path": "note.txt", "content": "hello"}],
        spawn_colony_internal("standard").bugs,
    )
    assert len(results) == 1
    assert results[0]["direct_execute"] is True
    assert results[0]["profiled"] is False
    results = run_tool_swarm([{"prompt": "list files in the folder"}], [])
    assert len(results) == 1
    assert results[0]["bug_id"] is None
    assert results[0]["profiled"] is False
