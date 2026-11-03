"""Tests for the ChatGPT medium reliability fixes (PR: medium reliability).

1. Blocking work is moved off the async event loop (asyncio.to_thread
   + a shared model semaphore) instead of waiting in the loop.
2. chunked_code_gen validates generated Python and distinguishes
   "written" from "validated" -- validation is a compile/AST check,
   never a claim that the code was run or tested.
3. Parallel tools reject empty / non-list / over-long inputs and
   cap aggregate output, instead of creating a zero-worker pool or
   returning unbounded text.

All tests run offline: model calls are mocked, no Ollama, no network.
"""

import asyncio
import json
import threading
import time

import pytest

from agent_farm import reliability
from agent_farm.reliability import (
    cap_aggregate_output,
    run_parallel,
    validate_parallel_items,
    validate_parallel_payload,
    validate_python_code,
)
from agent_farm.server import (
    api_swarm,
    chunked_code_gen,
    chunked_write,
    exec_swarm,
    file_swarm,
)


def run(coro):
    return asyncio.run(coro)


# ---------- parallel input validation ----------

def test_validate_parallel_items_rejects_empty_and_non_list():
    assert validate_parallel_items([], "tasks") is not None
    assert "must not be empty" in validate_parallel_items([], "tasks")
    assert validate_parallel_items("not a list", "tasks") is not None
    assert validate_parallel_items({"a": 1}, "tasks") is not None


def test_validate_parallel_items_rejects_too_many_and_accepts_limit():
    assert validate_parallel_items([1] * 33, "tasks") is not None
    assert validate_parallel_items([1] * 32, "tasks") is None


def test_validate_parallel_payload_rejects_oversize():
    assert validate_parallel_payload("x" * 100_001, "tasks") is not None
    assert validate_parallel_payload("x" * 100_000, "tasks") is None


def test_exec_swarm_rejects_empty_non_list_and_too_many():
    assert "error" in run(exec_swarm("[]"))
    assert "error" in run(exec_swarm(json.dumps([123])))
    assert "error" in run(exec_swarm(json.dumps(["echo hi"] * 33)))


def test_api_swarm_and_file_swarm_reject_empty():
    assert "error" in run(api_swarm("[]"))
    assert "error" in run(file_swarm("[]"))
    assert "error" in run(file_swarm(json.dumps([{"path": "/tmp/x"}])))


def test_exec_swarm_still_runs_and_reports_truncation_flag():
    result = run(exec_swarm(json.dumps(["echo hello"])))
    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["stdout"].strip() == "hello"
    assert "output_truncated" in result
    assert result["output_truncated"] is False


# ---------- shared parallel runner ----------

def test_run_parallel_preserves_order_and_caps_workers():
    current = {"n": 0, "max": 0}
    lock = threading.Lock()

    def work(i):
        with lock:
            current["n"] += 1
            current["max"] = max(current["max"], current["n"])
        time.sleep(0.05)
        with lock:
            current["n"] -= 1
        return i * 2

    results = run_parallel([lambda i=i: work(i) for i in range(6)], max_workers=2)
    assert results == [0, 2, 4, 6, 8, 10]
    assert current["max"] <= 2


def test_run_parallel_timeout_and_failure_do_not_stop_others():
    def slow():
        time.sleep(0.3)
        return "slow"

    def boom():
        raise ValueError("kaboom")

    results = run_parallel([lambda: "ok", slow, boom], max_workers=3, timeout=0.05)
    assert results[0] == "ok"
    assert results[1]["status"] == "timeout"
    assert results[2]["status"] == "failed"
    assert "kaboom" in results[2]["error"]


def test_cap_aggregate_output_truncates_total():
    results = [
        {"content": "a" * 30000},
        {"content": "b" * 30000},
    ]
    capped, truncated = cap_aggregate_output(results, ("content",))
    assert truncated is True
    assert sum(len(r["content"]) for r in capped) <= reliability.MAX_AGGREGATE_OUTPUT_CHARS
    assert capped[0]["content"] == "a" * 30000
    assert capped[1].get("truncated") is True


# ---------- model concurrency gate ----------

def test_model_semaphore_limits_simultaneous_model_calls(monkeypatch):
    import agent_farm.agent as agent_mod

    current = {"n": 0, "max": 0}
    lock = threading.Lock()

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"response": "ok"}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            with lock:
                current["n"] += 1
                current["max"] = max(current["max"], current["n"])
            time.sleep(0.05)
            with lock:
                current["n"] -= 1
            return FakeResponse()

    monkeypatch.setattr(agent_mod.httpx, "Client", FakeClient)
    results = run_parallel(
        [lambda: agent_mod.query_ollama_sync("qwen3:4b", "hi") for _ in range(8)],
        max_workers=8,
    )
    assert results == ["ok"] * 8
    assert current["max"] <= reliability.MAX_MODEL_CONCURRENCY


# ---------- Python code validation ----------

def test_validate_python_code_valid():
    result = validate_python_code("def add(a, b):\n    return a + b\n")
    assert result["validated"] is True
    assert result["syntax_ok"] is True
    assert result["errors"] == []
    assert result["function_names"] == ["add"]


def test_validate_python_code_syntax_error():
    result = validate_python_code("def broken(:\n    pass\n")
    assert result["validated"] is False
    assert result["syntax_ok"] is False
    assert result["errors"]


def test_validate_python_code_duplicate_functions():
    result = validate_python_code(
        "def foo():\n    pass\n\ndef foo():\n    pass\n"
    )
    assert result["validated"] is False
    assert any("Duplicate" in e for e in result["errors"])


def test_validate_python_code_possible_missing_import_is_warning_only():
    result = validate_python_code("def f():\n    return json.dumps({})\n")
    assert result["validated"] is True
    assert any("json" in w for w in result["warnings"])


def test_validate_python_code_empty_and_failed_fragment():
    assert validate_python_code("").get("validated") is False
    result = validate_python_code(
        "def f():\n    pass\n# Function 1 generation failed: boom\n"
    )
    assert result["validated"] is False


# ---------- chunked_code_gen: written vs validated ----------

def _mock_code_gen(monkeypatch, worker_bodies):
    import agent_farm.agent as agent_mod

    def fake_structured(model, prompt, schema=None):
        return {
            "sections": [
                {"title": f"func_{i}", "description": f"Function {i}"}
                for i in range(len(worker_bodies))
            ]
        }

    bodies = iter(worker_bodies)

    def fake_sync(model, prompt):
        return next(bodies)

    monkeypatch.setattr(agent_mod, "query_ollama_structured", fake_structured)
    monkeypatch.setattr(agent_mod, "query_ollama_sync", fake_sync)


def test_chunked_code_gen_validated(monkeypatch, tmp_path):
    _mock_code_gen(monkeypatch, [
        "def func_0():\n    return 1\n",
        "def func_1():\n    return 2\n",
    ])
    out = tmp_path / "out.py"
    result = run(chunked_code_gen(str(out), "two helpers", "python", 2))
    assert result["status"] == "validated"
    assert result["validation"]["validated"] is True
    assert out.exists()


def test_chunked_code_gen_duplicate_is_written_not_validated(monkeypatch, tmp_path):
    _mock_code_gen(monkeypatch, [
        "def same():\n    return 1\n",
        "def same():\n    return 2\n",
    ])
    out = tmp_path / "out.py"
    result = run(chunked_code_gen(str(out), "two helpers", "python", 2))
    assert result["status"] == "written_not_validated"
    assert result["validation"]["validated"] is False
    assert out.exists()


def test_chunked_code_gen_syntax_error_is_written_not_validated(monkeypatch, tmp_path):
    _mock_code_gen(monkeypatch, [
        "def func_0():\n    return 1\n",
        "def broken(:\n    pass\n",
    ])
    out = tmp_path / "out.py"
    result = run(chunked_code_gen(str(out), "two helpers", "python", 2))
    assert result["status"] == "written_not_validated"
    assert result["validation"]["syntax_ok"] is False


def test_chunked_code_gen_non_python_is_not_claimed_validated(monkeypatch, tmp_path):
    _mock_code_gen(monkeypatch, [
        "function func_0() { return 1; }",
        "function func_1() { return 2; }",
    ])
    out = tmp_path / "out.js"
    result = run(chunked_code_gen(str(out), "two helpers", "javascript", 2))
    assert result["status"] == "written_not_validated"
    assert result["validation"]["validated"] is False


# ---------- event loop stays responsive ----------

def test_chunked_write_does_not_block_event_loop(monkeypatch, tmp_path):
    import agent_farm.chunked_tools as chunked_mod

    def slow_plan(spec, num_sections):
        time.sleep(0.3)
        return [
            {"title": "A", "description": "first"},
            {"title": "B", "description": "second"},
        ]

    monkeypatch.setattr(chunked_mod, "plan_sections", slow_plan)
    monkeypatch.setattr(
        chunked_mod, "_run_section_generation",
        lambda sections, spec, n: ["hello", "world"],
    )

    async def main():
        ticks = {"n": 0}

        async def ticker():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        result = await chunked_write(str(tmp_path / "doc.md"), "topic", 2)
        ticker_task.cancel()
        return result, ticks["n"]

    result, ticks = run(main())
    # chunked_write status is now validated/written_not_validated
    # (Copilot roadmap #1, same vocabulary as chunked_code_gen),
    # not the old "completed" -- this test is about loop responsiveness,
    # not the status word, so accept either validated outcome.
    assert result["status"] in ("validated", "written_not_validated")
    # If chunked_write blocked the loop for 0.3s, the ticker could not
    # have ticked at all; a responsive loop ticks many times.
    assert ticks >= 5


# ---------- internal parallel guards ----------

def test_execute_parallel_swarm_empty_does_not_create_pool():
    from agent_farm.colony import execute_parallel_swarm, spawn_colony_internal
    from agent_farm.state import Colony

    colony = spawn_colony_internal("fast")
    assert execute_parallel_swarm([], colony) == []
    assert execute_parallel_swarm([{"prompt": "hi"}], Colony(id="x", bugs=[])) == []


def test_run_tool_swarm_without_bugs_fails_tasks_instead_of_crashing():
    from agent_farm.agent import run_tool_swarm

    results = run_tool_swarm([{"prompt": "hi"}], [])
    assert results[0]["status"] == "failed"
