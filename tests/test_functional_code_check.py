"""Tests for the functional code check and the UTF-8 write fix.

The functional check is a scope-aware undefined-name analysis, not
execution: generated code is never run during validation. It exists
because the harder GPU test (2026-11-04) produced an 8-function
task-list library that compiled and "validated", yet crashed the
first time it ran -- ``add_task`` appends to a global ``tasks`` the
file never creates, while ``search_tasks(tasks, ...)`` made the old
flat defined/used check think ``tasks`` was defined.

The UTF-8 fix exists because the same test's guide came back with
smart punctuation as bytes that are not valid UTF-8: on Windows,
``open(path, 'w')`` with no encoding writes in cp1252.
"""

import asyncio

from agent_farm.chunked_tools import chunked_code_gen
from agent_farm.reliability import find_undefined_names, validate_python_code
from agent_farm.tools import tool_read_file, tool_write_file


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# The hard GPU test's failure, reduced to the essential shape.
HARD_TASKS_CODE = (
    "def add_task(title):\n"
    "    tasks.append({'title': title})\n"
    "\n"
    "\n"
    "def search_tasks(tasks, keyword):\n"
    "    return [t for t in tasks if keyword in t['title']]\n"
)


def test_find_undefined_names_catches_global_never_defined():
    # ``tasks`` is an argument in search_tasks, so a flat
    # defined/used set would wrongly think it is defined.
    assert find_undefined_names(HARD_TASKS_CODE) == ["tasks"]


def test_find_undefined_names_allows_defined_global_and_closure():
    code = (
        "tasks = []\n"
        "\n"
        "\n"
        "def add_task(title):\n"
        "    tasks.append({'title': title})\n"
        "\n"
        "\n"
        "def outer():\n"
        "    z = 3\n"
        "    def inner():\n"
        "        return z\n"
        "    return inner()\n"
    )
    assert find_undefined_names(code) == []


def test_find_undefined_names_ignores_missing_module_import():
    # A missing import stays a warning in validate_python_code, not
    # an undefined-name error here.
    assert find_undefined_names("def f():\n    return json.dumps({})\n") == []


def test_find_undefined_names_bad_syntax_returns_empty():
    assert find_undefined_names("def broken(:\n    pass\n") == []


def test_validate_python_code_undefined_name_is_not_validated():
    result = validate_python_code(HARD_TASKS_CODE)
    assert result["validated"] is False
    assert result["syntax_ok"] is True
    assert any("Undefined name: tasks" in e for e in result["errors"])


def test_validate_python_code_defined_global_still_validates():
    code = (
        "tasks = []\n"
        "\n"
        "\n"
        "def add_task(title):\n"
        "    tasks.append({'title': title})\n"
    )
    result = validate_python_code(code)
    assert result["validated"] is True
    assert result["errors"] == []


def test_chunked_code_gen_undefined_name_is_written_not_validated(monkeypatch, tmp_path):
    import agent_farm.agent as agent_mod

    def fake_structured(model, prompt, schema=None):
        return {
            "sections": [
                {"title": "func_0", "description": "Function 0"},
                {"title": "func_1", "description": "Function 1"},
            ]
        }

    bodies = iter(
        [
            "def add_task(title):\n    tasks.append({'title': title})\n",
            "def search_tasks(tasks, keyword):\n    return [t for t in tasks]\n",
        ]
    )

    def fake_sync(model, prompt):
        return next(bodies)

    monkeypatch.setattr(agent_mod, "query_ollama_structured", fake_structured)
    monkeypatch.setattr(agent_mod, "query_ollama_sync", fake_sync)

    out = tmp_path / "out.py"
    result = run(chunked_code_gen(str(out), "two helpers", "python", 2))
    assert result["status"] == "written_not_validated"
    assert result["validation"]["validated"] is False
    assert any("Undefined name: tasks" in e for e in result["validation"]["errors"])
    # The file is still written -- "written, not validated" is the
    # safety net working, not a lost output.
    assert out.exists()


def test_write_file_uses_utf8(tmp_path):
    # Smart quotes and an em dash: on Windows these used to be
    # written as cp1252 bytes (0x92 / 0x97), which are not valid UTF-8.
    content = "It\u2019s a test \u2014 with smart punctuation.\n"
    out = tmp_path / "smart.txt"
    result = tool_write_file(str(out), content)
    assert result.get("status") == "success"
    raw = out.read_bytes()
    # Decode as UTF-8 (a cp1252 write would fail here: 0x92 / 0x97
    # are not valid UTF-8), then normalise newlines: Windows text
    # mode translates "\n" to "\r\n" on write, which is fine -- the
    # encoding is what this test is about, not the line endings.
    assert raw.decode("utf-8").replace("\r\n", "\n") == content
    # And reading it back through the tool returns the same text.
    read_back = tool_read_file(str(out))
    assert read_back.get("content") == content
