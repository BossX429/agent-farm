"""Reliability helpers for the ChatGPT medium findings (PR: medium reliability).

These helpers are deliberately small and dependency-free so they can be
used from both sync and async code without importing the MCP app:

* Parallel input limits -- a parallel tool must reject an empty list,
  a non-list, an over-long list, or an over-long payload instead of
  creating a zero-worker pool or spawning an unbounded number of
  threads.
* A shared parallel runner -- one place that caps workers, applies a
  per-task timeout, and returns results in input order. A timeout here
  means "the caller stopped waiting"; a thread already running in a
  ThreadPoolExecutor cannot be killed, so the underlying work may
  still finish in the background. Callers must not treat a timeout
  result as proof the work was cancelled.
* Aggregate output caps -- parallel results are truncated as a whole,
  not just per item, so one huge swarm cannot return an unbounded
  amount of text.
* Python code validation -- for chunked code generation, a generated
  file is *written* first and only *validated* if it passes the checks
  in :func:`validate_python_code`. Validation is a compile/AST check,
  not a test run: a validated file compiles, has no duplicate
  top-level function names, and has no failed generation fragments.
  Possible missing imports are reported as warnings, not hard errors,
  because static name analysis cannot know every name a caller will
  provide (for example names imported by the surrounding module).
"""

import asyncio
import ast
import builtins
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Parallel input limits -------------------------------------------------

MAX_PARALLEL_TASKS = 32
MAX_PARALLEL_PAYLOAD_CHARS = 100_000
MAX_AGGREGATE_OUTPUT_CHARS = 50_000

# Per-task timeouts (seconds) for the parallel runners. These bound how
# long a caller waits; they do not kill a running thread (see above).
# Model calls are allowed the longest wait because Ollama itself uses
# 120s/300s timeouts; exec/HTTP tools have their own, shorter timeouts.
MODEL_TASK_TIMEOUT_SECONDS = 330.0
EXEC_TASK_TIMEOUT_SECONDS = 70.0
HTTP_TASK_TIMEOUT_SECONDS = 40.0
FILE_TASK_TIMEOUT_SECONDS = 70.0

# How many model calls may run at once. The sync Ollama helpers in
# agent.py / ollama.py acquire this semaphore, so independent thread
# pools in different tools cannot each spawn an unbounded number of
# simultaneous model requests.
import threading

MODEL_SEMAPHORE = threading.BoundedSemaphore(4)
MAX_MODEL_CONCURRENCY = 4


def validate_parallel_payload(raw: str, name: str = "input") -> Optional[str]:
    """Return an error string if the raw parallel input is too large."""
    if raw is not None and len(raw) > MAX_PARALLEL_PAYLOAD_CHARS:
        return (
            f"{name} is too large: {len(raw)} characters, "
            f"maximum is {MAX_PARALLEL_PAYLOAD_CHARS}"
        )
    return None


def validate_parallel_items(
    items: Any, name: str = "items"
) -> Optional[str]:
    """Validate a parsed parallel list. Return an error string or None.

    An empty list is an error, not a zero-worker pool. A non-list is
    an error, not something to iterate over.
    """
    if not isinstance(items, list):
        return f"{name} must be a JSON array"
    if len(items) == 0:
        return f"{name} must not be empty"
    if len(items) > MAX_PARALLEL_TASKS:
        return (
            f"{name} has {len(items)} items, maximum is "
            f"{MAX_PARALLEL_TASKS} per request"
        )
    return None


def run_parallel(
    callables: List[Callable[[], Any]],
    max_workers: int,
    timeout: Optional[float] = None,
) -> List[Any]:
    """Run zero-arg callables in parallel, in input order.

    ``max_workers`` is capped at :data:`MAX_PARALLEL_TASKS` and at least
    1, so an empty or huge list cannot create a zero-worker or
    unbounded pool (callers should still validate with
    :func:`validate_parallel_items` first, so the user gets a clear
    error instead of a silently truncated run).

    Each callable's return value is kept in its original position. An
    exception in a callable becomes ``{"error": ..., "status": "failed"}``
    for that position. A per-task timeout becomes
    ``{"error": ..., "status": "timeout"}`` for that position; see the
    module docstring -- the thread itself is not killed.
    """
    if not callables:
        return []
    workers = max(1, min(int(max_workers), MAX_PARALLEL_TASKS))
    results: List[Any] = [None] * len(callables)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn) for fn in callables]
        for idx, future in enumerate(futures):
            try:
                results[idx] = future.result(timeout=timeout)
            except FutureTimeoutError:
                try:
                    from .action_log import log_action
                    log_action("parallel_task_timeout", "timeout",
                               f"task index {idx} timed out")
                except Exception:
                    pass
                results[idx] = {
                    "error": (
                        f"Task timed out after {timeout} seconds"
                        if timeout is not None
                        else "Task timed out"
                    ),
                    "status": "timeout",
                }
            except Exception as exc:  # noqa: BLE001 - per-task failure
                results[idx] = {"error": str(exc), "status": "failed"}
    return results


def cap_aggregate_output(
    results: List[Dict[str, Any]], fields: Tuple[str, ...]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Truncate parallel results so their total text stays bounded.

    Returns ``(results, truncated)``. Truncation walks the results in
    order and cuts the listed fields of later results first, so the
    earliest results stay intact. A result whose field was cut gets
    ``"truncated": True`` added.
    """
    total = 0
    for result in results:
        if isinstance(result, dict):
            for field in fields:
                value = result.get(field)
                if isinstance(value, str):
                    total += len(value)
    if total <= MAX_AGGREGATE_OUTPUT_CHARS:
        return results, False
    budget = MAX_AGGREGATE_OUTPUT_CHARS
    for result in results:
        if not isinstance(result, dict):
            continue
        for field in fields:
            value = result.get(field)
            if not isinstance(value, str):
                continue
            if budget <= 0:
                result[field] = ""
                result["truncated"] = True
            elif len(value) > budget:
                result[field] = value[:budget]
                result["truncated"] = True
                budget = 0
            else:
                budget -= len(value)
    return results, True


# --- Tracked async tasks ---------------------------------------------------
#
# Audit note (Manus second review): the review claimed an internal
# async function used fire-and-forget asyncio.create_task() without
# storing or awaiting it. A source audit of main at PR #8 found NO
# asyncio.create_task() call anywhere in agent_farm/ -- async tools
# use asyncio.to_thread() and await it. There was therefore no live
# fire-and-forget bug to fix. This helper exists so future code has
# one safe pattern, and the regression test in
# tests/test_manus_adversarial.py fails if a bare create_task()
# appears in agent_farm/ without being tracked here.
#
# A tracked task is always stored in a set (so it cannot be garbage
# collected before it finishes) and its exception is recorded if
# nobody awaits it. gather_tracked() awaits every task and lets the
# first exception surface to the caller instead of swallowing it.

_TRACKED_TASKS = set()
TRACKED_TASK_ERRORS = []


def create_tracked_task(coro, name=None):
    """Create an asyncio task that is stored until it finishes.

    If the task fails and nobody awaited it, the exception is
    appended to TRACKED_TASK_ERRORS instead of being silently
    dropped (the "exception was never retrieved" case that
    fire-and-forget create_task() hides).
    """
    task = asyncio.create_task(coro)
    if name:
        try:
            task.set_name(name)
        except Exception:
            pass
    _TRACKED_TASKS.add(task)

    def _done(t):
        _TRACKED_TASKS.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                # Recorded so a failure is never silently dropped.
                # gather_tracked() clears this list after it surfaces
                # the exception to its caller, so a cleared list means
                # the failure was seen; a non-empty list means a
                # tracked task failed without being gathered.
                TRACKED_TASK_ERRORS.append(exc)

    task.add_done_callback(_done)
    return task


async def gather_tracked(coros):
    """Await coroutines via tracked tasks; first exception surfaces.

    Unlike fire-and-forget, no task is left running unobserved and
    no exception is swallowed: asyncio.gather() propagates the
    first exception to this caller.
    """
    tasks = [create_tracked_task(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    finally:
        # Exceptions surfaced here were seen by the caller, so drop
        # matching records to avoid double-reporting them as
        # "unretrieved" from the done callback.
        TRACKED_TASK_ERRORS.clear()


# --- Python code validation -------------------------------------------------

_GENERATION_FAILED_RE = re.compile(r"generation failed", re.IGNORECASE)
_BUILTINS = set(dir(builtins))


def _collect_defined_names(tree: ast.AST) -> set:
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    return defined


def _collect_used_root_names(tree: ast.AST) -> set:
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Walk down to the root name: json.dumps -> "json".
            value = node.value
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                used.add(value.id)
    return used


def validate_python_code(content: str) -> Dict[str, Any]:
    """Validate generated Python code without running it.

    Returns a dict with:
    - ``syntax_ok``: False if the code does not parse/compile.
    - ``errors``: hard problems (syntax errors, duplicate top-level
      function names, failed generation fragments).
    - ``warnings``: possible missing imports (heuristic, see module
      docstring) and other non-fatal notes.
    - ``validated``: True only when there are no errors.
    - ``function_names`` / ``function_count``: top-level functions found.

    Validation is deliberately *not* execution: generated code is
    never run as part of validation, so a validated file is known to
    compile, not known to be correct or tested.
    """
    if not isinstance(content, str) or not content.strip():
        return {
            "syntax_ok": False,
            "errors": ["Generated code is empty"],
            "warnings": [],
            "validated": False,
            "function_names": [],
            "function_count": 0,
        }
    errors: List[str] = []
    warnings: List[str] = []
    try:
        tree = ast.parse(content)
        compile(content, "<generated>", "exec")
        syntax_ok = True
    except SyntaxError as exc:
        return {
            "syntax_ok": False,
            "errors": [f"Syntax error at line {exc.lineno}: {exc.msg}"],
            "warnings": [],
            "validated": False,
            "function_names": [],
            "function_count": 0,
        }

    function_names: List[str] = []
    seen = set()
    duplicates = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
            if node.name in seen:
                duplicates.add(node.name)
            seen.add(node.name)
    for name in sorted(duplicates):
        errors.append(f"Duplicate function name: {name}")
    if _GENERATION_FAILED_RE.search(content):
        errors.append("One or more functions failed to generate")

    defined = _collect_defined_names(tree)
    used = _collect_used_root_names(tree)
    # Names a generated module can reasonably rely on without importing:
    # builtins, names it defines itself, and the conventional ``self``/``cls``.
    possible_missing = sorted(
        name
        for name in used - defined - _BUILTINS - {"self", "cls"}
        if name not in function_names
    )
    # Only flag names that look like they were meant as modules or
    # known stdlib helpers used as a root (e.g. json/os/sys/re) -- a
    # bare name like ``helper`` could be supplied by the caller, so do
    # not flag arbitrary unknown names as missing imports.
    _KNOWN_MODULE_ROOTS = {
        "json", "os", "sys", "re", "math", "random", "datetime",
        "time", "pathlib", "collections", "itertools", "functools",
        "typing", "io", "subprocess", "httpx", "requests",
    }
    for name in possible_missing:
        if name in _KNOWN_MODULE_ROOTS:
            warnings.append(f"Possible missing import: {name}")

    return {
        "syntax_ok": syntax_ok,
        "errors": errors,
        "warnings": warnings,
        "validated": len(errors) == 0,
        "function_names": function_names,
        "function_count": len(function_names),
    }


def _is_failed_fragment(text: str) -> bool:
    """True if a generated chunk is a failure marker, not content."""
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    return (
        not stripped
        or stripped.startswith("[Section ")
        and "generation failed" in stripped
        or stripped.startswith("# Function ")
        and "generation failed" in stripped
    )


def validate_document_sections(planned_titles, contents) -> Dict[str, Any]:
    """Validate a chunked_write result without running or judging it.

    Checks only structural completion: the planned number of sections
    were generated, none are empty, and none are failure markers.
    This does NOT judge quality, accuracy, or whether the content
    matches the spec -- a validated document is complete, not correct.
    """
    errors = []
    warnings = []
    planned = len(planned_titles or [])
    generated = len(contents or [])
    if generated != planned:
        errors.append(f"Expected {planned} sections, got {generated}")
    empty = 0
    failed = 0
    for idx, content in enumerate(contents or []):
        if not isinstance(content, str) or not content.strip():
            empty += 1
            errors.append(f"Section {idx + 1} is empty")
        elif _is_failed_fragment(content):
            failed += 1
            errors.append(f"Section {idx + 1} generation failed")
        elif len(content.strip()) < 20:
            warnings.append(f"Section {idx + 1} is very short ({len(content.strip())} chars)")
    return {
        "validated": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "sections_planned": planned,
        "sections_generated": generated,
        "sections_empty": empty,
        "sections_failed": failed,
    }


def validate_analyses(planned_count: int, analyses) -> Dict[str, Any]:
    """Validate a chunked_analysis result, structurally only.

    Checks the planned number of perspectives were returned, none are
    empty, and none are the 'Error' placeholder. Like document
    validation, this does NOT judge quality or correctness.
    """
    errors = []
    warnings = []
    analyses = analyses or []
    if len(analyses) != planned_count:
        errors.append(f"Expected {planned_count} perspectives, got {len(analyses)}")
    empty = 0
    failed = 0
    for idx, item in enumerate(analyses):
        text = item.get("analysis", "") if isinstance(item, dict) else ""
        if item.get("perspective") == "Error" if isinstance(item, dict) else False:
            failed += 1
            errors.append(f"Perspective {idx + 1} analysis failed")
        elif not isinstance(text, str) or not text.strip():
            empty += 1
            errors.append(f"Perspective {idx + 1} analysis is empty")
        elif len(text.strip()) < 20:
            warnings.append(f"Perspective {idx + 1} analysis is very short")
    return {
        "validated": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "perspectives_planned": planned_count,
        "perspectives_generated": len(analyses),
        "perspectives_empty": empty,
        "perspectives_failed": failed,
    }
