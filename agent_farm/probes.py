"""Probe harness for the agent profiler factory (Phase 2).

A probe is a small fixed job, one per task type the router
already knows. Every bug runs the same jobs, so results are
comparable. Each job carries its own context: the grounded
model answers ONLY from given context, so a probe with no
context can only ever answer UNCERTAIN, which measures
nothing. The jobs are deliberately trivial -- probes measure
speed and reliability, not intelligence.

Probes run automatically when a colony spawns (locked
2026-11-10). One representative bug per unique role + model
is probed: cards are per role + model, so a second bug of
the same combo would only add sampling noise at double the
spawn cost. Extra bugs of a combo still feed their card
through real swarm runs.

A probe run never raises and never blocks a spawn: a failed
probe is recorded as a failure on the card, and the colony
is returned either way.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from .ollama import sync_query_ollama
from .profiles import classify_result, record_profile_run
from .state import Colony

# One job per task type. Each job's context is the whole job:
# the prompt asks for something answerable from the context
# alone, so a grounded call can produce a real answer.
PROBE_JOBS = [
    {
        "task_type": "write",
        "context": "A Python function named add takes two numbers, a and b, and returns their sum.",
        "prompt": "Write the function described in the context.",
    },
    {
        "task_type": "analyze",
        "context": "def average(nums):\n    return sum(nums) / len(nums)",
        "prompt": "Name one problem in the code in the context.",
    },
    {
        "task_type": "search",
        "context": "The farm has three colonies. Colony A has 4 bugs, colony B has 2 bugs, colony C has 6 bugs.",
        "prompt": "From the context, which colony has the most bugs?",
    },
    {
        "task_type": "check",
        "context": "Service status: ollama up, disk 62 percent full, memory 41 percent used.",
        "prompt": "From the context, is the ollama service up or down?",
    },
    {
        "task_type": "list",
        "context": "Files in the folder: notes.txt, main.py, helper.py, README.md.",
        "prompt": "From the context, list the Python files.",
    },
]


def _run_one_probe(bug: Any, job: Dict[str, str]):
    """Run one probe job for one bug; report worker start time.

    The start time is captured inside the worker thread, so
    time spent queued in the executor is not counted as model
    time (same rule as Phase 1). Returns the response, the
    worker start, and an error string if the call raised.
    sync_query_ollama normally returns an "Error: ..." string
    instead of raising; the error slot exists only as a safety
    net for a call that does raise.
    """
    started = time.monotonic()
    full_prompt = f"Context: {job['context']}\n\nTask: {job['prompt']}"
    try:
        response = sync_query_ollama(bug.model, full_prompt)
        return response, started, None
    except Exception as e:
        return None, started, str(e)


def run_probes(colony: Colony) -> List[Dict[str, Any]]:
    """Run every probe job for one bug per role + model.

    Records each result on that role + model's profile card
    and returns the probe results. Never raises: a colony
    with no bugs returns an empty list, and a failed probe
    is a recorded failure, not an exception.
    """
    try:
        bugs = colony.bugs or []
    except Exception:
        return []
    representatives = []
    seen = set()
    for bug in bugs:
        key = (bug.role, bug.model)
        if key not in seen:
            seen.add(key)
            representatives.append(bug)
    if not representatives:
        return []

    jobs = [(bug, job) for bug in representatives for job in PROBE_JOBS]
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), 8))) as executor:
        futures = {}
        for bug, job in jobs:
            future = executor.submit(_run_one_probe, bug, job)
            futures[future] = (bug, job)
        for future in as_completed(futures):
            bug, job = futures[future]
            try:
                response, started, error = future.result()
            except Exception as e:
                response, error = None, str(e)
                started = time.monotonic()
            elapsed = time.monotonic() - started
            success, failure_kind = classify_result(response, error)
            record_profile_run(
                bug.role,
                bug.model,
                job["task_type"],
                elapsed,
                success,
                failure_kind,
            )
            results.append({
                "bug_id": bug.id,
                "role": bug.role,
                "model": bug.model,
                "task_type": job["task_type"],
                "status": "success" if success else "failed",
                "failure_kind": failure_kind,
                "elapsed_seconds": round(elapsed, 2),
            })
    return results


def summarize_probe_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Short summary of a probe run, for spawn return payloads."""
    successes = sum(1 for r in results if r["status"] == "success")
    return {
        "probed": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "elapsed_seconds": round(
            sum(r["elapsed_seconds"] for r in results), 2
        ),
    }
