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

A probe run never raises and never blocks a spawn for long:
a failed probe is recorded as a failure on the card, and the
colony is returned either way. A slow model cannot hold a
spawn hostage either: the run returns within
PROBE_WALL_BUDGET_SECONDS, and jobs still running then are
reported as incomplete.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from .ollama import sync_query_ollama
from .colony_profiles import record_colony_run
from .profiles import classify_result, record_profile_run
from .reliability import MAX_MODEL_CONCURRENCY, MODEL_SEMAPHORE
from .state import Colony, prompt_for_model

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

# Correctness: every probe job has one known right answer, so
# correctness is a deterministic check, not a second model
# judging (which would cost VRAM and add its own errors). A
# checker returns True when the response gives the right answer,
# False when it gives a wrong one, and None when there is nothing
# to judge: a failed response (error / empty / UNCERTAIN) is a
# failure, not an incorrect answer, and a task type with no
# checker is not measured. Correctness is only meaningful on
# usable answers: among the answers a model gave, how many were
# right? A bare verdict ("CLEAN") fails every checker, and a
# vague dodge fails too, while a right answer passes even if it
# is phrased differently.
def _norm(text: str) -> str:
    """Lowercase with all whitespace removed, for substring checks."""
    return re.sub(r"\s+", "", (text or "").lower())


def _has_word(text: str, word: str) -> bool:
    """True when word appears as a whole word (case-insensitive)."""
    return re.search(r"\b" + re.escape(word) + r"\b", text or "", re.IGNORECASE) is not None


def check_correctness(task_type: str, response: Any) -> Any:
    """Check one probe response against the job's known answer.

    Returns True (right), False (wrong), or None (no judgment:
    the response is not a usable answer, or the task type has
    no checker). Callers pass only usable answers.
    """
    if not isinstance(response, str) or not response.strip():
        return None
    stripped = response.strip()
    # An error or a grounded refusal is not a wrong answer; it
    # is a failure, judged by classify_result, not here.
    if stripped.startswith("Error:") or stripped.startswith("UNCERTAIN"):
        return None
    if task_type == "write":
        # The context names the function "add", so the answer
        # must define add and return the sum of its two
        # arguments. Any variable names count ("a + b", "x+y",
        # "a1 + b2"), and parentheses around the sum count too,
        # but a subtraction does not, and an unrelated function
        # that merely contains a sum does not count.
        norm = _norm(response)
        return "defadd" in norm and re.search(
            r"return\(?[a-z_][a-z0-9_]*\+[a-z_][a-z0-9_]*\)?", norm
        ) is not None
    if task_type == "analyze":
        # The problem is dividing by len(nums) when nums is
        # empty. "division" or "empty" name it directly. A bare
        # "zero" does not: it must come with the cause ("len",
        # a crash, a failure, an error), or "the answer is zero"
        # would pass without identifying the problem.
        lowered = response.lower()
        if "division" in lowered or "empty" in lowered:
            return True
        if "zero" in lowered:
            return any(
                w in lowered for w in ("len", "crash", "fail", "error")
            )
        return False
    if task_type == "search":
        # Colony C has the most bugs. The word "C" is the answer,
        # and "the third colony" is the same answer in words;
        # "A" and "B" do not contain either. The count (6) is not
        # required: a bare "C" is still correct.
        return _has_word(response, "c") or "third" in response.lower()
    if task_type == "check":
        # The ollama service is up. "running" and "online" are
        # the same answer in different words.
        lowered = response.lower()
        return (
            _has_word(response, "up")
            or "running" in lowered
            or "online" in lowered
        )
    if task_type == "list":
        # The Python files are main.py and helper.py.
        lowered = response.lower()
        return "main.py" in lowered and "helper.py" in lowered
    return None


# A probe run never takes longer than this wall time to return.
# A slow model can otherwise delay colony creation for minutes
# (each call can wait up to its 120-second request timeout, and
# the pool runs several waves). Jobs still running when the
# budget is spent keep going in the background, but the spawn
# no longer waits for them.
PROBE_WALL_BUDGET_SECONDS = 180.0


def _run_one_probe(bug: Any, job: Dict[str, str]):
    """Run one probe job for one bug; report worker start time.

    The semaphore slot is acquired BEFORE the clock starts, so
    time spent waiting for a model slot is not recorded as model
    time (the same rule as Phase 1's executor queue). The call
    runs with slot_held=True: the probe already holds the slot,
    so sync_query_ollama must not acquire a second one. Returns
    the response, the worker start, and an error string if the
    call raised. sync_query_ollama normally returns an
    "Error: ..." string instead of raising; the error slot exists
    only as a safety net for a call that does raise.
    """
    full_prompt = f"Context: {job['context']}\n\nTask: {job['prompt']}"
    # Review-tuned models return a verdict ("CLEAN") for a bare
    # probe prompt; give them the do-the-task instruction, or the
    # probe records a success for garbage and the card is inflated.
    full_prompt = prompt_for_model(bug.model, full_prompt)

    try:
        with MODEL_SEMAPHORE:
            started = time.monotonic()
            response = sync_query_ollama(bug.model, full_prompt, slot_held=True)
    except Exception as e:
        return None, time.monotonic(), str(e)
    return response, started, None


def run_probes(colony: Colony) -> List[Dict[str, Any]]:
    """Run every probe job for one bug per role + model.

    Records each result on that role + model's profile card
    and returns the probe results. Never raises: a colony
    with no bugs returns an empty list, and a failed probe
    is a recorded failure, not an exception. A probe run
    returns within PROBE_WALL_BUDGET_SECONDS: jobs still
    running when the budget is spent are reported as
    incomplete (one entry carrying their count) and keep
    going in the background, but the caller is not held up
    by a slow model.
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
    # One worker per model slot (MAX_MODEL_CONCURRENCY, the same
    # size as the semaphore in reliability.py). A bigger pool only
    # parks extra workers inside sync_query_ollama waiting for a
    # semaphore slot, with their clocks already running, so their
    # recorded times include the wait -- that was the 14-110s
    # spread on the first real run. With the pool matched to the
    # semaphore, a job waits in the executor queue instead, where
    # its clock has not started yet, and the recorded time is pure
    # model time. Wall time does not change: throughput is bounded
    # by the semaphore either way.
    #
    # The executor is managed by hand, not with a `with` block:
    # on a budget timeout the pool must shut down without waiting
    # for the slow jobs, or the budget would not bound anything.
    run_started = time.monotonic()
    deadline = run_started + PROBE_WALL_BUDGET_SECONDS
    executor = ThreadPoolExecutor(
        max_workers=max(1, min(len(jobs), MAX_MODEL_CONCURRENCY))
    )
    incomplete = 0
    try:
        futures = {}
        for bug, job in jobs:
            future = executor.submit(_run_one_probe, bug, job)
            futures[future] = (bug, job)
        for future in as_completed(
            futures, timeout=max(0.0, deadline - time.monotonic())
        ):
            bug, job = futures[future]
            try:
                response, started, error = future.result()
            except Exception as e:
                response, error = None, str(e)
                started = time.monotonic()
            elapsed = time.monotonic() - started
            success, failure_kind = classify_result(response, error)
            # Correctness is only judged on a usable answer: a
            # failed response is a failure, not a wrong answer.
            correct = check_correctness(job["task_type"], response) if success else None
            # The save result is reported on the probe, not
            # swallowed: a caller that sees profile_saved False
            # knows the run was not stored.
            saved = record_profile_run(
                bug.role,
                bug.model,
                job["task_type"],
                elapsed,
                success,
                failure_kind,
                correct,
            )
            # Colony-type measurement: the same probe run feeds
            # the colony type's card, so a new colony type earns
            # trust from its probes instead of waiting for real
            # task runs. record_colony_run never raises, so
            # profiling cannot break the probe run.
            colony_saved = record_colony_run(
                colony.colony_type,
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
                "correct": correct,
                "elapsed_seconds": round(elapsed, 2),
                "started_monotonic": started,
                "profile_saved": saved,
                "colony_profile_saved": colony_saved,
            })
    except TimeoutError:
        # Budget spent: jobs still running are reported as
        # incomplete and left to finish in the background.
        # They record nothing on the cards: an unfinished job
        # is not a measured run.
        incomplete = sum(1 for f in futures if not f.done())
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    if incomplete:
        results.append({
            "bug_id": None,
            "role": None,
            "model": None,
            "task_type": None,
            "status": "incomplete",
            "failure_kind": "probe_timeout",
            "elapsed_seconds": 0.0,
            "started_monotonic": run_started,
            "profile_saved": False,
            "incomplete": incomplete,
        })
    return results


def summarize_probe_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Short summary of a probe run, for spawn return payloads.

    elapsed_seconds is the wall time of the run, not the sum of
    the per-call times: concurrent calls overlap, so adding them
    reports a duration far longer than the run took. Each result
    carries its worker start on the run's monotonic clock, and
    the wall time is the span from the first start to the last
    finish. Results without that timing info fall back to the
    sum, which is the best available for hand-built input.
    total_call_seconds keeps the cumulative per-call time for
    callers that want it.
    """
    real = [r for r in results if r["status"] != "incomplete"]
    incomplete = sum(
        r.get("incomplete", 0) for r in results if r["status"] == "incomplete"
    )
    successes = sum(1 for r in real if r["status"] == "success")
    total_call = round(sum(r["elapsed_seconds"] for r in real), 2)
    starts = [
        r["started_monotonic"] for r in real
        if r.get("started_monotonic") is not None
    ]
    ends = [
        r["started_monotonic"] + r["elapsed_seconds"] for r in real
        if r.get("started_monotonic") is not None
    ]
    wall = round(max(ends) - min(starts), 2) if starts and ends else total_call
    return {
        "probed": len(real),
        "successes": successes,
        "failures": len(real) - successes,
        "incomplete": incomplete,
        "elapsed_seconds": wall,
        "total_call_seconds": total_call,
        "profile_save_failures": sum(
            1 for r in real if r.get("profile_saved") is False
        ),
    }
