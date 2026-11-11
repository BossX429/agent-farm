"""Colony core: bug/colony creation, routing, and parallel execution."""

import os
import time
import uuid
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from .state import (
    MODEL_TIERS,
    Bug,
    Colony,
    SharedConsciousness,
    NanoArmor,
    colonies,
    stats,
)
from .ollama import sync_query_ollama
from .metrics import record_bug_run
from .profiles import (
    TASK_TYPE_ROLES,
    classify_result,
    profile_pick,
    record_profile_run,
    task_type_for_prompt,
)

def _unique_id(prefix: str) -> str:
    """Unique, human-readable ID.

    The old code used second-resolution timestamps, so two colonies spawned
    in the same second -- or one colony with duplicate roles (two workers,
    four guardians, ...) -- produced identical IDs and overwrote / confused
    each other. uuid4 makes collisions effectively impossible.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def create_bug(colony_id: str, role: str) -> Bug:
    """Create a bug with appropriate model for role"""
    model = MODEL_TIERS.get(role, MODEL_TIERS["worker"])
    bug_id = _unique_id(f"{colony_id}-{role}")
    return Bug(id=bug_id, role=role, model=model)


def spawn_colony_internal(colony_type: str = "standard") -> Colony:
    """Spawn a new colony with bugs based on type"""
    colony_id = _unique_id("colony")
    
    if colony_type == "fast":
        # Speed-focused: more scouts, fewer heavy bugs
        roles = ["scout", "scout", "worker", "worker"]
    elif colony_type == "heavy":
        # Power-focused: reasoning and memory bugs
        roles = ["memory", "memory", "learner", "guardian"]
    elif colony_type == "hybrid":
        # Balanced: mix of all roles
        roles = ["scout", "worker", "memory", "guardian", "learner"]
    else:
        # Standard: typical work colony
        roles = ["scout", "worker", "worker", "memory"]
    
    bugs = [create_bug(colony_id, role) for role in roles]
    colony = Colony(id=colony_id, bugs=bugs)
    colonies[colony_id] = colony
    stats["colonies_spawned"] += 1
    
    return colony


def get_best_bug_for_task(task: Dict, colony: Colony) -> Bug:
    """Smart routing - match task to appropriate bug role.

    Measured profile evidence comes first: a trusted card for
    the task type picks the bug. Otherwise the shared keyword
    table (profiles.TASK_TYPE_KEYWORDS) routes by role, so the
    router and the profiler always agree on what a prompt is.
    The final fallback is the first worker, then any bug.
    """
    task_type = task_type_for_prompt(task.get("prompt", ""))
    picked = profile_pick(task_type, colony.bugs)
    if picked is not None:
        return picked
    role = TASK_TYPE_ROLES.get(task_type)
    if role:
        bugs = [b for b in colony.bugs if b.role == role]
        if bugs:
            return bugs[0]
    # Default: return first available worker, then any bug
    workers = [b for b in colony.bugs if b.role == "worker"]
    if workers:
        return workers[0]
    return colony.bugs[0]


def _run_bug_query(bug: Bug, full_prompt: str) -> Tuple[str, float]:
    """Run one bug's model call; report when the worker started it.

    The start time is captured inside the worker thread, so time a
    task spends queued in the executor is not counted as model time.
    The model semaphore wait inside sync_query_ollama is still
    included: that wait is part of the call itself. Uses a monotonic
    clock, so system clock changes cannot corrupt the duration.
    """
    started = time.monotonic()
    response = sync_query_ollama(bug.model, full_prompt)
    return response, started


def execute_parallel_swarm(tasks: List[Dict[str, str]], colony: Colony) -> List[Dict]:
    """Execute tasks in TRUE PARALLEL using ThreadPoolExecutor"""
    start = time.time()
    results = []
    # No tasks, or no bugs to run them: return without creating a
    # pool at all (ThreadPoolExecutor(max_workers=0) would raise).
    if not tasks or not colony.bugs:
        return results
    
    # Smart routing - match tasks to appropriate bugs.
    # Whether a pick was profile-driven is captured HERE, at
    # assignment time, and carried into the result. It is the
    # router's decision, so it must be read when the router
    # decides: re-reading the card after the run would let a
    # card change mid-call rewrite a fallback as profiled.
    # The pick counts as profiled only when the bug that
    # finally runs the task IS the one the profile picked. A
    # load-spread swap to another bug, even one with its own
    # trusted card, is a fallback, not evidence.
    bug_assignments = []
    used_bugs = set()
    for task in tasks:
        task_type = task_type_for_prompt(task.get("prompt", ""))
        profiled_bug = profile_pick(task_type, colony.bugs)
        bug = get_best_bug_for_task(task, colony)
        # Try to spread load if same bug keeps getting picked
        if bug.id in used_bugs and len(colony.bugs) > 1:
            # Find another bug of same role or similar
            alternatives = [b for b in colony.bugs if b.id not in used_bugs]
            if alternatives:
                # Prefer same role
                same_role = [b for b in alternatives if b.role == bug.role]
                bug = same_role[0] if same_role else alternatives[0]
        used_bugs.add(bug.id)
        profiled = profiled_bug is not None and bug.id == profiled_bug.id
        bug_assignments.append((bug, task, profiled))
    
    # Execute in parallel (workers capped, never zero -- see guard above)
    with ThreadPoolExecutor(max_workers=max(1, min(len(colony.bugs), 8))) as executor:
        futures = {}
        for bug, task, profiled in bug_assignments:
            prompt = task.get("prompt", "")
            context = task.get("context", "")
            full_prompt = f"Context: {context}\n\nTask: {prompt}" if context else prompt
            future = executor.submit(_run_bug_query, bug, full_prompt)
            futures[future] = (bug, task, profiled, time.monotonic())

        for future in as_completed(futures):
            bug, task, profiled, submitted = futures[future]
            response = None
            error = None
            try:
                response, started = future.result()
            except Exception as e:
                # The model call itself raised. sync_query_ollama
                # normally returns an "Error: ..." string instead, so
                # this path is the exception, not the rule. No worker
                # start was recorded, so fall back to the submit time.
                error = str(e)
                started = submitted
            # Profiler factory Phase 1: per-bug elapsed time and status.
            # This only records what the run already knows; it does not
            # change which bug was picked or what the bug returns.
            # The unrounded duration is what gets recorded; only the
            # result dict rounds it, so short runs are not lost.
            elapsed = time.monotonic() - started
            elapsed_seconds = round(elapsed, 2)
            if error is None and isinstance(response, str) and response.strip().startswith("Error:"):
                # sync_query_ollama returns an "Error: ..." string on
                # a failed model call instead of raising, so a bare
                # success check would count a failed call as a success.
                # Same "Error:" convention as
                # reliability._is_failed_fragment.
                error = response
                response = None
            # The "profiled" flag was captured at assignment time
            # and carried here; it is about the routing decision,
            # not the outcome, so it goes on success and failure
            # results alike. task_type is still needed below to
            # record this run on the bug's card.
            task_type = task_type_for_prompt(task.get("prompt", ""))
            if error is None:
                result = {
                    "bug_id": bug.id,
                    "role": bug.role,
                    "model": bug.model,
                    "task": task.get("prompt", "")[:50],
                    "response": response,
                    "status": "success",
                    "elapsed_seconds": elapsed_seconds,
                    "profiled": profiled,
                }
                results.append(result)
                record_bug_run(bug.id, elapsed, True, role=bug.role, model=bug.model)
                bug.status = "completed"
                bug.last_task = task.get("prompt", "")[:50]
            else:
                result = {
                    "bug_id": bug.id,
                    "role": bug.role,
                    "model": bug.model,
                    "error": error,
                    "status": "failed",
                    "elapsed_seconds": elapsed_seconds,
                    "profiled": profiled,
                }
                results.append(result)
                record_bug_run(bug.id, elapsed, False, role=bug.role, model=bug.model)
            # Profiler factory Phase 2: feed the run into the bug's
            # profile card. A card counts a run as a success only
            # when it produced a usable answer, so this uses
            # classify_result, not the call outcome above: a call
            # that completed but answered UNCERTAIN counts as a
            # success in the per-bug metrics and as a failure of
            # kind "uncertain" on the card. record_profile_run
            # never raises, so profiling cannot break the swarm.
            # Its return is reported on the result, not swallowed:
            # a caller that sees profile_saved False knows the run
            # was not stored.
            card_success, failure_kind = classify_result(response, error)
            result["profile_saved"] = record_profile_run(
                bug.role,
                bug.model,
                task_type,
                elapsed,
                card_success,
                failure_kind,
            )

    elapsed = time.time() - start
    colony.tasks_completed += len(tasks)
    stats["tasks_executed"] += len(tasks)
    stats["swarm_runs"] += 1

    # No speedup is reported here: deriving "sequential time" from elapsed
    # time would be circular (see the stats note above). Callers that need
    # timing should use elapsed_seconds, not an invented speedup ratio.
    return results


def entangle_colony(colony: Colony) -> Dict:
    """Bond all bugs via SharedConsciousness"""
    # Collect discoveries from all bugs
    for bug in colony.bugs:
        colony.consciousness.discoveries.extend(bug.discoveries)
    
    return {
        "colony_id": colony.id,
        "bugs_entangled": len(colony.bugs),
        "shared_discoveries": len(colony.consciousness.discoveries),
        "shared_patterns": len(colony.consciousness.patterns)
    }
