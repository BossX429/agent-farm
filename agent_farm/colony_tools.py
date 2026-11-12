"""MCP tools that manage colonies."""

import os
import uuid
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import probes
from .mcp_app import mcp
from .metrics import track_tool_run
from .colony_profiles import (
    colony_profiles_summary,
    prune_stale_colony_profiles,
    resolve_colony_type,
)
from .profiles import profiles_summary, prune_stale_profiles
from .state import MODEL_TIERS, OLLAMA_URL, colonies, stats
from .colony import (
    spawn_colony_internal,
    execute_parallel_swarm,
    entangle_colony,
)
from .ollama import query_ollama, sync_query_ollama, sync_query_raw, synthesize_swarm_results
from .reliability import validate_parallel_items, validate_parallel_payload


# ============ MCP TOOLS ============

async def _probe_summary_for(colony) -> dict:
    """Run the probe harness for a colony and summarize it.

    A failed probe run returns an error summary, never a failed
    spawn. If the caller is cancelled while probes are running,
    the colony is removed from the registry before the
    cancellation propagates: a colony nobody can reach by id is
    an orphan, and its probe threads finish in the background
    on their own.
    """
    try:
        probe_results = await asyncio.to_thread(probes.run_probes, colony)
        return probes.summarize_probe_results(probe_results)
    except asyncio.CancelledError:
        colonies.pop(colony.id, None)
        raise
    except Exception:
        return {
            "probed": 0,
            "successes": 0,
            "failures": 0,
            "incomplete": 0,
            "elapsed_seconds": 0.0,
            "total_call_seconds": 0.0,
            "profile_save_failures": 0,
            "error": "probe run failed",
        }


@mcp.tool()
@track_tool_run()
async def spawn_colony(colony_type: str = "standard", run_probes: bool = True) -> dict:
    """
    Spawn a new bug colony. Types: standard, fast, heavy, hybrid
    - standard: scout, worker, worker, memory
    - fast: scout, scout, worker, worker (speed)
    - heavy: memory, memory, learner, guardian (power)
    - hybrid: all roles balanced

    Profiler factory Phase 2: unless run_probes is False, the
    probe harness runs automatically after the spawn, one
    representative bug per role + model, and the results land
    on the profile cards in profiles/. Probing costs extra
    model calls and time before the colony can do real work,
    and a probe run that fails (Ollama down, for example) is
    recorded as failed probes, never as a failed spawn.
    """
    colony = spawn_colony_internal(colony_type)
    # Stale profile cards (a legacy duplicate, or a role + model
    # combo no bug can use anymore) are removed on every spawn,
    # so they can never accumulate. The prune never raises.
    prune_stale_profiles()
    prune_stale_colony_profiles()
    probe_summary = None
    if run_probes:
        probe_summary = await _probe_summary_for(colony)
    return {
        "colony_id": colony.id,
        "type": colony_type,
        "bugs": [{"id": b.id, "role": b.role, "model": b.model} for b in colony.bugs],
        "created": colony.created,
        "probes": probe_summary,
    }


@mcp.tool()
@track_tool_run()
async def deploy_swarm(colony_id: str, tasks: str) -> dict:
    """
    Deploy swarm for parallel task execution.
    tasks: JSON array of {prompt, context?} objects
    Example: [{"prompt": "analyze X"}, {"prompt": "summarize Y", "context": "..."}]
    """
    if colony_id not in colonies:
        return {"error": f"Colony {colony_id} not found"}
    
    colony = colonies[colony_id]

    payload_error = validate_parallel_payload(tasks, "tasks")
    if payload_error:
        return {"error": payload_error}
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError:
        # Single task as string
        task_list = [{"prompt": tasks}]

    items_error = validate_parallel_items(task_list, "tasks")
    if items_error:
        return {"error": items_error}

    results = await asyncio.to_thread(execute_parallel_swarm, task_list, colony)
    
    return {
        "colony_id": colony_id,
        "tasks_executed": len(task_list),
        "bugs_used": len(colony.bugs),
        "results": results,
    }


@mcp.tool()
@track_tool_run()
async def quick_swarm(tasks: str, colony_type: str = "auto", run_probes: bool = True) -> dict:
    """
    One-shot: spawn colony + deploy swarm + return results
    tasks: JSON array or single prompt string

    Profiler factory Phase 2: unless run_probes is False, the
    probe harness runs automatically after the spawn and before
    the tasks, and the results land on the profile cards in
    profiles/. Probing costs extra model calls and time before
    the tasks run.

    Colony-type measurement (Phase 3): colony_type defaults to
    "auto", which picks the best colony type for the tasks from
    trusted colony-type cards and falls back to "fast" when no
    card qualifies, so behaviour is unchanged until cards are
    trusted. An explicit type is always used as-is.
    """
    payload_error = validate_parallel_payload(tasks, "tasks")
    if payload_error:
        return {"error": payload_error}
    try:
        task_list = json.loads(tasks)
    except (json.JSONDecodeError, TypeError):
        task_list = [{"prompt": tasks}]
    colony_type, colony_type_profiled = resolve_colony_type(
        colony_type, task_list, "fast"
    )
    colony = spawn_colony_internal(colony_type)
    # Stale profile cards are removed on every spawn, so they
    # can never accumulate. The prune never raises.
    prune_stale_profiles()
    prune_stale_colony_profiles()

    items_error = validate_parallel_items(task_list, "tasks")
    if items_error:
        return {"error": items_error}

    probe_summary = None
    if run_probes:
        probe_summary = await _probe_summary_for(colony)

    results = await asyncio.to_thread(execute_parallel_swarm, task_list, colony)
    
    return {
        "colony_id": colony.id,
        "colony_type": colony_type,
        "colony_type_profiled": colony_type_profiled,
        "tasks": len(task_list),
        "results": results,
        "probes": probe_summary,
    }


@mcp.tool()
async def list_colonies() -> dict:
    """List all active colonies with their bugs"""
    return {
        "colonies": [
            {
                "id": c.id,
                "bugs": len(c.bugs),
                "tasks_completed": c.tasks_completed,
                "created": c.created,
                "roles": [b.role for b in c.bugs]
            }
            for c in colonies.values()
        ],
        "total": len(colonies)
    }


@mcp.tool()
async def colony_status(colony_id: str) -> dict:
    """Get detailed status of a specific colony"""
    if colony_id not in colonies:
        return {"error": f"Colony {colony_id} not found"}
    
    colony = colonies[colony_id]
    return {
        "colony_id": colony.id,
        "bugs": [
            {
                "id": b.id,
                "role": b.role,
                "model": b.model,
                "status": b.status,
                "last_task": b.last_task,
                "discoveries": len(b.discoveries)
            }
            for b in colony.bugs
        ],
        "consciousness": {
            "discoveries": len(colony.consciousness.discoveries),
            "patterns": len(colony.consciousness.patterns),
            "threats": len(colony.consciousness.threats),
            "rag_cached": len(colony.consciousness.rag_cache)
        },
        "armor": {
            "active": colony.armor.shield,
            "attacks_blocked": colony.armor.attacks_blocked
        },
        "tasks_completed": colony.tasks_completed
    }


@mcp.tool()
@track_tool_run()
async def quick_colony() -> dict:
    """
    Quick status check - ONE CALL for colony health
    Returns: verdict, issues, active colonies, stats
    """
    issues = []
    
    # Check for idle colonies
    idle_colonies = [c.id for c in colonies.values() if c.tasks_completed == 0]
    if len(idle_colonies) > 3:
        issues.append(f"{len(idle_colonies)} idle colonies - consider cleanup")
    
    # Check model availability
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                issues.append("Ollama not responding")
    except:
        issues.append("Ollama connection failed")
    
    return {
        "verdict": "healthy" if not issues else "issues_found",
        "issues": issues,
        "active_colonies": len(colonies),
        "total_spawned": stats["colonies_spawned"],
        "tasks_executed": stats["tasks_executed"],
        "swarm_runs": stats["swarm_runs"],
        # No speedup field: none is measured (see stats note above).
        "models": MODEL_TIERS
    }


def _action_log_summary() -> dict:
    """Observability summary for the opt-in action log (Copilot #4).

    Returns only counts and the last action/outcome -- never log
    content, which action_log already redacts. If logging is off,
    this reports enabled=False with no records, rather than failing.
    """
    try:
        from .action_log import _log_path
    except Exception:
        return {"enabled": False, "records": 0, "last_action": None, "last_outcome": None}
    path = _log_path()
    if not path:
        return {"enabled": False, "records": 0, "last_action": None, "last_outcome": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except FileNotFoundError:
        return {"enabled": True, "records": 0, "last_action": None, "last_outcome": None}
    except Exception:
        return {"enabled": True, "records": 0, "last_action": None, "last_outcome": None}
    last = None
    if lines:
        try:
            last = json.loads(lines[-1])
        except Exception:
            last = None
    return {
        "enabled": True,
        "records": len(lines),
        "last_action": last.get("action") if last else None,
        "last_outcome": last.get("outcome") if last else None,
        "last_ts": last.get("ts") if last else None,
    }


@mcp.tool()
async def farm_stats() -> dict:
    """Get comprehensive farm statistics"""
    from .metrics import get_metrics

    return {
        "colonies_spawned": stats["colonies_spawned"],
        "active_colonies": len(colonies),
        "tasks_executed": stats["tasks_executed"],
        "discoveries": stats["discoveries"],
        "swarm_runs": stats["swarm_runs"],
        # No speedup field: none is measured (see stats note above).
        # No token-cost field: Ollama responses here do not report
        # token counts uniformly, so a cost number would be invented.
        "model_tiers": MODEL_TIERS,
        # Copilot roadmap #3 (metrics) and #4 (observability):
        # per-tool counts/durations/validation outcomes, plus the
        # action-log summary. Metrics now cover chunked tools (with
        # validation outcomes) and swarm/colony tools (via the
        # track_tool_run decorator, counts and durations only).
        # Read-only tools (list_colonies, colony_status, farm_stats
        # itself) are not counted, so "metrics" is a tool-run record,
        # not a count of every MCP call.
        "metrics": get_metrics(),
        "metrics_coverage": (
            "chunked tools with validation outcomes; "
            "swarm and colony tools with counts/durations only; "
            "read-only tools not counted"
        ),
        # Profiler factory Phase 2: one summary line per stored
        # profile card (role, model, samples, trusted, success
        # rate, avg seconds). Full cards live in profiles/.
        "profiles": profiles_summary(),
        # Colony-type measurement (Phase 3): one summary line
        # per stored colony-type card. Full cards live in
        # colony_profiles/.
        "colony_profiles": colony_profiles_summary(),
        "observability": {
            "action_log": _action_log_summary(),
        },
    }


@mcp.tool()
@track_tool_run()
async def dissolve_colony(colony_id: str) -> dict:
    """Dissolve a colony and free resources"""
    if colony_id not in colonies:
        return {"error": f"Colony {colony_id} not found"}
    
    colony = colonies.pop(colony_id)
    return {
        "dissolved": colony_id,
        "bugs_released": len(colony.bugs),
        "tasks_completed": colony.tasks_completed,
        "remaining_colonies": len(colonies)
    }


@mcp.tool()
@track_tool_run()
async def cleanup_idle() -> dict:
    """Remove all colonies with 0 tasks completed"""
    idle = [cid for cid, c in colonies.items() if c.tasks_completed == 0]
    for cid in idle:
        colonies.pop(cid)
    
    return {
        "dissolved": len(idle),
        "colony_ids": idle,
        "remaining": len(colonies)
    }
