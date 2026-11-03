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
from .mcp_app import mcp
from .state import MODEL_TIERS, OLLAMA_URL, colonies, stats
from .colony import (
    spawn_colony_internal,
    execute_parallel_swarm,
    entangle_colony,
)
from .ollama import query_ollama, sync_query_ollama, sync_query_raw, synthesize_swarm_results


# ============ MCP TOOLS ============

@mcp.tool()
async def spawn_colony(colony_type: str = "standard") -> dict:
    """
    Spawn a new bug colony. Types: standard, fast, heavy, hybrid
    - standard: scout, worker, worker, memory
    - fast: scout, scout, worker, worker (speed)
    - heavy: memory, memory, learner, guardian (power)
    - hybrid: all roles balanced
    """
    colony = spawn_colony_internal(colony_type)
    return {
        "colony_id": colony.id,
        "type": colony_type,
        "bugs": [{"id": b.id, "role": b.role, "model": b.model} for b in colony.bugs],
        "created": colony.created
    }


@mcp.tool()
async def deploy_swarm(colony_id: str, tasks: str) -> dict:
    """
    Deploy swarm for parallel task execution.
    tasks: JSON array of {prompt, context?} objects
    Example: [{"prompt": "analyze X"}, {"prompt": "summarize Y", "context": "..."}]
    """
    if colony_id not in colonies:
        return {"error": f"Colony {colony_id} not found"}
    
    colony = colonies[colony_id]
    
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError:
        # Single task as string
        task_list = [{"prompt": tasks}]
    
    results = execute_parallel_swarm(task_list, colony)
    
    return {
        "colony_id": colony_id,
        "tasks_executed": len(task_list),
        "bugs_used": len(colony.bugs),
        "results": results,
    }


@mcp.tool()
async def quick_swarm(tasks: str, colony_type: str = "fast") -> dict:
    """
    One-shot: spawn colony + deploy swarm + return results
    tasks: JSON array or single prompt string
    """
    colony = spawn_colony_internal(colony_type)
    
    try:
        task_list = json.loads(tasks)
    except (json.JSONDecodeError, TypeError):
        task_list = [{"prompt": tasks}]
    
    results = execute_parallel_swarm(task_list, colony)
    
    return {
        "colony_id": colony.id,
        "tasks": len(task_list),
        "results": results
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


@mcp.tool()
async def farm_stats() -> dict:
    """Get comprehensive farm statistics"""
    return {
        "colonies_spawned": stats["colonies_spawned"],
        "active_colonies": len(colonies),
        "tasks_executed": stats["tasks_executed"],
        "discoveries": stats["discoveries"],
        "swarm_runs": stats["swarm_runs"],
        # No speedup field: none is measured (see stats note above).
        "model_tiers": MODEL_TIERS
    }


@mcp.tool()
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
