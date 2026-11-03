"""Colony core: bug/colony creation, routing, and parallel execution."""

import os
import uuid
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
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
    """Smart routing - match task to appropriate bug role"""
    prompt = task.get("prompt", "").lower()
    
    # Write operations need workers
    write_keywords = ["write", "create file", "save to", "output to", "append to"]
    if any(kw in prompt for kw in write_keywords):
        workers = [b for b in colony.bugs if b.role == "worker"]
        if workers:
            return workers[0]
    
    # HTTP operations need workers
    http_keywords = ["http", "api", "fetch", "request", "post to", "get from"]
    if any(kw in prompt for kw in http_keywords):
        workers = [b for b in colony.bugs if b.role == "worker"]
        if workers:
            return workers[0]
    
    # Knowledge base queries need memory bugs
    kb_keywords = ["kmkb", "knowledge", "search knowledge", "ask knowledge"]
    if any(kw in prompt for kw in kb_keywords):
        memory = [b for b in colony.bugs if b.role == "memory"]
        if memory:
            return memory[0]
    
    # Code analysis needs learners
    analysis_keywords = ["analyze code", "review code", "code quality"]
    if any(kw in prompt for kw in analysis_keywords):
        learners = [b for b in colony.bugs if b.role == "learner"]
        if learners:
            return learners[0]
    
    # System monitoring needs guardians
    monitor_keywords = ["monitor", "health check", "system status", "check service"]
    if any(kw in prompt for kw in monitor_keywords):
        guardians = [b for b in colony.bugs if b.role == "guardian"]
        if guardians:
            return guardians[0]
    
    # Recon/exploration needs scouts
    recon_keywords = ["list", "find", "explore", "scan", "read file", "check if"]
    if any(kw in prompt for kw in recon_keywords):
        scouts = [b for b in colony.bugs if b.role == "scout"]
        if scouts:
            return scouts[0]
    
    # Default: return first available worker, then any bug
    workers = [b for b in colony.bugs if b.role == "worker"]
    if workers:
        return workers[0]
    return colony.bugs[0]


def execute_parallel_swarm(tasks: List[Dict[str, str]], colony: Colony) -> List[Dict]:
    """Execute tasks in TRUE PARALLEL using ThreadPoolExecutor"""
    import time
    start = time.time()
    results = []
    
    # Smart routing - match tasks to appropriate bugs
    bug_assignments = []
    used_bugs = set()
    for task in tasks:
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
        bug_assignments.append((bug, task))
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=len(colony.bugs)) as executor:
        futures = {}
        for bug, task in bug_assignments:
            prompt = task.get("prompt", "")
            context = task.get("context", "")
            full_prompt = f"Context: {context}\n\nTask: {prompt}" if context else prompt
            future = executor.submit(sync_query_ollama, bug.model, full_prompt)
            futures[future] = (bug, task)
        
        for future in as_completed(futures):
            bug, task = futures[future]
            try:
                response = future.result()
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "model": bug.model,
                    "task": task.get("prompt", "")[:50],
                    "response": response,
                    "status": "success"
                })
                bug.status = "completed"
                bug.last_task = task.get("prompt", "")[:50]
            except Exception as e:
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "error": str(e),
                    "status": "failed"
                })
    
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
