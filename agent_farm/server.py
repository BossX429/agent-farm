"""
Agent Farm v3.4 - Chunked Write Edition
AI organism evolution and parallel task execution
Bug colonies with specialized roles, tool-enabled agents
TRUE PARALLEL via ThreadPoolExecutor

NEW IN v3.4:
- Chunked Write Pattern: bugs write sections in parallel, Python assembles
- chunked_write: Generate large documents (unlimited size)
- chunked_code_gen: Generate multi-function code files
- chunked_analysis: Multi-perspective analysis with synthesis
- Bypasses 500-char bug limitation via decomposition

v3.3: Ollama structured output for reliable tool parsing
v3.2: Synthesizer role with qwen2.5:14b
v3.1: 8.6x speed improvement, auto-format results

Models (all qwen3 for reliability):
- Scout: qwen3:4b (2.5GB)
- Worker: qwen3:4b (2.5GB)
- Memory: qwen3:4b (2.5GB)
- Guardian: qwen3:4b (2.5GB)
- Learner: qwen3:4b (2.5GB)
- Synthesizer: qwen2.5:14b (8.99GB)

Tools: 27 MCP tools
Performance: 8.6x faster than v3.0
"""

import os
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-farm")
OLLAMA_URL = "http://localhost:11434"

# Model tiers - optimized for 7900 XTX 24GB VRAM
# ALL models must be capable of following tool format
# Using qwen3:4b across the board for consistency
MODEL_TIERS = {
    "scout": "qwen3:4b",       # 2.5GB - reconnaissance
    "worker": "qwen3:4b",      # 2.5GB - task execution
    "memory": "qwen3:4b",      # 2.5GB - was 8b but inconsistent tool format
    "guardian": "qwen3:4b",    # 2.5GB - validation, protection
    "learner": "qwen3:4b",     # 2.5GB - was 8b but inconsistent tool format
    "synthesizer": "qwen2.5:14b",  # 8.99GB - synthesis/summarization (accuracy over speed)
}

# Grounding instruction to prevent hallucination
GROUNDING = "/nothink You are a specialized AI bug. Answer ONLY from given context. If unsure, say 'UNCERTAIN'. Be concise."


@dataclass
class Bug:
    """Individual bug in a colony"""
    id: str
    role: str
    model: str
    status: str = "idle"
    last_task: str = ""
    discoveries: List[str] = field(default_factory=list)


@dataclass 
class SharedConsciousness:
    """Shared state across entangled bugs"""
    discoveries: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    rag_cache: Dict[str, str] = field(default_factory=dict)


@dataclass
class NanoArmor:
    """Protection layer for bugs"""
    shield: bool = True      # Path traversal protection
    filter: bool = True      # Input sanitization  
    validation: bool = True  # Data verification
    scanner: bool = True     # Threat detection
    cache: bool = True       # Response caching
    formatter: bool = True   # Output formatting
    repair: bool = True      # Self-healing
    sensor: bool = True      # Environment monitoring
    attacks_blocked: int = 0


@dataclass
class Colony:
    """Bug colony with shared consciousness"""
    id: str
    bugs: List[Bug] = field(default_factory=list)
    consciousness: SharedConsciousness = field(default_factory=SharedConsciousness)
    armor: NanoArmor = field(default_factory=NanoArmor)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks_completed: int = 0


# Active colonies
colonies: Dict[str, Colony] = {}

# Stats tracking
stats = {
    "colonies_spawned": 0,
    "tasks_executed": 0,
    "discoveries": 0,
    "swarm_runs": 0,
    "parallel_speedup": []
}


async def query_ollama(model: str, prompt: str, grounded: bool = True) -> str:
    """Query Ollama with optional grounding instruction"""
    full_prompt = f"{GROUNDING}\n\n{prompt}" if grounded else prompt
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=120.0
            )
            if resp.status_code == 200:
                return resp.json().get("response", "No response")
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def sync_query_ollama(model: str, prompt: str, grounded: bool = True) -> str:
    """Synchronous wrapper for parallel execution"""
    full_prompt = f"{GROUNDING}\n\n{prompt}" if grounded else prompt
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=120.0
            )
            if resp.status_code == 200:
                return resp.json().get("response", "No response")
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def sync_query_raw(model: str, prompt: str) -> str:
    """Raw query WITHOUT grounding - for deep analysis that needs full thinking"""
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=300.0  # 5 min for complex analysis
            )
            if resp.status_code == 200:
                return resp.json().get("response", "No response")
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def synthesize_swarm_results(results: List[Dict], task_context: str = "") -> str:
    """
    Synthesize swarm results into a unified summary using the synthesizer model.
    Uses qwen2.5:14b for accuracy - no hallucinations.
    
    Args:
        results: List of swarm result dicts (with 'answer' or 'response' keys)
        task_context: Optional context about what the swarm was doing
    
    Returns:
        Synthesized summary string
    """
    model = MODEL_TIERS["synthesizer"]
    
    # Extract answers from results
    findings = []
    for i, r in enumerate(results, 1):
        answer = r.get("answer") or r.get("response") or r.get("result", "")
        if isinstance(answer, dict):
            answer = str(answer)
        task = r.get("task", f"Task {i}")
        role = r.get("role", "bug")
        status = r.get("status", "unknown")
        
        if answer and status not in ["failed", "error"]:
            findings.append(f"[{role}] {task}:\n{answer}")
    
    if not findings:
        return "No valid results to synthesize."
    
    combined = "\n\n---\n\n".join(findings)
    
    context_line = f"\nCONTEXT: {task_context}\n" if task_context else ""
    
    prompt = f"""You are a synthesis expert. Combine the following findings into ONE coherent summary.

STRICT RULES:
1. ONLY use information from the findings below - DO NOT add anything
2. If findings conflict, note the conflict
3. If something is uncertain, say "uncertain" or "unclear"
4. Be concise - bullet points are fine
5. No speculation, no assumptions, no hallucination
{context_line}
FINDINGS:
{combined}

---

SYNTHESIZED SUMMARY:"""
    
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=180.0  # 3 min for synthesis
            )
            if resp.status_code == 200:
                response = resp.json().get("response", "Synthesis failed")
                # Clean up common artifacts
                response = response.strip()
                if response.startswith("SYNTHESIZED SUMMARY:"):
                    response = response[20:].strip()
                return response
            return f"Synthesis error: {resp.status_code}"
    except Exception as e:
        return f"Synthesis error: {str(e)}"


def create_bug(colony_id: str, role: str) -> Bug:
    """Create a bug with appropriate model for role"""
    model = MODEL_TIERS.get(role, MODEL_TIERS["worker"])
    bug_id = f"{colony_id}-{role}-{datetime.now().strftime('%H%M%S')}"
    return Bug(id=bug_id, role=role, model=model)


def spawn_colony_internal(colony_type: str = "standard") -> Colony:
    """Spawn a new colony with bugs based on type"""
    colony_id = f"colony-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    # Calculate speedup (vs sequential estimate)
    sequential_estimate = elapsed * len(tasks) / max(len(colony.bugs), 1)
    if sequential_estimate > 0:
        speedup = sequential_estimate / elapsed
        stats["parallel_speedup"].append(round(speedup, 2))
    
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
        "avg_speedup": sum(stats["parallel_speedup"][-5:]) / max(len(stats["parallel_speedup"][-5:]), 1)
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
    
    avg_speedup = sum(stats["parallel_speedup"][-10:]) / max(len(stats["parallel_speedup"][-10:]), 1)
    
    return {
        "verdict": "healthy" if not issues else "issues_found",
        "issues": issues,
        "active_colonies": len(colonies),
        "total_spawned": stats["colonies_spawned"],
        "tasks_executed": stats["tasks_executed"],
        "swarm_runs": stats["swarm_runs"],
        "avg_parallel_speedup": round(avg_speedup, 2),
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
        "parallel_speedups": stats["parallel_speedup"][-10:],
        "avg_speedup": round(sum(stats["parallel_speedup"]) / max(len(stats["parallel_speedup"]), 1), 2),
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


@mcp.tool()
async def code_review_swarm(code: str, filepath: str = "") -> dict:
    """
    Run parallel code review with 4 specialized perspectives.
    Pass code directly OR filepath to read from disk.
    Returns: security, performance, style, and refactoring analysis.
    """
    import time
    start = time.time()
    
    # Read from file if filepath provided
    if filepath and not code:
        try:
            with open(filepath, 'r') as f:
                code = f.read()
        except Exception as e:
            return {"error": f"Failed to read {filepath}: {str(e)}"}
    
    if not code:
        return {"error": "No code provided. Pass code or filepath."}
    
    # Truncate if too long (keep first 500 lines max)
    lines = code.split('\n')
    if len(lines) > 500:
        code = '\n'.join(lines[:500])
        truncated = True
    else:
        truncated = False
    
    # Use qwen3:8b for all reviews - needs thinking capability
    model = "qwen3:8b"
    
    # Build explicit prompts with code embedded directly
    prompts = {
        "security": f"""You are a senior security engineer. Analyze the following code for security vulnerabilities.

RULES:
- You MUST analyze the EXACT code provided below
- List EVERY security issue you find
- Format: [SEVERITY] Issue description (line number if possible)
- Severities: CRITICAL, HIGH, MEDIUM, LOW
- If you find no issues, say "No security issues found"
- Do NOT give generic advice - only issues in THIS code

CODE TO ANALYZE:
```
{code}
```

SECURITY ISSUES FOUND:""",

        "performance": f"""You are a performance optimization expert. Analyze the following code for performance problems.

RULES:
- You MUST analyze the EXACT code provided below
- List EVERY performance issue you find
- Format: [IMPACT] Issue description -> Fix
- Impacts: HIGH, MEDIUM, LOW
- If you find no issues, say "No performance issues found"
- Do NOT give generic advice - only issues in THIS code

CODE TO ANALYZE:
```
{code}
```

PERFORMANCE ISSUES FOUND:""",

        "style": f"""You are a code quality reviewer. Analyze the following code for style and readability issues.

RULES:
- You MUST analyze the EXACT code provided below
- List EVERY style/readability issue you find
- Format: Issue description -> Recommendation
- Consider: naming, formatting, comments, structure, PEP8
- If you find no issues, say "No style issues found"
- Do NOT give generic advice - only issues in THIS code

CODE TO ANALYZE:
```
{code}
```

STYLE ISSUES FOUND:""",

        "refactoring": f"""You are a senior software architect. Suggest refactoring opportunities for the following code.

RULES:
- You MUST analyze the EXACT code provided below
- List concrete refactoring suggestions
- Format: [PRIORITY] What to refactor -> How to improve it
- Priorities: HIGH, MEDIUM, LOW
- Include brief code snippets for complex suggestions
- If code is already clean, say "No major refactoring needed"
- Do NOT give generic advice - only suggestions for THIS code

CODE TO ANALYZE:
```
{code}
```

REFACTORING SUGGESTIONS:"""
    }
    
    # Execute all 4 reviews in parallel using ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(sync_query_raw, model, prompt): review_type
            for review_type, prompt in prompts.items()
        }
        for future in as_completed(futures):
            review_type = futures[future]
            try:
                results[review_type] = future.result()
            except Exception as e:
                results[review_type] = f"Error: {str(e)}"
    
    elapsed = time.time() - start
    
    return {
        "filepath": filepath or "(direct input)",
        "lines_of_code": len(lines),
        "truncated": truncated,
        "model": model,
        "review": results,
        "elapsed_seconds": round(elapsed, 2)
    }


# ============ EXECUTION SWARMS ============

@mcp.tool()
async def code_gen_swarm(spec: str, language: str = "python") -> dict:
    """
    Generate code from spec with 4 parallel perspectives.
    Returns: main code, tests, docstring, usage examples.
    """
    import time
    start = time.time()
    model = "qwen3:8b"
    
    prompts = {
        "code": f"""You are an expert {language} developer. Generate clean, production-ready code.

SPECIFICATION:
{spec}

RULES:
- Write complete, working {language} code
- Include error handling
- Use best practices
- Add inline comments for complex logic
- Output ONLY the code, no explanations

CODE:""",

        "tests": f"""You are a test engineer. Generate comprehensive tests for this specification.

SPECIFICATION:
{spec}

RULES:
- Write complete test cases in {language}
- Cover edge cases and error conditions
- Use appropriate testing framework (pytest for Python)
- Output ONLY the test code, no explanations

TESTS:""",

        "docs": f"""You are a technical writer. Generate documentation for this specification.

SPECIFICATION:
{spec}

RULES:
- Write a clear docstring/documentation
- Include parameters, return values, exceptions
- Add usage examples in the docstring
- Output ONLY the documentation, no explanations

DOCUMENTATION:""",

        "examples": f"""You are a developer advocate. Generate usage examples for this specification.

SPECIFICATION:
{spec}

RULES:
- Write 3-5 practical usage examples
- Show different use cases
- Include comments explaining each example
- Output ONLY the example code, no explanations

EXAMPLES:"""
    }
    
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(sync_query_raw, model, prompt): gen_type
            for gen_type, prompt in prompts.items()
        }
        for future in as_completed(futures):
            gen_type = futures[future]
            try:
                results[gen_type] = future.result()
            except Exception as e:
                results[gen_type] = f"Error: {str(e)}"
    
    elapsed = time.time() - start
    return {
        "spec": spec[:100] + "..." if len(spec) > 100 else spec,
        "language": language,
        "generated": results,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def file_swarm(operations: str) -> dict:
    """
    Execute parallel file operations.
    operations: JSON array of {action, path, content?}
    Actions: read, write, append, exists, delete
    Example: [{"action": "write", "path": "/tmp/test.txt", "content": "hello"}]
    """
    import time
    start = time.time()
    
    try:
        ops = json.loads(operations)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON for operations"}
    
    def execute_file_op(op: dict) -> dict:
        action = op.get("action", "")
        path = op.get("path", "")
        content = op.get("content", "")
        
        # Safety: block dangerous paths
        dangerous = ["/etc", "/usr", "/bin", "/sbin", "/boot", "/root", "/sys", "/proc"]
        if any(path.startswith(d) for d in dangerous):
            return {"action": action, "path": path, "error": "Dangerous path blocked"}
        
        try:
            if action == "read":
                with open(path, 'r') as f:
                    return {"action": action, "path": path, "content": f.read(), "status": "success"}
            elif action == "write":
                with open(path, 'w') as f:
                    f.write(content)
                return {"action": action, "path": path, "bytes": len(content), "status": "success"}
            elif action == "append":
                with open(path, 'a') as f:
                    f.write(content)
                return {"action": action, "path": path, "bytes": len(content), "status": "success"}
            elif action == "exists":
                import os
                return {"action": action, "path": path, "exists": os.path.exists(path), "status": "success"}
            elif action == "delete":
                import os
                os.remove(path)
                return {"action": action, "path": path, "status": "success"}
            else:
                return {"action": action, "path": path, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"action": action, "path": path, "error": str(e)}
    
    results = []
    with ThreadPoolExecutor(max_workers=len(ops)) as executor:
        futures = [executor.submit(execute_file_op, op) for op in ops]
        for future in as_completed(futures):
            results.append(future.result())
    
    elapsed = time.time() - start
    return {
        "operations": len(ops),
        "results": results,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def exec_swarm(commands: str) -> dict:
    """
    Execute shell commands in parallel (with safety checks).
    commands: JSON array of command strings
    Example: ["ls -la", "pwd", "whoami"]
    BLOCKED: rm -rf, sudo, dd, mkfs, chmod 777, etc.
    """
    import time
    import subprocess
    start = time.time()
    
    try:
        cmds = json.loads(commands)
    except json.JSONDecodeError:
        cmds = [commands]  # Single command
    
    # Dangerous command patterns
    blocked = [
        "rm -rf", "rm -r /", "sudo", "dd if=", "mkfs", "chmod 777",
        ":(){ :|:& };:", "> /dev/sd", "mv /* ", "wget|sh", "curl|sh",
        "chmod -R 777", "chown -R", "> /dev/null", "shutdown", "reboot",
        "init 0", "init 6", "kill -9 1", "pkill", "killall"
    ]
    
    def execute_cmd(cmd: str) -> dict:
        # Safety check
        cmd_lower = cmd.lower()
        for b in blocked:
            if b in cmd_lower:
                return {"command": cmd, "error": f"Blocked dangerous pattern: {b}", "status": "blocked"}
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # 30 second max
            )
            return {
                "command": cmd,
                "stdout": result.stdout[:2000] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
                "returncode": result.returncode,
                "status": "success" if result.returncode == 0 else "failed"
            }
        except subprocess.TimeoutExpired:
            return {"command": cmd, "error": "Timeout (30s)", "status": "timeout"}
        except Exception as e:
            return {"command": cmd, "error": str(e), "status": "error"}
    
    results = []
    with ThreadPoolExecutor(max_workers=min(len(cmds), 8)) as executor:
        futures = [executor.submit(execute_cmd, cmd) for cmd in cmds]
        for future in as_completed(futures):
            results.append(future.result())
    
    elapsed = time.time() - start
    return {
        "commands": len(cmds),
        "results": results,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def api_swarm(requests: str) -> dict:
    """
    Execute parallel HTTP API requests.
    requests: JSON array of {url, method?, headers?, body?}
    Example: [{"url": "https://api.example.com/data"}, {"url": "...", "method": "POST", "body": "{}"}]
    """
    import time
    start = time.time()
    
    try:
        reqs = json.loads(requests)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON for requests"}
    
    def execute_request(req: dict) -> dict:
        url = req.get("url", "")
        method = req.get("method", "GET").upper()
        headers = req.get("headers", {})
        body = req.get("body", None)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                if method == "GET":
                    resp = client.get(url, headers=headers)
                elif method == "POST":
                    resp = client.post(url, headers=headers, content=body)
                elif method == "PUT":
                    resp = client.put(url, headers=headers, content=body)
                elif method == "DELETE":
                    resp = client.delete(url, headers=headers)
                else:
                    return {"url": url, "error": f"Unsupported method: {method}"}
                
                return {
                    "url": url,
                    "method": method,
                    "status_code": resp.status_code,
                    "body": resp.text[:2000] if resp.text else "",
                    "status": "success"
                }
        except Exception as e:
            return {"url": url, "method": method, "error": str(e), "status": "failed"}
    
    results = []
    with ThreadPoolExecutor(max_workers=min(len(reqs), 10)) as executor:
        futures = [executor.submit(execute_request, req) for req in reqs]
        for future in as_completed(futures):
            results.append(future.result())
    
    elapsed = time.time() - start
    return {
        "requests": len(reqs),
        "results": results,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def kmkb_swarm(queries: str, synthesize: bool = True) -> dict:
    """
    Query KMKB from multiple angles in parallel, optionally synthesize.
    queries: JSON array of query strings OR single topic to auto-expand
    Example: ["what is X?", "how does X work?", "examples of X"]
    Or just: "Agent Farm" (auto-expands to multiple queries)
    """
    import time
    start = time.time()
    
    KMKB_URL = "http://localhost:11434"  # Ollama for synthesis
    
    try:
        q_list = json.loads(queries)
    except (json.JSONDecodeError, TypeError):
        # Auto-expand single topic into multiple queries
        topic = queries
        q_list = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"What are the key features of {topic}?",
            f"What problems does {topic} solve?",
            f"How do I use {topic}?"
        ]
    
    def query_kmkb(query: str) -> dict:
        """Query KMKB via its search endpoint"""
        try:
            with httpx.Client(timeout=30.0) as client:
                # Try KMKB's ask endpoint on the default port
                resp = client.post(
                    "http://127.0.0.1:8765/ask",
                    json={"question": query, "top_k": 3},
                    timeout=30.0
                )
                if resp.status_code == 200:
                    return {"query": query, "result": resp.json(), "status": "success"}
        except:
            pass
        
        # Fallback: use Ollama with the query directly
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{KMKB_URL}/api/generate",
                    json={"model": "qwen3:4b", "prompt": query, "stream": False},
                    timeout=60.0
                )
                if resp.status_code == 200:
                    return {"query": query, "result": resp.json().get("response", ""), "status": "ollama_fallback"}
        except Exception as e:
            return {"query": query, "error": str(e), "status": "failed"}
        
        return {"query": query, "error": "No response", "status": "failed"}
    
    results = []
    with ThreadPoolExecutor(max_workers=len(q_list)) as executor:
        futures = [executor.submit(query_kmkb, q) for q in q_list]
        for future in as_completed(futures):
            results.append(future.result())
    
    # Synthesize if requested
    synthesis = None
    if synthesize and len(results) > 1:
        combined = "\n\n".join([
            f"Q: {r['query']}\nA: {r.get('result', r.get('error', 'No result'))}"
            for r in results
        ])
        synth_prompt = f"""Synthesize these research findings into a coherent summary:

{combined}

SYNTHESIS (be concise, extract key insights):"""
        
        synthesis = sync_query_raw("qwen3:8b", synth_prompt)
    
    elapsed = time.time() - start
    return {
        "queries": len(q_list),
        "results": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


# ============ TOOL-ENABLED AGENTS ============

@mcp.tool()
async def tool_swarm(tasks: str, colony_type: str = "hybrid", deep: bool = False, synthesize: bool = False) -> dict:
    """
    Deploy tool-enabled agents that can use real system tools.
    Each bug role has different tool permissions:
    - scout: read_file, list_dir, file_exists, system_status, process_list
    - worker: read_file, write_file, exec_cmd, http_get, http_post
    - memory: read_file, kmkb_search, kmkb_ask, list_dir
    - guardian: system_status, process_list, disk_usage, check_service
    - learner: read_file, analyze_code, list_dir, kmkb_search
    
    tasks: JSON array of task objects:
      Standard: {"prompt": "Do something"}
      Write:    {"path": "/file.txt", "content": "data..."}  <-- DIRECT EXECUTE for long content
    
    Example: [{"prompt": "Check system health"}, {"path": "/tmp/out.txt", "content": "results"}]
    
    DIRECT EXECUTE: Write tasks with content >300 chars bypass LLM entirely.
    Bugs can't reliably echo long content - we write directly instead.
    
    deep: Enable deep work mode - bugs chain multiple tool calls for complex tasks
    synthesize: If True, uses qwen2.5:14b to synthesize results into unified summary
    """
    import time
    from .agent import run_tool_swarm
    from .tools import execute_tool
    start = time.time()
    
    # Spawn colony
    colony = spawn_colony_internal(colony_type)
    
    # Parse tasks
    try:
        task_list = json.loads(tasks)
    except (json.JSONDecodeError, TypeError):
        task_list = [{"prompt": tasks}]
    
    # DIRECT EXECUTE: Intercept heavy write tasks - bugs can't echo long content
    CONTENT_THRESHOLD = 300  # chars - below this bugs handle fine
    direct_results = []
    bug_tasks = []
    
    for task in task_list:
        prompt = task.get("prompt", "")
        content = task.get("content", "")
        path = task.get("path", "")
        
        # Check if this is a write task with long content
        is_write_task = content and path
        is_heavy = len(content) > CONTENT_THRESHOLD
        
        if is_write_task and is_heavy:
            # Direct execute - skip the LLM entirely
            result = execute_tool("write_file", "worker", path=path, content=content)
            direct_results.append({
                "bug_id": "direct-execute",
                "role": "worker",
                "model": "none",
                "task": f"write_file({path})",
                "answer": f"Wrote {len(content)} chars to {path}" if "error" not in result else result.get("error"),
                "tool_calls": 1,
                "iterations": 1,
                "status": "completed" if "error" not in result else "failed",
                "direct_execute": True
            })
        else:
            bug_tasks.append(task)
    
    # Run remaining tasks through bugs
    bug_results = run_tool_swarm(bug_tasks, colony.bugs, deep=deep) if bug_tasks else []
    results = direct_results + bug_results

    
    elapsed = time.time() - start
    colony.tasks_completed += len(task_list)
    stats["tasks_executed"] += len(task_list)
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = synthesize_swarm_results(results, task_context=f"Tool swarm with {colony_type} colony")
    
    return {
        "colony_id": colony.id,
        "colony_type": colony_type,
        "tasks": len(task_list),
        "deep_mode": deep,
        "results": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def system_health_swarm(synthesize: bool = True) -> dict:
    """
    Quick system health check using tool-enabled bugs.
    Deploys guardian bugs to check CPU, memory, disk, and services.
    
    synthesize: If True (default), uses qwen2.5:14b to synthesize results into unified summary
    """
    import time
    from .agent import run_tool_swarm
    start = time.time()
    
    # Spawn guardian colony
    colony_id = f"health-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    bugs = [
        create_bug(colony_id, "guardian"),
        create_bug(colony_id, "guardian"),
        create_bug(colony_id, "guardian"),
        create_bug(colony_id, "guardian"),
    ]
    
    tasks = [
        {"prompt": "Check overall system status (CPU, memory, disk) and report if anything is concerning."},
        {"prompt": "List the top processes using the most CPU. Report any that seem unusual."},
        {"prompt": "Check if ollama service is running and healthy."},
        {"prompt": "Check disk usage and report if any partition is above 80% full."},
    ]
    
    results = run_tool_swarm(tasks, bugs)
    elapsed = time.time() - start
    
    # Summarize health
    issues = []
    for r in results:
        if "error" in r.get("answer", "").lower() or "warning" in r.get("answer", "").lower():
            issues.append(r.get("answer", ""))
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = synthesize_swarm_results(results, task_context="System health check")
    
    return {
        "verdict": "healthy" if not issues else "issues_found",
        "issues": issues,
        "checks": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def recon_swarm(target_path: str, deep: bool = False, synthesize: bool = True) -> dict:
    """
    Reconnaissance swarm - scouts explore a directory/codebase.
    Uses scout bugs with read-only tools to map out a target.
    
    target_path: Directory to explore
    deep: Enable deep work mode (default False) - multi-iteration for complex analysis
    synthesize: If True (default), uses qwen2.5:14b to synthesize findings into unified report
    """
    import time
    from .agent import run_tool_swarm
    start = time.time()
    
    # Expand ~ to home directory
    if target_path.startswith("~"):
        target_path = os.path.expanduser(target_path)
    
    colony_id = f"recon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # All scouts with exec_cmd for recon (they have it now)
    bugs = [
        create_bug(colony_id, "scout"),
        create_bug(colony_id, "scout"),
        create_bug(colony_id, "scout"),
        create_bug(colony_id, "scout"),
    ]
    
    tasks = [
        {"prompt": f"List the contents of {target_path} using list_dir. Report what type of project this is."},
        {"prompt": f"Use exec_cmd to run 'du -sh {target_path}/*' and report the sizes of each item."},
        {"prompt": f"Use exec_cmd to run 'find {target_path} -maxdepth 2 -name \"*.py\" -o -name \"*.toml\" -o -name \"*.json\" | head -20' and list what config/code files exist."},
        {"prompt": f"Use exec_cmd to run 'cat {target_path}/README.md 2>/dev/null || cat {target_path}/README 2>/dev/null || echo No README found' and summarize."},
    ]
    
    results = run_tool_swarm(tasks, bugs, deep=deep)
    elapsed = time.time() - start
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = synthesize_swarm_results(results, task_context=f"Reconnaissance of {target_path}")
    
    return {
        "target": target_path,
        "findings": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def worker_task(task: str, context: str = "") -> dict:
    """
    Single worker bug with tools to complete a task.
    Worker has: read_file, write_file, exec_cmd, http_get, http_post
    
    task: What to do
    context: Optional context/background
    """
    import time
    from .agent import run_tool_agent
    start = time.time()
    
    result = run_tool_agent(
        role="worker",
        model=MODEL_TIERS["worker"],
        task=task,
        context=context,
        deep=True  # Workers always use deep mode for complex tasks
    )
    
    elapsed = time.time() - start
    
    return {
        "task": task[:100],
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
        "status": result["status"],
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def heavy_write(path: str, content: str) -> dict:
    """
    Direct file write - NO LLM involved. Use for large content.
    
    Bugs can't reliably echo content >300 chars through the LLM.
    This tool writes directly to disk, bypassing the model entirely.
    
    path: File path to write to
    content: Content to write (any length)
    
    Returns: success/error status and bytes written
    """
    from .tools import execute_tool
    
    result = execute_tool("write_file", "worker", path=path, content=content)
    
    if "error" in result:
        return {
            "status": "failed",
            "error": result["error"],
            "path": path
        }
    
    return {
        "status": "completed",
        "path": path,
        "bytes_written": result.get("bytes_written", len(content)),
        "direct_execute": True
    }


@mcp.tool()
async def deep_analysis_swarm(target_path: str, analysis_type: str = "full", synthesize: bool = True) -> dict:
    """
    Deep analysis swarm using WORKERS with exec_cmd for thorough system analysis.
    Workers can run shell commands (find, du, grep, etc.) for real analysis.
    
    target_path: Directory/path to analyze
    analysis_type: 'full', 'redundant', 'sizes', 'cleanup'
    synthesize: If True (default), uses qwen2.5:14b to synthesize findings
    
    Use this for finding: redundant files, cache sizes, disk usage, log files, etc.
    """
    import time
    from .agent import run_tool_swarm
    start = time.time()
    
    colony_id = f"deep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Workers have exec_cmd - essential for deep analysis
    bugs = [
        create_bug(colony_id, "worker"),
        create_bug(colony_id, "worker"),
        create_bug(colony_id, "worker"),
        create_bug(colony_id, "worker"),
    ]
    
    if analysis_type == "redundant":
        tasks = [
            {"prompt": f"Find all __pycache__ directories under {target_path} and calculate total size. Use: exec_cmd(find {target_path} -name '__pycache__' -type d -exec du -sh {{}} \\;)"},
            {"prompt": f"Find all .venv or venv directories under {target_path} and report their sizes. Use exec_cmd with find and du."},
            {"prompt": f"Find duplicate or redundant files. Check for: .pyc files, .log files over 1MB, temp files. Use exec_cmd with find."},
            {"prompt": f"Look for old/stale files not accessed in 30+ days under {target_path}. Use: exec_cmd(find {target_path} -atime +30 -type f | head -20)"},
        ]
    elif analysis_type == "sizes":
        tasks = [
            {"prompt": f"Get disk usage breakdown for {target_path}. Use: exec_cmd(du -sh {target_path}/*) to show each subdirectory size."},
            {"prompt": f"Find the 10 largest files under {target_path}. Use: exec_cmd(find {target_path} -type f -exec du -h {{}} + | sort -rh | head -10)"},
            {"prompt": f"Find all files over 10MB under {target_path}. Use: exec_cmd(find {target_path} -size +10M -type f -exec ls -lh {{}} \\;)"},
            {"prompt": f"Count total files and directories under {target_path}. Use exec_cmd with find and wc -l."},
        ]
    elif analysis_type == "cleanup":
        tasks = [
            {"prompt": f"Find all .log files under {target_path} and report sizes. Use: exec_cmd(find {target_path} -name '*.log' -exec du -sh {{}} \\;)"},
            {"prompt": f"Find cache directories (.cache, __pycache__, node_modules/.cache). Use exec_cmd with find."},
            {"prompt": f"Check for backup files (*~, *.bak, *.swp). Use: exec_cmd(find {target_path} -name '*~' -o -name '*.bak' -o -name '*.swp')"},
            {"prompt": f"Find empty directories that could be removed. Use: exec_cmd(find {target_path} -type d -empty)"},
        ]
    else:  # full
        tasks = [
            {"prompt": f"Get complete disk usage breakdown for {target_path}. Use: exec_cmd(du -sh {target_path}/*) and summarize what's using space."},
            {"prompt": f"Find all __pycache__ and .pyc files. Calculate total size. Use exec_cmd with find and du."},
            {"prompt": f"Find all log files over 1MB. Use: exec_cmd(find {target_path} -name '*.log' -size +1M -exec ls -lh {{}} \\;)"},
            {"prompt": f"Identify potential cleanup targets: old venvs, caches, temp files. Use exec_cmd to find and size them."},
        ]
    
    results = run_tool_swarm(tasks, bugs, deep=True)
    elapsed = time.time() - start
    
    # Extract findings summary
    findings = []
    for r in results:
        if r.get("status") == "completed" and r.get("answer"):
            findings.append(r["answer"])
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = synthesize_swarm_results(results, task_context=f"Deep {analysis_type} analysis of {target_path}")
    
    return {
        "target": target_path,
        "analysis_type": analysis_type,
        "findings": results,
        "summary": findings,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def synthesize(results: str, context: str = "") -> dict:
    """
    Standalone synthesis tool - synthesize any JSON results into unified summary.
    Uses qwen2.5:14b for accuracy.
    
    results: JSON array of result objects (with 'answer' or 'response' keys)
    context: Optional context about what these results are from
    
    Example: synthesize('[{"answer": "CPU at 5%"}, {"answer": "Memory at 20%"}]', "health check")
    """
    import time
    start = time.time()
    
    try:
        result_list = json.loads(results)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON for results"}
    
    synthesis = synthesize_swarm_results(result_list, task_context=context)
    elapsed = time.time() - start
    
    return {
        "input_count": len(result_list),
        "model": MODEL_TIERS["synthesizer"],
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


# ============ CHUNKED WRITE PATTERN ============
# Bugs generate small pieces, Python concatenates directly

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    },
    "required": ["sections"]
}


def plan_sections(spec: str, num_sections: int = 5) -> List[Dict]:
    """
    Use a bug to plan document sections.
    Returns list of {title, description} for each section.
    """
    from .agent import query_ollama_structured
    
    prompt = f"""Create an outline with exactly {num_sections} sections for this document:

SPECIFICATION: {spec}

Return JSON with a "sections" array. Each section needs:
- title: Short section title
- description: 1-2 sentence description of what this section should cover

Make sections logical and sequential. Keep descriptions under 100 chars."""

    response = query_ollama_structured(
        MODEL_TIERS["learner"],  # Use learner for planning
        prompt,
        SECTION_SCHEMA
    )
    
    if "error" in response:
        # Fallback: generate generic sections
        return [
            {"title": f"Section {i+1}", "description": f"Part {i+1} of the document about: {spec[:50]}"}
            for i in range(num_sections)
        ]
    
    sections = response.get("sections", [])
    if len(sections) < num_sections:
        # Pad with generic sections
        for i in range(len(sections), num_sections):
            sections.append({"title": f"Section {i+1}", "description": "Additional content"})
    
    return sections[:num_sections]


def generate_section_content(section: Dict, spec: str, section_num: int, total: int) -> str:
    """
    Have a worker bug generate content for one section.
    Keeps output under 400 chars to avoid corruption.
    """
    from .agent import query_ollama_sync
    
    prompt = f"""Write section {section_num} of {total} for this document.

DOCUMENT TOPIC: {spec}

SECTION TITLE: {section['title']}
SECTION SHOULD COVER: {section['description']}

RULES:
- Write 3-5 paragraphs for this section ONLY
- Be concise but informative
- Do NOT include other sections
- Do NOT add headers for other sections
- Start with the section title as a header

Write the section content now:"""

    response = query_ollama_sync(MODEL_TIERS["worker"], prompt)
    
    # Clean up response
    content = response.strip()
    
    # Remove any thinking tags
    import re
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    
    return content


@mcp.tool()
async def chunked_write(
    output_path: str,
    spec: str,
    num_sections: int = 5,
    doc_type: str = "markdown"
) -> dict:
    """
    Generate large documents by having bugs write sections in parallel.
    Bypasses the long-content limitation by chunking work.
    
    HOW IT WORKS:
    1. Planner bug creates outline (structured JSON)
    2. Worker bugs generate sections in PARALLEL
    3. Python concatenates directly (NO LLM involved)
    4. heavy_write saves result
    
    output_path: Where to save the document
    spec: What the document should be about
    num_sections: How many sections (default 5, max 10)
    doc_type: 'markdown', 'text', or 'code'
    
    EXAMPLE:
    chunked_write("/tmp/report.md", "Analysis of Python best practices", 5)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    start = time.time()
    
    # Clamp sections
    num_sections = max(2, min(10, num_sections))
    
    # Step 1: Plan sections
    sections = plan_sections(spec, num_sections)
    
    # Step 2: Generate sections in parallel
    section_contents = [None] * num_sections
    
    with ThreadPoolExecutor(max_workers=min(4, num_sections)) as executor:
        futures = {
            executor.submit(
                generate_section_content,
                sections[i],
                spec,
                i + 1,
                num_sections
            ): i
            for i in range(num_sections)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                section_contents[idx] = future.result()
            except Exception as e:
                section_contents[idx] = f"[Section {idx+1} generation failed: {str(e)}]"
    
    # Step 3: Concatenate directly (NO LLM)
    if doc_type == "markdown":
        header = f"# {spec}\n\n"
        separator = "\n\n---\n\n"
    elif doc_type == "code":
        header = f"# {spec}\n# Auto-generated by Agent Farm chunked_write\n\n"
        separator = "\n\n"
    else:
        header = f"{spec}\n{'='*len(spec)}\n\n"
        separator = "\n\n"
    
    full_content = header + separator.join(section_contents)
    
    # Step 4: Direct write (bypasses LLM)
    from .tools import tool_write_file
    write_result = tool_write_file(output_path, full_content)
    
    elapsed = time.time() - start
    
    return {
        "status": "completed" if "error" not in write_result else "failed",
        "output_path": output_path,
        "spec": spec,
        "sections_planned": len(sections),
        "sections_generated": len([s for s in section_contents if s]),
        "total_chars": len(full_content),
        "total_lines": full_content.count('\n') + 1,
        "section_titles": [s["title"] for s in sections],
        "write_result": write_result,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def chunked_code_gen(
    output_path: str,
    spec: str,
    language: str = "python",
    num_functions: int = 4
) -> dict:
    """
    Generate code files by having bugs write functions in parallel.
    Each bug writes one function, Python assembles the file.
    
    output_path: Where to save the code file
    spec: What the code should do
    language: 'python', 'javascript', 'bash', etc.
    num_functions: How many functions to generate (default 4, max 8)
    
    EXAMPLE:
    chunked_code_gen("/tmp/utils.py", "File utilities: read, write, copy, delete", "python", 4)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .agent import query_ollama_sync, query_ollama_structured
    start = time.time()
    
    num_functions = max(2, min(8, num_functions))
    
    # Step 1: Plan functions
    plan_prompt = f"""Plan {num_functions} functions for this code:

SPECIFICATION: {spec}
LANGUAGE: {language}

Return JSON with "sections" array where each item has:
- title: function name (e.g., "read_file", "calculate_sum")
- description: what the function does (1 sentence)

Make functions logical and complementary."""

    plan_response = query_ollama_structured(
        MODEL_TIERS["learner"],
        plan_prompt,
        SECTION_SCHEMA
    )
    
    functions = plan_response.get("sections", [])
    if len(functions) < num_functions:
        for i in range(len(functions), num_functions):
            functions.append({"title": f"helper_{i}", "description": "Helper function"})
    functions = functions[:num_functions]
    
    # Step 2: Generate functions in parallel
    def generate_function(func: Dict, idx: int) -> str:
        prompt = f"""Write ONE {language} function:

FUNCTION NAME: {func['title']}
PURPOSE: {func['description']}
CONTEXT: Part of a module for: {spec}

RULES:
- Write ONLY this one function
- Include docstring/comments
- Keep it focused and clean
- Do NOT include imports (those go at top)
- Do NOT include other functions

Write the function now:"""
        
        response = query_ollama_sync(MODEL_TIERS["worker"], prompt)
        
        # Clean up
        import re
        content = response.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'```\w*\n?', '', content)
        content = re.sub(r'```', '', content)
        return content.strip()
    
    function_contents = [None] * num_functions
    
    with ThreadPoolExecutor(max_workers=min(4, num_functions)) as executor:
        futures = {
            executor.submit(generate_function, functions[i], i): i
            for i in range(num_functions)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                function_contents[idx] = future.result()
            except Exception as e:
                function_contents[idx] = f"# Function {idx} generation failed: {str(e)}"
    
    # Step 3: Assemble file (NO LLM)
    if language == "python":
        header = f'''"""
{spec}
Auto-generated by Agent Farm chunked_code_gen
"""

'''
        separator = "\n\n\n"
    elif language in ["javascript", "typescript"]:
        header = f'''/**
 * {spec}
 * Auto-generated by Agent Farm chunked_code_gen
 */

'''
        separator = "\n\n"
    elif language == "bash":
        header = f'''#!/bin/bash
# {spec}
# Auto-generated by Agent Farm chunked_code_gen

'''
        separator = "\n\n"
    else:
        header = f"// {spec}\n// Auto-generated by Agent Farm\n\n"
        separator = "\n\n"
    
    full_content = header + separator.join(function_contents)
    
    # Step 4: Write directly
    from .tools import tool_write_file
    write_result = tool_write_file(output_path, full_content)
    
    elapsed = time.time() - start
    
    return {
        "status": "completed" if "error" not in write_result else "failed",
        "output_path": output_path,
        "language": language,
        "spec": spec,
        "functions_planned": len(functions),
        "functions_generated": len([f for f in function_contents if f]),
        "function_names": [f["title"] for f in functions],
        "total_chars": len(full_content),
        "total_lines": full_content.count('\n') + 1,
        "write_result": write_result,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
async def chunked_analysis(
    target: str,
    question: str,
    num_perspectives: int = 4
) -> dict:
    """
    Analyze something from multiple perspectives in parallel.
    Each bug analyzes from a different angle, results are synthesized.
    
    target: What to analyze (file path, concept, code, etc.)
    question: The analysis question
    num_perspectives: How many different angles (default 4)
    
    EXAMPLE:
    chunked_analysis("/home/kyle/repos/project", "What are the main architectural patterns?", 4)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .agent import query_ollama_sync
    start = time.time()
    
    num_perspectives = max(2, min(6, num_perspectives))
    
    # Define analysis perspectives
    perspectives = [
        {"name": "Structure", "focus": "organization, layout, components"},
        {"name": "Patterns", "focus": "design patterns, conventions, idioms"},
        {"name": "Quality", "focus": "code quality, best practices, issues"},
        {"name": "Performance", "focus": "efficiency, bottlenecks, optimization"},
        {"name": "Security", "focus": "vulnerabilities, risks, safety"},
        {"name": "Maintainability", "focus": "readability, modularity, documentation"},
    ][:num_perspectives]
    
    # Generate analyses in parallel
    def analyze_perspective(persp: Dict) -> Dict:
        prompt = f"""Analyze from the {persp['name']} perspective.

TARGET: {target}
QUESTION: {question}
FOCUS ON: {persp['focus']}

Provide a focused analysis (3-5 key points) from this perspective only.
Be specific and actionable."""
        
        response = query_ollama_sync(MODEL_TIERS["learner"], prompt)
        
        import re
        content = response.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        return {
            "perspective": persp["name"],
            "focus": persp["focus"],
            "analysis": content
        }
    
    analyses = []
    
    with ThreadPoolExecutor(max_workers=min(4, num_perspectives)) as executor:
        futures = [executor.submit(analyze_perspective, p) for p in perspectives]
        
        for future in as_completed(futures):
            try:
                analyses.append(future.result())
            except Exception as e:
                analyses.append({"perspective": "Error", "analysis": str(e)})
    
    # Synthesize
    synthesis = synthesize_swarm_results(
        [{"answer": a["analysis"], "role": a["perspective"]} for a in analyses],
        task_context=f"Multi-perspective analysis of {target}: {question}"
    )
    
    elapsed = time.time() - start
    
    return {
        "target": target,
        "question": question,
        "perspectives": len(analyses),
        "analyses": analyses,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


def main():
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
