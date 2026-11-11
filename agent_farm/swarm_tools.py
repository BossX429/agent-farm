"""MCP tools that run swarms: code, file, exec, api, kmkb, and others."""

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
from .metrics import track_tool_run
from .state import MODEL_TIERS, OLLAMA_URL, colonies, stats
from .colony import (
    _unique_id,
    create_bug,
    spawn_colony_internal,
    execute_parallel_swarm,
    entangle_colony,
)
from .ollama import query_ollama, sync_query_ollama, sync_query_raw, synthesize_swarm_results
from .action_log import log_action, redact_url
from .reliability import (
    EXEC_TASK_TIMEOUT_SECONDS,
    FILE_TASK_TIMEOUT_SECONDS,
    HTTP_TASK_TIMEOUT_SECONDS,
    MODEL_TASK_TIMEOUT_SECONDS,
    cap_aggregate_output,
    run_parallel,
    validate_parallel_items,
    validate_parallel_payload,
)



@mcp.tool()
@track_tool_run()
async def code_review_swarm(code: str, filepath: str = "") -> dict:
    """
    Run parallel code review with 4 specialized perspectives.
    Pass code directly OR filepath to read from disk.
    Returns: security, performance, style, and refactoring analysis.
    """
    import time
    start = time.time()
    
    # Read from file if filepath provided, through the same safe read
    # path as tool_read_file -- not a direct open(). Previously this
    # opened any path directly, with no blocked-path check and no byte
    # limit, so a model could ask it to read a sensitive file. Now it
    # is blocked for system directories and sensitive files, and capped
    # at 100KB by tool_read_file, before the 500-line truncation below.
    if filepath and not code:
        from .tools import tool_read_file, is_swarm_read_allowed

        if not is_swarm_read_allowed(filepath):
            return {"error": f"Blocked path: {filepath}"}
        result = tool_read_file(filepath)
        if "error" in result:
            return {"error": f"Failed to read {filepath}: {result['error']}"}
        code = result["content"]

    if not code:
        return {"error": "No code provided. Pass code or filepath."}

    # Truncate if too long (keep first 500 lines max)
    lines = code.split('\n')
    if len(lines) > 500:
        code = '\n'.join(lines[:500])
        truncated = True
    else:
        truncated = False
    
    # Use the synthesizer tier, not a hardcoded model outside MODEL_TIERS.
    # A model outside MODEL_TIERS was previously hardcoded here even though it is not a
    # documented tier, so reviews would fail or pull an unexpected model
    # on installs that only have the documented 4b + 14b models.
    model = MODEL_TIERS["synthesizer"]
    
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
    
    # Execute all 4 reviews in parallel, off the event loop, with a
    # per-review timeout. The shared parallel runner returns results
    # in input order, so callers do not depend on completion order.
    review_types = list(prompts.keys())
    raw_results = await asyncio.to_thread(
        run_parallel,
        [
            (lambda p=prompts[t]: sync_query_raw(model, p))
            for t in review_types
        ],
        max_workers=4,
        timeout=MODEL_TASK_TIMEOUT_SECONDS,
    )
    results = {}
    for review_type, raw in zip(review_types, raw_results):
        if isinstance(raw, dict):
            results[review_type] = f"Error: {raw.get('error', 'unknown error')}"
        else:
            results[review_type] = raw

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
@track_tool_run()
async def code_gen_swarm(spec: str, language: str = "python") -> dict:
    """
    Generate code from spec with 4 parallel perspectives.
    Returns: main code, tests, docstring, usage examples.
    """
    import time
    start = time.time()
    # See code_review_swarm: use the documented synthesizer tier.
    model = MODEL_TIERS["synthesizer"]

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
    
    gen_types = list(prompts.keys())
    raw_results = await asyncio.to_thread(
        run_parallel,
        [
            (lambda p=prompts[t]: sync_query_raw(model, p))
            for t in gen_types
        ],
        max_workers=4,
        timeout=MODEL_TASK_TIMEOUT_SECONDS,
    )
    results = {}
    for gen_type, raw in zip(gen_types, raw_results):
        if isinstance(raw, dict):
            results[gen_type] = f"Error: {raw.get('error', 'unknown error')}"
        else:
            results[gen_type] = raw

    elapsed = time.time() - start
    return {
        "spec": spec[:100] + "..." if len(spec) > 100 else spec,
        "language": language,
        "generated": results,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
async def file_swarm(operations: str) -> dict:
    """
    Execute parallel file operations.
    operations: JSON array of {action, path, content?}
    Actions: read, write, append, exists, delete
    Example: [{"action": "write", "path": "/tmp/test.txt", "content": "hello"}]
    """
    import time
    start = time.time()
    
    payload_error = validate_parallel_payload(operations, "operations")
    if payload_error:
        return {"error": payload_error}
    try:
        ops = json.loads(operations)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON for operations"}

    items_error = validate_parallel_items(ops, "operations")
    if items_error:
        return {"error": items_error}
    for op in ops:
        if not isinstance(op, dict) or not isinstance(op.get("action"), str):
            return {"error": "Each operation must be an object with a string 'action'"}

    def execute_file_op(op: dict) -> dict:
        action = op.get("action", "")
        raw_path = op.get("path", "")
        content = op.get("content", "")

        # Safety, unified with tools.py -- previously file_swarm kept its
        # own blocked-directory list and allowed write/append/delete
        # anywhere outside it, including the whole home directory
        # (credentials, SSH files, Documents, etc.), while write_file
        # only allows /tmp, /home/kyle/repos, and /home/kyle/Desktop.
        # Now: reads/exists use the shared swarm-read check (blocked
        # system dirs + sensitive files, canonical/symlink-resolved),
        # and writes/appends/deletes use the shared write allowlist.
        # Delete is therefore only allowed inside a write-allowed
        # directory, not anywhere in home. NOTE: this is still not a
        # sandbox -- see the exec_cmd note in tools.py. Run locally
        # with trusted prompts only.
        from .tools import (
            ALLOWED_WRITE_DIRS,
            tool_read_file,
            tool_write_file,
            tool_file_exists,
            is_swarm_read_allowed,
            is_write_path_allowed,
            _canonical_path,
        )
        import os as _os

        if action in ("read", "exists"):
            if raw_path and not is_swarm_read_allowed(raw_path):
                log_action("file_blocked", "blocked", f"file_swarm blocked {action}: {raw_path}")
                return {"action": action, "path": raw_path, "error": "Dangerous path blocked"}
        elif action in ("write", "append", "delete"):
            if raw_path and not is_write_path_allowed(raw_path):
                log_action("file_blocked", "blocked", f"file_swarm blocked {action}: {raw_path}")
                return {"action": action, "path": raw_path,
                        "error": f"Write only allowed in: {ALLOWED_WRITE_DIRS}"}

        try:
            if action == "read":
                result = tool_read_file(raw_path)
                if "error" in result:
                    return {"action": action, "path": raw_path, "error": result["error"]}
                return {"action": action, "path": raw_path,
                        "content": result["content"], "status": "success"}
            elif action == "write":
                result = tool_write_file(raw_path, content)
                if "error" in result:
                    return {"action": action, "path": raw_path, "error": result["error"]}
                return {"action": action, "path": raw_path, "bytes": result["bytes_written"], "status": "success"}
            elif action == "append":
                path = _canonical_path(raw_path)
                # Explicit UTF-8, matching tool_write_file: without
                # this, Windows appends in cp1252.
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(content)
                return {"action": action, "path": raw_path, "bytes": len(content), "status": "success"}
            elif action == "exists":
                result = tool_file_exists(raw_path)
                if "error" in result:
                    return {"action": action, "path": raw_path, "error": result["error"]}
                return {"action": action, "path": raw_path, "exists": result["exists"], "status": "success"}
            elif action == "delete":
                _os.remove(_canonical_path(raw_path))
                return {"action": action, "path": raw_path, "status": "success"}
            else:
                return {"action": action, "path": raw_path, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"action": action, "path": raw_path, "error": str(e)}

    # Cap workers like exec_swarm (8): the old max_workers=len(ops) let a
    # huge operation list spike threads. Results are returned in input
    # order, so callers can match results to ops, and the total text
    # returned is capped (see reliability.cap_aggregate_output).
    raw_results = await asyncio.to_thread(
        run_parallel,
        [(lambda o=op: execute_file_op(o)) for op in ops],
        max_workers=min(len(ops), 8),
        timeout=FILE_TASK_TIMEOUT_SECONDS,
    )
    results = []
    for raw in raw_results:
        if isinstance(raw, dict) and "action" not in raw:
            results.append({"action": "", "error": raw.get("error", "unknown error"), "status": "failed"})
        else:
            results.append(raw)
    results, output_truncated = cap_aggregate_output(results, ("content",))

    elapsed = time.time() - start
    return {
        "operations": len(ops),
        "results": results,
        "output_truncated": output_truncated,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
async def exec_swarm(commands: str) -> dict:
    """
    Execute commands in parallel (no shell at all; with safety checks).
    commands: JSON array of command strings
    Example: ["ls -la", "pwd", "whoami"]
    BLOCKED: rm -rf, sudo, dd, mkfs, chmod 777, pipe-to-shell, etc.
    NOTE: no shell is used at all -- only simple commands and safe
    pipelines (see tools.tool_exec_cmd). Still not a sandbox --
    local trusted use only.

    It delegates to tools.tool_exec_cmd for every command, so both
    paths enforce exactly the same rules.
    """
    import time
    from .tools import tool_exec_cmd
    start = time.time()

    payload_error = validate_parallel_payload(commands, "commands")
    if payload_error:
        return {"error": payload_error}
    try:
        cmds = json.loads(commands)
    except json.JSONDecodeError:
        cmds = [commands]  # Single command

    items_error = validate_parallel_items(cmds, "commands")
    if items_error:
        return {"error": items_error}
    if not all(isinstance(cmd, str) for cmd in cmds):
        return {"error": "Each command must be a string"}

    def execute_cmd(cmd: str) -> dict:
        result = tool_exec_cmd(cmd)
        if "error" in result:
            return {
                "command": cmd,
                "error": result["error"],
                "status": "blocked" if result["error"].startswith("Blocked") else "error",
            }
        return {
            "command": cmd,
            "stdout": (result.get("stdout") or "")[:2000],
            "stderr": (result.get("stderr") or "")[:500],
            "returncode": result.get("returncode"),
            "status": "success" if result.get("returncode") == 0 else "failed",
        }
    
    raw_results = await asyncio.to_thread(
        run_parallel,
        [(lambda c=cmd: execute_cmd(c)) for cmd in cmds],
        max_workers=min(len(cmds), 8),
        timeout=EXEC_TASK_TIMEOUT_SECONDS,
    )
    results = []
    for raw in raw_results:
        if isinstance(raw, dict) and "command" not in raw:
            results.append({"command": "", "error": raw.get("error", "unknown error"), "status": raw.get("status", "failed")})
        else:
            results.append(raw)
    results, output_truncated = cap_aggregate_output(results, ("stdout", "stderr"))

    elapsed = time.time() - start
    return {
        "commands": len(cmds),
        "results": results,
        "output_truncated": output_truncated,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
async def api_swarm(requests: str, allow_mutating: bool = False) -> dict:
    """
    Execute parallel HTTP API requests.
    requests: JSON array of {url, method?, headers?, body?}
    Example: [{"url": "https://api.example.com/data"}, {"url": "...", "method": "POST", "body": "{}"}]
    allow_mutating: False by default. When False, only GET requests
    run; POST, PUT, and DELETE are blocked before any network call,
    because SSRF checks alone do not stop a model from making a
    destructive call to a public service. Pass allow_mutating=True
    explicitly when a mutating call is intended. This is a per-call
    opt-in, not a server-wide setting, so a single call cannot
    silently enable mutating requests for later calls.

    This previously called httpx directly with no URL validation,
    bypassing the SSRF guard in tools.http_get/http_post. It now
    delegates to tools._http_request for every request, so public-URL
    only, redirect re-validation, IP pinning, and method checks apply
    here too.
    """
    import time
    from .tools import _http_request
    start = time.time()

    payload_error = validate_parallel_payload(requests, "requests")
    if payload_error:
        return {"error": payload_error}
    try:
        reqs = json.loads(requests)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON for requests"}

    items_error = validate_parallel_items(reqs, "requests")
    if items_error:
        return {"error": items_error}
    for req in reqs:
        if not isinstance(req, dict) or not isinstance(req.get("url"), str):
            return {"error": "Each request must be an object with a string 'url'"}

    def execute_request(req: dict) -> dict:
        url = req.get("url", "")
        method = req.get("method", "GET").upper()
        headers = req.get("headers", {}) or {}
        body = req.get("body") or ""
        if method in ("POST", "PUT", "DELETE") and not allow_mutating:
            log_action("api_mutating_blocked", "blocked",
                       f"mutating method blocked: {method} to {redact_url(url)}")
            return {
                "url": url,
                "method": method,
                "error": "Mutating HTTP method blocked: api_swarm is GET-only by default. "
                         "Pass allow_mutating=true to api_swarm to enable POST/PUT/DELETE.",
                "status": "blocked",
            }
        if method in ("POST", "PUT", "DELETE"):
            # Per-call opt-in only (there is no server-wide flag that
            # silently enables mutation for later calls), and the
            # opt-in is visible in the action log when enabled.
            # Structurally making mutation impossible would break
            # this tool's intended purpose (calling APIs), so the
            # safe default + explicit per-call opt-in + logging is
            # the strongest alternative that keeps the tool usable.
            log_action("api_mutating_allowed", "allowed",
                       f"mutating method explicitly allowed: {method} to {redact_url(url)}")
        result = _http_request(method, url, body=body, headers=headers)
        if "error" in result:
            return {
                "url": url,
                "method": method,
                "error": result["error"],
                "status": "blocked" if "Blocked" in result["error"] else "failed",
            }
        return {
            "url": url,
            "method": method,
            "status_code": result.get("status"),
            "body": (result.get("body") or "")[:2000],
            "status": "success",
        }
    
    raw_results = await asyncio.to_thread(
        run_parallel,
        [(lambda r=req: execute_request(r)) for req in reqs],
        max_workers=min(len(reqs), 10),
        timeout=HTTP_TASK_TIMEOUT_SECONDS,
    )
    results = []
    for raw in raw_results:
        if isinstance(raw, dict) and "url" not in raw:
            results.append({"url": "", "error": raw.get("error", "unknown error"), "status": raw.get("status", "failed")})
        else:
            results.append(raw)
    results, output_truncated = cap_aggregate_output(results, ("body",))

    elapsed = time.time() - start
    return {
        "requests": len(reqs),
        "results": results,
        "output_truncated": output_truncated,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
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
    
    payload_error = validate_parallel_payload(queries, "queries")
    if payload_error:
        return {"error": payload_error}
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
    items_error = validate_parallel_items(q_list, "queries")
    if items_error:
        return {"error": items_error}
    if not all(isinstance(q, str) for q in q_list):
        return {"error": "Each query must be a string"}

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
                    json={"model": MODEL_TIERS["worker"], "prompt": query, "stream": False},
                    timeout=60.0
                )
                if resp.status_code == 200:
                    return {"query": query, "result": resp.json().get("response", ""), "status": "ollama_fallback"}
        except Exception as e:
            return {"query": query, "error": str(e), "status": "failed"}
        
        return {"query": query, "error": "No response", "status": "failed"}
    
    # Cap workers at 8 like exec_swarm/file_swarm: the old
    # max_workers=len(q_list) let a long query list spike threads.
    raw_results = await asyncio.to_thread(
        run_parallel,
        [(lambda q=q: query_kmkb(q)) for q in q_list],
        max_workers=min(len(q_list), 8),
        timeout=HTTP_TASK_TIMEOUT_SECONDS,
    )
    results = []
    for raw in raw_results:
        if isinstance(raw, dict) and "query" not in raw:
            results.append({"query": "", "error": raw.get("error", "unknown error"), "status": raw.get("status", "failed")})
        else:
            results.append(raw)

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
        
        synthesis = await asyncio.to_thread(sync_query_raw, MODEL_TIERS["synthesizer"], synth_prompt)
    
    elapsed = time.time() - start
    return {
        "queries": len(q_list),
        "results": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


# ============ TOOL-ENABLED AGENTS ============

@mcp.tool()
@track_tool_run()
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
    payload_error = validate_parallel_payload(tasks, "tasks")
    if payload_error:
        return {"error": payload_error}
    try:
        task_list = json.loads(tasks)
    except (json.JSONDecodeError, TypeError):
        task_list = [{"prompt": tasks}]

    items_error = validate_parallel_items(task_list, "tasks")
    if items_error:
        return {"error": items_error}

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
    
    # Run remaining tasks through bugs, off the event loop. Also
    # validate the task list: an empty list is an error, not a
    # no-op swarm, and over-long lists are rejected.
    if not task_list:
        return {"error": "tasks must not be empty"}
    bug_results = await asyncio.to_thread(run_tool_swarm, bug_tasks, colony.bugs, deep) if bug_tasks else []
    results = direct_results + bug_results

    
    elapsed = time.time() - start
    colony.tasks_completed += len(task_list)
    stats["tasks_executed"] += len(task_list)
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = await asyncio.to_thread(synthesize_swarm_results, results, task_context=f"Tool swarm with {colony_type} colony")
    
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
@track_tool_run()
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
    colony_id = _unique_id("health")
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
    
    results = await asyncio.to_thread(run_tool_swarm, tasks, bugs)
    elapsed = time.time() - start
    
    # Summarize health
    issues = []
    for r in results:
        if "error" in r.get("answer", "").lower() or "warning" in r.get("answer", "").lower():
            issues.append(r.get("answer", ""))
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = await asyncio.to_thread(synthesize_swarm_results, results, task_context="System health check")
    
    return {
        "verdict": "healthy" if not issues else "issues_found",
        "issues": issues,
        "checks": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
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
    
    colony_id = _unique_id("recon")
    
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
    
    results = await asyncio.to_thread(run_tool_swarm, tasks, bugs, deep)
    elapsed = time.time() - start
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = await asyncio.to_thread(synthesize_swarm_results, results, task_context=f"Reconnaissance of {target_path}")
    
    return {
        "target": target_path,
        "findings": results,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
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
    
    result = await asyncio.to_thread(
        run_tool_agent,
        role="worker",
        model=MODEL_TIERS["worker"],
        task=task,
        context=context,
        deep=True,  # Workers always use deep mode for complex tasks
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
@track_tool_run()
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
@track_tool_run()
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
    
    colony_id = _unique_id("deep")
    
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
    
    results = await asyncio.to_thread(run_tool_swarm, tasks, bugs, True)
    elapsed = time.time() - start
    
    # Extract findings summary
    findings = []
    for r in results:
        if r.get("status") == "completed" and r.get("answer"):
            findings.append(r["answer"])
    
    # Synthesize if requested
    synthesis = None
    if synthesize:
        synthesis = await asyncio.to_thread(synthesize_swarm_results, results, task_context=f"Deep {analysis_type} analysis of {target_path}")
    
    return {
        "target": target_path,
        "analysis_type": analysis_type,
        "findings": results,
        "summary": findings,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


@mcp.tool()
@track_tool_run()
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
    
    synthesis = await asyncio.to_thread(synthesize_swarm_results, result_list, task_context=context)
    elapsed = time.time() - start
    
    return {
        "input_count": len(result_list),
        "model": MODEL_TIERS["synthesizer"],
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }


