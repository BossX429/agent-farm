"""
Agent Farm v3.4 - Structured Output Edition
- Ollama structured output for reliable tool parsing
- JSON schema enforcement via constrained decoding
- Regex fallback for edge cases
- Deep mode for multi-step operations
"""

import json
import os
import re
import time
import httpx
from typing import Dict, List, Optional, Any
from .tools import (
    get_tools_for_role,
    get_tool_descriptions_for_role,
    execute_tool,
    TOOL_REGISTRY
)
from .colony_profiles import record_colony_run
from .profiles import (
    TASK_TYPE_ROLES,
    classify_result,
    profile_pick,
    record_profile_run,
    task_type_for_prompt,
)
from .reliability import MODEL_SEMAPHORE

OLLAMA_URL = "http://localhost:11434"
MAX_TOOL_ITERATIONS = 5


# ============ JSON SCHEMAS FOR STRUCTURED OUTPUT ============

# Schema for tool calls - Ollama will enforce this structure
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["tool", "answer"]
        },
        "tool_name": {
            "type": "string"
        },
        "tool_args": {
            "type": "object"
        },
        "answer": {
            "type": "string"
        }
    },
    "required": ["action"]
}

# Simpler schema for just getting a tool call
SIMPLE_TOOL_SCHEMA = {
    "type": "object", 
    "properties": {
        "tool": {"type": "string"},
        "arg": {"type": "string"}
    },
    "required": ["tool"]
}


# ============ OLLAMA QUERY FUNCTIONS ============

def query_ollama_sync(model: str, prompt: str) -> str:
    """Synchronous Ollama query - freeform text response"""
    try:
        with MODEL_SEMAPHORE, httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120.0
            )
            if resp.status_code == 200:
                return resp.json().get("response", "No response")
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def query_ollama_structured(model: str, prompt: str, schema: dict = None) -> Dict[str, Any]:
    """
    Query Ollama with structured JSON output.
    Uses constrained decoding to GUARANTEE valid JSON matching schema.
    
    Args:
        model: Ollama model name
        prompt: The prompt to send
        schema: JSON schema to enforce (uses SIMPLE_TOOL_SCHEMA if None)
    
    Returns:
        Parsed JSON dict, or {"error": "..."} on failure
    """
    if schema is None:
        schema = SIMPLE_TOOL_SCHEMA

    try:
        with MODEL_SEMAPHORE, httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": schema,  # Enforces JSON schema via grammar
                    "options": {"temperature": 0.1}  # Low temp for deterministic output
                },
                timeout=120.0
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "{}")
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"error": f"Invalid JSON: {content[:200]}"}
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ============ PROMPT BUILDERS ============

def build_structured_prompt(role: str, task: str, context: str = "") -> str:
    """Build prompt optimized for structured JSON output"""
    tools = get_tools_for_role(role)
    tools_list = ", ".join(tools)
    
    # Build tool descriptions with arg names
    tool_info = []
    for t in tools:
        if t == "system_status":
            tool_info.append("system_status (no args) - get CPU/memory/disk stats")
        elif t == "process_list":
            tool_info.append("process_list (arg: cpu or memory) - list top processes")
        elif t == "disk_usage":
            tool_info.append("disk_usage (arg: path like /) - check disk space. Write the path with forward slashes, e.g. C:/Users/kyleh/agent-farm - never backslashes")
        elif t == "check_service":
            tool_info.append("check_service (arg: service name) - check if service running")
        elif t == "exec_cmd":
            tool_info.append("exec_cmd (arg: shell command) - run any shell command")
        elif t == "search_code":
            tool_info.append("search_code (arg: pattern) - search INSIDE files for a text pattern, in the working directory. USE WHEN you know the text to find. DO NOT USE to discover what files exist - use list_dir first")
        elif t == "read_file":
            tool_info.append("read_file (arg: file path) - read file contents")
        elif t == "list_dir":
            tool_info.append("list_dir (arg: directory path) - list the files in a directory. USE WHEN you need to discover what files exist. DO NOT USE to search inside files - use search_code for that. Write the path with forward slashes, e.g. C:/Users/kyleh/agent-farm - never backslashes")
        elif t == "write_file":
            tool_info.append("write_file (arg: path) - write to file (content separate)")
        elif t == "http_get":
            tool_info.append("http_get (arg: url) - HTTP GET request")
        elif t == "kmkb_search":
            tool_info.append("kmkb_search (arg: query) - search knowledge base")
        elif t == "kmkb_ask":
            tool_info.append("kmkb_ask (arg: question) - ask knowledge base")
        else:
            tool_info.append(f"{t}")
    
    tools_desc = "\n".join(f"  - {t}" for t in tool_info)
    
    context_line = f"\nCONTEXT: {context}" if context else ""
    # The model cannot see where it is: the PR #30 live proof
    # had it list a hallucinated placeholder path five times.
    # Tell it the working directory, which is the repo root when
    # the server runs from the repo, and to use it as the base
    # for list_dir, read_file, and search_code.
    workdir_line = f"\nWORKING DIRECTORY: {os.getcwd().replace(chr(92), '/')} - use this as the base path for list_dir, read_file, and search_code. Do not invent paths."

    return f"""You are a {role} bug. Pick ONE tool to call for this task.
{workdir_line}
AVAILABLE TOOLS:
{tools_desc}

COMMON PATTERNS:
- System health -> system_status
- Check service -> check_service with service name
- Disk space -> disk_usage with path (/)
- Top processes -> process_list with "cpu" or "memory"
- Search file contents -> search_code with the pattern (use this, not grep: grep does not exist on Windows)
- Run command -> exec_cmd with the command
- Read file -> read_file with path
- List folder -> list_dir with path

PATHS: Always write Windows paths with forward slashes, e.g. C:/Users/kyleh/agent-farm - never backslashes, never a trailing slash. Backslashes break JSON and produce paths Windows rejects.

EXAMPLES:
- TASK: "List the Python files in the agent_farm folder" -> list_dir with C:/Users/kyleh/agent-farm/agent_farm (list_dir first - search_code searches INSIDE files, it cannot list them)
- TASK: "Where is the synthesizer model named?" -> search_code with pattern synthesizer (the text is known, so search inside files)
- TASK: "List the colony types agent-farm supports" -> search_code with pattern COLONY_TYPES (the types are defined in code, not in a folder - search for the constant name in capitals, then read the file it is in)
- If a tool returns no results or an error, do NOT answer "no results found". Try a different tool or different arguments first - the tool, the pattern, or the path may be wrong, not the answer.
{context_line}
TASK: {task}

Respond with JSON: {{"tool": "tool_name", "arg": "argument_value"}}
For tools with no args, use empty string: {{"tool": "system_status", "arg": ""}}"""


def build_continuation_structured(task: str, history: List[Dict]) -> str:
    """Build prompt for deciding next action after tool results"""
    # Summarize history
    history_lines = []
    for h in history[-3:]:
        summary = _summarize_result(h['result'])
        history_lines.append(f"- {h['tool']}({h.get('args', {})}) -> {summary}")
    
    history_str = "\n".join(history_lines)
    
    # Get last stdout if any
    last_result = history[-1]['result'] if history else {}
    stdout = ""
    if isinstance(last_result, dict) and last_result.get('stdout'):
        stdout = f"\nLast output:\n{last_result['stdout'][:500]}"
    
    return f"""Previous tool calls:
{history_str}
{stdout}

TASK: {task}

Do you have enough info to answer? 
- If YES: {{"action": "answer", "answer": "your complete answer with data"}}
- If NO: {{"action": "tool", "tool": "tool_name", "arg": "argument"}}"""


def build_agent_prompt(role: str, task: str, context: str = "", deep: bool = False) -> str:
    """Build prompt - deep mode enables multi-step reasoning (legacy freeform)"""
    tools_desc = get_tool_descriptions_for_role(role)
    
    deep_instructions = ""
    if deep:
        deep_instructions = """
DEEP WORK MODE - You may need MULTIPLE tool calls to complete this task.
After each tool result, decide: Do I have enough info to answer? If NO, call another tool.

IMPORTANT - WHEN TO USE exec_cmd vs the native tools:
- list_dir: ONLY for seeing what files exist in a directory
- search_code: for searching file contents (use this, NOT grep:
  grep, find, du, and df do not exist on Windows, and exec_cmd
  runs with no shell, so those commands always fail)
- disk_usage: for disk space (use this, NOT df -h)
- exec_cmd: only for commands that exist on this machine.
  Bare `git` is resolved to the direct binary when AGENT_FARM_GIT
  is set; otherwise use the full path to the binary.

When a task asks for sizes, counts, or analysis, use the native
tool above first and exec_cmd only for what it cannot do.
"""

    return f"""You are a {role} bug. You MUST call tools to get real data. Do NOT make up answers.

TOOLS:
{tools_desc}
{deep_instructions}
QUICK REFERENCE:
- System stats -> system_status()
- Disk space -> disk_usage(/)
- Service check -> check_service(name)
- Top processes -> process_list(cpu)
- Shell command -> exec_cmd(command)  <-- USE THIS FOR COMPLEX QUERIES
- Read file -> read_file(/path)
- List dir contents -> list_dir(/path)

{f"CONTEXT: {context}" if context else ""}
TASK: {task}

Call a tool now. Write TOOL: followed by the tool call:
TOOL:"""


def build_answer_prompt(task: str, tool_name: str, result: Dict) -> str:
    """Force immediate answer after getting data - MUST use actual data"""
    result_str = json.dumps(result, indent=2)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "..."
    
    # Extract stdout for emphasis if present
    stdout_emphasis = ""
    if isinstance(result, dict) and result.get("stdout"):
        stdout = result["stdout"].strip()
        if stdout:
            stdout_emphasis = f"\n\nACTUAL DATA FROM COMMAND:\n{stdout[:1000]}\n"
    
    return f"""You called {tool_name} and got this result:
{result_str}
{stdout_emphasis}
TASK: {task}

CRITICAL: Your answer MUST include the actual data/numbers from the result above.
- If the result shows files/directories, LIST THEM with their sizes
- If the result shows a count, STATE THE NUMBER
- If the result is empty, say "no results found"
- NEVER make up data - only use what's in the result

ANSWER:"""


# ============ RESULT PROCESSING ============

def _is_empty_result(result: Dict) -> bool:
    """True when a tool result carries no usable information.

    An empty result is not an answer: search_code with 0 matches,
    list_dir with 0 items, an error, or empty stdout all mean the
    agent tried the wrong tool, the wrong arguments, or the wrong
    path -- not that nothing exists. The loop retries a different
    tool instead of forcing an answer from this.
    """
    if not isinstance(result, dict):
        return True
    if "error" in result:
        return True
    if "total_matches" in result:
        return result.get("total_matches", 0) == 0
    if "items" in result:
        return result.get("count", len(result.get("items", []))) == 0
    if "stdout" in result:
        return not result.get("stdout", "").strip()
    if "content" in result:
        return not result.get("content", "").strip()
    return False


def _summarize_result(result: Dict) -> str:
    """Summarize a tool result for history"""
    if "error" in result:
        return f"Error: {result['error']}"
    if "total_matches" in result and result.get("total_matches", 0) == 0:
        return "EMPTY: 0 matches - this does NOT prove nothing exists, the pattern, folder, or tool may be wrong"
    if "items" in result and result.get("count", len(result.get("items", []))) == 0:
        return "EMPTY: 0 items - this does NOT prove the folder is empty, the path or tool may be wrong"
    if "stdout" in result:
        out = result.get('stdout', '')[:200]
        return out if out else "(no output)"
    if "items" in result:
        return f"{result.get('count', 0)} items"
    if "content" in result:
        return f"{len(result.get('content', ''))} chars"
    if "cpu_percent" in result:
        return f"CPU:{result['cpu_percent']}% MEM:{result.get('memory_percent')}%"
    return str(result)[:200]


def format_result_simple(result: Dict) -> Optional[str]:
    """Format ONLY for simple system queries - NOT directory listings"""
    if "error" in result:
        return None
    
    # System status - ok to auto-format
    if "cpu_percent" in result:
        return f"CPU: {result.get('cpu_percent')}%, Temp: {result.get('cpu_temp')}C, Memory: {result.get('memory_percent')}%, Disk: {result.get('disk_percent')}%"
    
    # Process list - ok to auto-format
    if "processes" in result:
        procs = result.get("processes", [])[:5]
        return "Top processes: " + ", ".join([
            f"{p.get('name')} ({p.get('memory_percent', p.get('cpu_percent', 0)):.1f}%)" 
            for p in procs
        ])
    
    # Service check - ok to auto-format
    if "service" in result:
        status = result.get('status', 'unknown')
        running = "running" if status == "active" else status
        return f"{result.get('service')} is {running}"
    
    # Disk usage - ok to auto-format
    if "free_gb" in result:
        return f"{result.get('free_gb')}GB free ({result.get('percent')}% used)"
    
    return None


def clean_answer(answer: str) -> str:
    """Clean up answer - strip markdown, code blocks, thinking artifacts"""
    if not answer:
        return answer
    
    # Remove markdown code blocks
    answer = re.sub(r'```\w*\n?', '', answer)
    answer = re.sub(r'```', '', answer)
    
    # Remove ** bold markers
    answer = re.sub(r'\*\*', '', answer)
    
    # Remove thinking tags sometimes added by models
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<thinking>.*?</thinking>', '', answer, flags=re.DOTALL)
    
    # Remove leading/trailing whitespace and newlines
    answer = answer.strip()
    
    if not answer:
        return "No valid answer extracted"
    
    return answer


# ============ TOOL CALL PARSING ============

def parse_structured_response(response: Dict, role: str) -> tuple:
    """
    Parse structured JSON response into (tool_name, kwargs) or (None, answer)
    
    Args:
        response: Dict from query_ollama_structured
        role: Bug role for tool validation
    
    Returns:
        (tool_name, kwargs) for tool calls
        (None, answer_str) for answers
        (None, None) for parse failures
    """
    if "error" in response:
        return None, None
    
    valid_tools = set(TOOL_REGISTRY.keys())
    allowed_tools = set(get_tools_for_role(role))
    
    # Handle action-based schema (continuation prompts)
    if response.get("action") == "answer":
        answer = response.get("answer", "")
        return None, answer if answer else None
    
    # Handle tool call
    tool_name = response.get("tool", response.get("tool_name", "")).lower().strip()
    
    if not tool_name or tool_name not in valid_tools:
        return None, None
    
    if tool_name not in allowed_tools:
        return None, None
    
    # Get argument
    arg = response.get("arg", response.get("tool_args", ""))
    
    # Handle tool_args as dict
    if isinstance(arg, dict):
        return tool_name, arg
    
    # Convert single arg to appropriate kwarg
    arg = str(arg).strip() if arg else ""
    
    # Map to correct parameter name
    param_map = {
        'read_file': 'path', 'write_file': 'path', 'list_dir': 'path',
        'disk_usage': 'path', 'file_exists': 'path', 'exec_cmd': 'cmd',
        'check_service': 'name', 'http_get': 'url', 'http_post': 'url',
        'kmkb_search': 'query', 'kmkb_ask': 'question',
        'process_list': 'sort_by', 'analyze_code': 'code',
        'search_code': 'pattern',
    }
    
    kwargs = {}
    if arg:
        param = param_map.get(tool_name)
        if param:
            kwargs[param] = arg
    
    return tool_name, kwargs


def parse_tool_call_regex(response: str) -> tuple:
    """
    FALLBACK: Parse tool call using regex - handles freeform text responses.
    Used when structured output fails or for legacy compatibility.
    """
    valid_tools = set(TOOL_REGISTRY.keys())
    
    param_map = {
        'read_file': 'path', 'write_file': 'path', 'list_dir': 'path',
        'disk_usage': 'path', 'file_exists': 'path', 'exec_cmd': 'cmd',
        'check_service': 'name', 'http_get': 'url', 'http_post': 'url',
        'kmkb_search': 'query', 'kmkb_ask': 'question',
        'process_list': 'sort_by', 'analyze_code': 'code',
        'search_code': 'pattern',
    }
    
    # Try JSON first (model might output JSON even without schema enforcement)
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            tool = data.get("tool", "").lower().strip()
            arg = data.get("arg", "")
            if tool in valid_tools:
                param = param_map.get(tool)
                kwargs = {param: arg} if param and arg else {}
                return tool, kwargs
    except:
        pass
    
    # FORMAT 1: TOOL: name\nARGS: value
    args_match = re.search(r'TOOL:\s*(\w+)\s*\n\s*ARGS?:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if args_match:
        tool_name = args_match.group(1).lower()
        if tool_name in valid_tools:
            args_str = args_match.group(2).strip().split('\n')[0].strip()
            param = param_map.get(tool_name)
            if param and args_str:
                return tool_name, {param: args_str}
    
    # FORMAT 2: TOOL: name(args)
    match = re.search(r'TOOL:\s*(\w+)\s*\(([^)]*)\)', response, re.IGNORECASE)
    if match:
        tool_name = match.group(1).lower()
        if tool_name in valid_tools:
            args_str = match.group(2).strip().strip('"\'')
            param = param_map.get(tool_name)
            kwargs = {param: args_str} if param and args_str else {}
            return tool_name, kwargs
    
    # FORMAT 3: TOOL: name value
    match = re.search(r'TOOL:\s*(\w+)\s+([^\n]+)', response, re.IGNORECASE)
    if match and match.group(1).lower() in valid_tools:
        tool_name = match.group(1).lower()
        args_str = match.group(2).strip()
        param = param_map.get(tool_name)
        if param and args_str:
            return tool_name, {param: args_str}
    
    # FORMAT 4: TOOL: name (no args)
    match = re.search(r'TOOL:\s*(\w+)(?:\s|$)', response, re.IGNORECASE)
    if match and match.group(1).lower() in valid_tools:
        return match.group(1).lower(), {}
    
    return None, None


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from freeform response"""
    match = re.search(r'ANSWER:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return clean_answer(answer)
    return None


# ============ MAIN AGENT RUNNER ============

def run_tool_agent(role: str, model: str, task: str, context: str = "", deep: bool = False) -> Dict:
    """
    Run a tool-enabled agent with structured output.
    
    Strategy:
    1. Try structured JSON output first (reliable parsing)
    2. Fall back to regex parsing if structured fails
    3. Auto-format simple system queries
    4. Deep mode enables multi-step reasoning
    
    Args:
        role: Bug role (worker, scout, guardian, etc.)
        model: Ollama model name
        task: Task description
        context: Optional additional context
        deep: Enable multi-step mode
    
    Returns:
        Dict with answer, tool_calls, iterations, status
    """
    tool_history = []
    use_structured = True  # Start with structured, fall back if needed
    
    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        
        # === TRY STRUCTURED OUTPUT FIRST ===
        if use_structured:
            prompt = build_structured_prompt(role, task, context)
            response = query_ollama_structured(model, prompt, SIMPLE_TOOL_SCHEMA)
            
            # Check for structured output error
            if "error" in response:
                # Fall back to regex mode
                use_structured = False
            else:
                tool_name, result = parse_structured_response(response, role)
                
                # Got an answer?
                if tool_name is None and isinstance(result, str) and result:
                    # Completion guard: a tool agent must not answer
                    # from nothing. If no tool has returned usable
                    # data yet, the answer is a guess - nudge it to
                    # call a tool instead. On the last iteration the
                    # answer is accepted so the loop terminates.
                    has_data = any(
                        not _is_empty_result(h["result"])
                        for h in tool_history
                    )
                    if not has_data and iteration < MAX_TOOL_ITERATIONS:
                        tried = {h["tool"] for h in tool_history}
                        untried = [t for t in get_tools_for_role(role) if t not in tried]
                        context = (
                            "You have not found anything yet - you "
                            "cannot answer without tool data. Call a "
                            "tool. You have not yet tried: "
                            + ", ".join(untried) + "."
                        )
                        continue
                    return {
                        "answer": clean_answer(result),
                        "tool_calls": tool_history,
                        "iterations": iteration,
                        "status": "completed",
                        "mode": "structured"
                    }
                
                # Got a valid tool call?
                if tool_name:
                    kwargs = result if isinstance(result, dict) else {}
                    tool_result = execute_tool(tool_name, role, **kwargs)
                    
                    tool_history.append({
                        "tool": tool_name,
                        "args": kwargs,
                        "result": tool_result
                    })
                    
                    
                    # Empty result (or error): do NOT force an answer
                    # from nothing. Nudge the model to try a different
                    # tool (or different arguments) and keep looping
                    # while iterations remain. An empty result proves
                    # nothing. On the last iteration, fall through to
                    # the forced-answer path below instead of looping
                    # past the end.
                    if _is_empty_result(tool_result) and iteration < MAX_TOOL_ITERATIONS:
                        tried = {h["tool"] for h in tool_history}
                        untried = [t for t in get_tools_for_role(role) if t not in tried]
                        why = ""
                        if isinstance(tool_result, dict) and "error" in tool_result:
                            why = " Error: " + str(tool_result["error"]) + "."
                        context = (
                            "That tool returned no results." + why + " This "
                            "does NOT prove nothing exists - the tool, the "
                            "arguments, or the path may be wrong. Try a "
                            "different tool or different arguments. You have "
                            "not yet tried: " + ", ".join(untried) + ". Do not "
                            "answer 'no results found' until you have tried "
                            "at least two different tools."
                        )
                        continue

                    # Anti-repeat: the same tool with the same arguments
                    # three times means the model is stuck - tell it so
                    # instead of letting it burn the remaining iterations.
                    same_calls = [
                        h for h in tool_history
                        if h["tool"] == tool_name and h.get("args", {}) == kwargs
                    ]
                    if len(same_calls) >= 3 and iteration < MAX_TOOL_ITERATIONS:
                        context = (
                            "You have called " + tool_name + " with the same "
                            "arguments three times. Stop repeating it - try a "
                            "different tool or different arguments."
                        )
                        continue

                    # Auto-format simple results
                    auto_answer = format_result_simple(tool_result)
                    if auto_answer and not deep:
                        return {
                            "answer": auto_answer,
                            "tool_calls": tool_history,
                            "iterations": iteration,
                            "status": "completed",
                            "mode": "structured+autoformat"
                        }
                    
                    # Deep mode or complex result - ask for continuation
                    if deep:
                        cont_prompt = build_continuation_structured(task, tool_history)
                        cont_schema = {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["tool", "answer"]},
                                "tool": {"type": "string"},
                                "arg": {"type": "string"},
                                "answer": {"type": "string"}
                            },
                            "required": ["action"]
                        }
                        cont_response = query_ollama_structured(model, cont_prompt, cont_schema)
                        
                        if cont_response.get("action") == "answer":
                            return {
                                "answer": clean_answer(cont_response.get("answer", "")),
                                "tool_calls": tool_history,
                                "iterations": iteration,
                                "status": "completed",
                                "mode": "structured+deep"
                            }
                        elif cont_response.get("action") == "tool":
                            # Next iteration will use the new tool
                            continue
                    else:
                        # Quick mode - force answer from result
                        answer_prompt = build_answer_prompt(task, tool_name, tool_result)
                        final_response = query_ollama_sync(model, answer_prompt)
                        answer = final_response.strip()
                        if answer.upper().startswith("ANSWER:"):
                            answer = answer[7:].strip()
                        return {
                            "answer": clean_answer(answer),
                            "tool_calls": tool_history,
                            "iterations": iteration + 1,
                            "status": "completed", 
                            "mode": "structured+forced"
                        }
                    
                    continue
        
        # === FALLBACK TO REGEX MODE ===
        prompt = build_agent_prompt(role, task, context, deep=deep)
        response = query_ollama_sync(model, prompt)
        
        # Check for answer
        answer = extract_answer(response)
        if answer and tool_history:
            return {
                "answer": answer,
                "tool_calls": tool_history,
                "iterations": iteration,
                "status": "completed",
                "mode": "regex"
            }
        
        # Try to parse tool call
        tool_name, kwargs = parse_tool_call_regex(response)
        if tool_name:
            result = execute_tool(tool_name, role, **kwargs)
            
            tool_history.append({
                "tool": tool_name,
                "args": kwargs,
                "result": result
            })
            
            if isinstance(result, dict) and "error" in result:
                context = f"Tool error: {result['error']}"
                continue
            
            auto_answer = format_result_simple(result)
            if auto_answer and not deep:
                return {
                    "answer": auto_answer,
                    "tool_calls": tool_history,
                    "iterations": iteration,
                    "status": "completed",
                    "mode": "regex+autoformat"
                }
            
            # Force answer
            answer_prompt = build_answer_prompt(task, tool_name, result)
            final_response = query_ollama_sync(model, answer_prompt)
            answer = final_response.strip()
            if answer.upper().startswith("ANSWER:"):
                answer = answer[7:].strip()
            return {
                "answer": clean_answer(answer),
                "tool_calls": tool_history,
                "iterations": iteration + 1,
                "status": "completed",
                "mode": "regex+forced"
            }
        
        # No valid tool - nudge
        tools_list = ", ".join(get_tools_for_role(role))
        context = f"You must call a tool. Available: {tools_list}"
    
    # Max iterations
    final_answer = "Task incomplete - max iterations reached"
    if tool_history:
        final_answer = f"Partial results from {len(tool_history)} tool calls: "
        final_answer += "; ".join([_summarize_result(h['result']) for h in tool_history])
    
    return {
        "answer": final_answer,
        "tool_calls": tool_history,
        "iterations": MAX_TOOL_ITERATIONS,
        "status": "partial" if tool_history else "failed",
        "mode": "timeout"
    }


# ============ TASK ROUTING ============

def get_best_bug_for_task(task: Dict, bugs: List) -> 'Bug':
    """Smart routing - match task to appropriate bug role.

    Uses the shared keyword table (profiles.TASK_TYPE_KEYWORDS),
    the same table colony.get_best_bug_for_task routes with and
    the profiler records with, so all three agree on what a
    prompt is. Measured profile evidence comes first, the same
    as on the colony path. The final fallback is the first
    worker, then any bug.
    """
    task_type = task_type_for_prompt(task.get("prompt", ""))
    picked = profile_pick(task_type, bugs)
    if picked is not None:
        return picked
    role = TASK_TYPE_ROLES.get(task_type)
    if role:
        matches = [b for b in bugs if b.role == role]
        if matches:
            return matches[0]
    # Default: worker if available, else first bug
    workers = [b for b in bugs if b.role == "worker"]
    if workers:
        return workers[0]
    return bugs[0]

def run_tool_swarm(tasks: List[Dict], bugs: List, deep: bool = False, colony_type: Optional[str] = None) -> List[Dict]:
    """Run tool-enabled agents in parallel.

    Tasks with ``path`` + ``content`` are direct file writes: they are
    executed with write_file immediately (any length), instead of being
    handed to an LLM as an empty prompt -- which previously meant short
    write tasks were silently dropped. All other tasks go to bugs as before.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []

    # Direct-execute write tasks first; they need no LLM round-trip.
    # write_file is only permitted for the worker role, so use that role
    # for the permission check regardless of which bug was assigned.
    direct_tasks = []
    bug_tasks = []
    for task in tasks or []:
        if isinstance(task, dict) and task.get("path") and "content" in task:
            direct_tasks.append(task)
        else:
            bug_tasks.append(task)
    for task in direct_tasks:
        result = execute_tool(
            "write_file", "worker", path=task["path"], content=task["content"]
        )
        results.append({
            "bug_id": "direct-execute",
            "role": "worker",
            "model": "none",
            "task": f"write_file({task['path']})",
            "answer": (
                f"Wrote {len(task['content'])} chars to {task['path']}"
                if "error" not in result else result.get("error")
            ),
            "tool_calls": 1,
            "iterations": 1,
            "status": "completed" if "error" not in result else "failed",
            "mode": "direct",
            "direct_execute": True,
            # No bug ran this task and no profile was consulted,
            # so the pick was not profile-driven.
            "profiled": False,
        })

    if not bug_tasks:
        # Nothing left for bugs to do (e.g. all tasks were direct writes).
        # Return early: ThreadPoolExecutor(max_workers=0) would raise.
        return results

    if not bugs:
        # No bugs to run tasks on: fail the tasks instead of crashing
        # on an empty bug list or creating a zero-worker pool.
        for task in bug_tasks:
            results.append({
                "bug_id": None,
                "role": None,
                "task": task.get("prompt", "")[:50] if isinstance(task, dict) else "",
                "error": "No bugs available to run this task",
                "status": "failed",
                # No bug existed to pick, so nothing was profiled.
                "profiled": False,
            })
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
    for task in bug_tasks:
        task_type = task_type_for_prompt(task.get("prompt", ""))
        profiled_bug = profile_pick(task_type, bugs)
        bug = get_best_bug_for_task(task, bugs)
        if bug.id in used_bugs and len(bugs) > 1:
            alternatives = [b for b in bugs if b.id not in used_bugs]
            if alternatives:
                same_role = [b for b in alternatives if b.role == bug.role]
                bug = same_role[0] if same_role else alternatives[0]
        used_bugs.add(bug.id)
        profiled = profiled_bug is not None and bug.id == profiled_bug.id
        bug_assignments.append((bug, task, profiled))
    
    with ThreadPoolExecutor(max_workers=max(1, min(len(bugs), 8))) as executor:
        futures = {
            executor.submit(
                run_tool_agent,
                bug.role, bug.model,
                task.get("prompt", ""),
                task.get("context", ""),
                deep=deep
            ): (bug, task, profiled, time.monotonic())
            for bug, task, profiled in bug_assignments
        }

        for future in as_completed(futures):
            bug, task, profiled, started = futures[future]
            # Profiler factory: tool-swarm work is real work, so it
            # feeds the bug's profile card too. The elapsed time
            # covers the whole agent run (model calls plus tool
            # calls), not just the model time. Direct-execute tasks
            # (model "none") and no-bug failures never reach this
            # loop, so they are not recorded: no model ran for them.
            try:
                result = future.result()
                elapsed = time.monotonic() - started
                card_success, failure_kind = classify_result(result["answer"])
                if result["status"] != "completed":
                    card_success = False
                    failure_kind = failure_kind or "error"
                # The "profiled" flag was captured at assignment
                # time and carried here; it is about the routing
                # decision, not the outcome. task_type is still
                # needed below to record this run on the card.
                task_type = task_type_for_prompt(task.get("prompt", ""))
                saved = record_profile_run(
                    bug.role,
                    bug.model,
                    task_type,
                    elapsed,
                    card_success,
                    failure_kind,
                )
                # Colony-type measurement: when the caller knows
                # the colony type, the same run feeds its card
                # too, so tool_swarm work counts alongside
                # quick_swarm work. record_colony_run never
                # raises, so profiling cannot break the swarm.
                colony_saved = (
                    record_colony_run(
                        colony_type,
                        task_type,
                        elapsed,
                        card_success,
                        failure_kind,
                    )
                    if colony_type
                    else False
                )
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "model": bug.model,
                    "task": task.get("prompt", "")[:50],
                    "answer": result["answer"],
                    "tool_calls": len(result["tool_calls"]),
                    "iterations": result["iterations"],
                    "status": result["status"],
                    "mode": result.get("mode", "unknown"),
                    "elapsed_seconds": round(elapsed, 2),
                    "profile_saved": saved,
                    "colony_profile_saved": colony_saved,
                    "profiled": profiled,
                })
            except Exception as e:
                elapsed = time.monotonic() - started
                task_type = task_type_for_prompt(task.get("prompt", ""))
                saved = record_profile_run(
                    bug.role,
                    bug.model,
                    task_type,
                    elapsed,
                    False,
                    "error",
                )
                colony_saved = (
                    record_colony_run(
                        colony_type, task_type, elapsed, False, "error"
                    )
                    if colony_type
                    else False
                )
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "error": str(e),
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                    "profile_saved": saved,
                    "colony_profile_saved": colony_saved,
                    "profiled": profiled,
                })

    return results
