"""
Agent Farm v3.3 - Structured Output Edition
- Ollama structured output for reliable tool parsing
- JSON schema enforcement via constrained decoding
- Regex fallback for edge cases
- Deep mode for multi-step operations
"""

import json
import re
import httpx
from typing import Dict, List, Optional, Any
from .tools import (
    get_tools_for_role, 
    get_tool_descriptions_for_role,
    execute_tool,
    TOOL_REGISTRY
)

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
        with httpx.Client() as client:
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
        with httpx.Client() as client:
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
            tool_info.append("disk_usage (arg: path like /) - check disk space")
        elif t == "check_service":
            tool_info.append("check_service (arg: service name) - check if service running")
        elif t == "exec_cmd":
            tool_info.append("exec_cmd (arg: shell command) - run any shell command")
        elif t == "read_file":
            tool_info.append("read_file (arg: file path) - read file contents")
        elif t == "list_dir":
            tool_info.append("list_dir (arg: directory path) - list directory contents")
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
    
    return f"""You are a {role} bug. Pick ONE tool to call for this task.

AVAILABLE TOOLS:
{tools_desc}

COMMON PATTERNS:
- System health -> system_status
- Check service -> check_service with service name
- Disk space -> disk_usage with path (/)
- Top processes -> process_list with "cpu" or "memory"
- Run command -> exec_cmd with the command
- Read file -> read_file with path
- List folder -> list_dir with path
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

IMPORTANT - WHEN TO USE exec_cmd vs list_dir:
- list_dir: ONLY for seeing what files exist in a directory
- exec_cmd: For EVERYTHING ELSE including:
  * Finding file sizes: exec_cmd(du -sh /path)
  * Counting files: exec_cmd(find /path -name "*.py" | wc -l)
  * Finding large files: exec_cmd(find /path -size +10M)
  * Searching content: exec_cmd(grep -r "pattern" /path)
  * Getting disk usage: exec_cmd(df -h)
  * Complex queries: exec_cmd(find /path -name "__pycache__" -exec du -sh {} \\;)

When task asks for sizes, counts, or analysis - USE exec_cmd with shell commands!
"""

    return f"""You are a {role} bug. You MUST call tools to get real data. Do NOT make up answers.

TOOLS:
{tools_desc}
{deep_instructions}
QUICK REFERENCE:
- System stats -> system_status()
- Disk space -> disk_usage(/) or exec_cmd(df -h)
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

def _summarize_result(result: Dict) -> str:
    """Summarize a tool result for history"""
    if "error" in result:
        return f"Error: {result['error']}"
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
                    
                    # Check for error
                    if isinstance(tool_result, dict) and "error" in tool_result:
                        # Continue to next iteration with error context
                        context = f"Previous tool error: {tool_result['error']}. Try different approach."
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
    """Smart routing - match task to appropriate bug role"""
    prompt = task.get("prompt", "").lower()
    
    # Write operations need workers
    write_keywords = ["write", "create file", "save to", "output to", "append to"]
    if any(kw in prompt for kw in write_keywords):
        workers = [b for b in bugs if b.role == "worker"]
        if workers:
            return workers[0]
    
    # HTTP operations need workers  
    http_keywords = ["http", "api", "fetch", "request", "post to", "get from"]
    if any(kw in prompt for kw in http_keywords):
        workers = [b for b in bugs if b.role == "worker"]
        if workers:
            return workers[0]
    
    # Knowledge base queries need memory bugs
    kb_keywords = ["kmkb", "knowledge", "search knowledge", "ask knowledge"]
    if any(kw in prompt for kw in kb_keywords):
        memory = [b for b in bugs if b.role == "memory"]
        if memory:
            return memory[0]
    
    # System monitoring needs guardians
    monitor_keywords = ["monitor", "health check", "system status", "check service"]
    if any(kw in prompt for kw in monitor_keywords):
        guardians = [b for b in bugs if b.role == "guardian"]
        if guardians:
            return guardians[0]
    
    # Default: worker if available, else first bug
    workers = [b for b in bugs if b.role == "worker"]
    if workers:
        return workers[0]
    return bugs[0]


def run_tool_swarm(tasks: List[Dict], bugs: List, deep: bool = False) -> List[Dict]:
    """Run tool-enabled agents in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    
    # Smart routing - match tasks to appropriate bugs
    bug_assignments = []
    used_bugs = set()
    for task in tasks:
        bug = get_best_bug_for_task(task, bugs)
        if bug.id in used_bugs and len(bugs) > 1:
            alternatives = [b for b in bugs if b.id not in used_bugs]
            if alternatives:
                same_role = [b for b in alternatives if b.role == bug.role]
                bug = same_role[0] if same_role else alternatives[0]
        used_bugs.add(bug.id)
        bug_assignments.append((bug, task))
    
    with ThreadPoolExecutor(max_workers=len(bugs)) as executor:
        futures = {
            executor.submit(
                run_tool_agent,
                bug.role, bug.model,
                task.get("prompt", ""),
                task.get("context", ""),
                deep=deep
            ): (bug, task)
            for bug, task in bug_assignments
        }
        
        for future in as_completed(futures):
            bug, task = futures[future]
            try:
                result = future.result()
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "model": bug.model,
                    "task": task.get("prompt", "")[:50],
                    "answer": result["answer"],
                    "tool_calls": len(result["tool_calls"]),
                    "iterations": result["iterations"],
                    "status": result["status"],
                    "mode": result.get("mode", "unknown")
                })
            except Exception as e:
                results.append({
                    "bug_id": bug.id,
                    "role": bug.role,
                    "error": str(e),
                    "status": "failed"
                })
    
    return results
