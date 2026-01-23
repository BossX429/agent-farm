"""
Agent Farm Tool System v3.1 - Role-Based Tool Access
Each bug role gets specific tools they can use.
Speed optimized with expanded permissions.
"""

import os
import json
import subprocess
import httpx
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

# Role-specific tool permissions
# All roles get core system tools to ensure they can answer system questions
ROLE_TOOLS = {
    "scout": ["read_file", "list_dir", "file_exists", "system_status", "process_list", "disk_usage", "check_service", "exec_cmd"],
    "worker": ["read_file", "write_file", "list_dir", "exec_cmd", "http_get", "http_post", "system_status", "disk_usage", "check_service"],
    "memory": ["read_file", "kmkb_search", "kmkb_ask", "list_dir", "system_status", "process_list", "disk_usage", "check_service", "exec_cmd"],
    "guardian": ["system_status", "process_list", "disk_usage", "check_service", "read_file", "list_dir", "exec_cmd"],
    "learner": ["read_file", "analyze_code", "list_dir", "kmkb_search", "system_status", "process_list", "disk_usage", "check_service", "exec_cmd"],
}

# Safety limits
MAX_FILE_SIZE = 100000  # 100KB max read
MAX_OUTPUT = 10000     # 10KB max output
EXEC_TIMEOUT = 60      # 60 second max for commands

# Blocked paths for file operations
BLOCKED_PATHS = ["/etc/shadow", "/etc/passwd", "/root", "/boot", "/sys", "/proc/kcore"]

# Blocked command patterns
BLOCKED_COMMANDS = [
    "rm -rf", "rm -r /", "sudo", "dd if=", "mkfs", "chmod 777",
    ":(){ :|:& };:", "> /dev/sd", "mv /* ", "shutdown", "reboot",
    "init 0", "init 6", "kill -9 1", "pkill -9", "killall"
]


def is_path_safe(path: str) -> bool:
    """Check if path is safe to access"""
    abs_path = os.path.abspath(path)
    for blocked in BLOCKED_PATHS:
        if abs_path.startswith(blocked):
            return False
    return True


def is_command_safe(cmd: str) -> bool:
    """Check if command is safe to execute"""
    cmd_lower = cmd.lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return False
    return True


# ============ TOOL IMPLEMENTATIONS ============

def expand_path(path: str) -> str:
    """Expand ~ and environment variables in path"""
    return os.path.expanduser(os.path.expandvars(path))


def tool_read_file(path: str) -> Dict:
    """Read a file (with safety limits)"""
    path = expand_path(path)
    if not is_path_safe(path):
        return {"error": f"Blocked path: {path}"}
    try:
        with open(path, 'r') as f:
            content = f.read(MAX_FILE_SIZE)
        return {"path": path, "content": content, "truncated": len(content) >= MAX_FILE_SIZE}
    except Exception as e:
        return {"error": str(e)}


def tool_write_file(path: str, content: str) -> Dict:
    """Write to a file (with safety limits)"""
    path = expand_path(path)
    if not is_path_safe(path):
        return {"error": f"Blocked path: {path}"}
    # Only allow writing to safe directories
    allowed_dirs = ["/tmp", "/home/kyle/repos", "/home/kyle/Desktop"]
    abs_path = os.path.abspath(path)
    if not any(abs_path.startswith(d) for d in allowed_dirs):
        return {"error": f"Write only allowed in: {allowed_dirs}"}
    try:
        with open(path, 'w') as f:
            f.write(content)
        return {"path": path, "bytes_written": len(content), "status": "success"}
    except Exception as e:
        return {"error": str(e)}


def tool_list_dir(path: str) -> Dict:
    """List directory contents"""
    path = expand_path(path)
    if not is_path_safe(path):
        return {"error": f"Blocked path: {path}"}
    try:
        items = os.listdir(path)
        return {"path": path, "items": items[:100], "count": len(items)}
    except Exception as e:
        return {"error": str(e)}


def tool_file_exists(path: str) -> Dict:
    """Check if file exists"""
    path = expand_path(path)
    return {"path": path, "exists": os.path.exists(path)}


def tool_exec_cmd(cmd: str) -> Dict:
    """Execute a shell command (with safety checks)"""
    if not is_command_safe(cmd):
        return {"error": f"Blocked dangerous command pattern"}
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=EXEC_TIMEOUT
        )
        return {
            "command": cmd,
            "stdout": result.stdout[:MAX_OUTPUT],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out ({EXEC_TIMEOUT}s)"}
    except Exception as e:
        return {"error": str(e)}


def tool_http_get(url: str) -> Dict:
    """HTTP GET request"""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            return {"url": url, "status": resp.status_code, "body": resp.text[:MAX_OUTPUT]}
    except Exception as e:
        return {"error": str(e)}


def tool_http_post(url: str, body: str = "") -> Dict:
    """HTTP POST request"""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, content=body)
            return {"url": url, "status": resp.status_code, "body": resp.text[:MAX_OUTPUT]}
    except Exception as e:
        return {"error": str(e)}


def tool_system_status() -> Dict:
    """Get basic system status"""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get CPU temp if available
        temps = psutil.sensors_temperatures()
        cpu_temp = None
        if temps:
            for name, entries in temps.items():
                if entries:
                    cpu_temp = entries[0].current
                    break
        
        return {
            "cpu_percent": cpu,
            "cpu_temp": cpu_temp,
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
    except Exception as e:
        return {"error": str(e)}


def tool_process_list(sort_by: str = "cpu") -> Dict:
    """List top processes"""
    try:
        import psutil
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                procs.append(p.info)
            except:
                pass
        
        key = 'cpu_percent' if sort_by == 'cpu' else 'memory_percent'
        procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=True)
        return {"processes": procs[:15], "count": len(procs)}
    except Exception as e:
        return {"error": str(e)}


def tool_disk_usage(path: str = "/") -> Dict:
    """Get disk usage for a path"""
    path = expand_path(path)
    try:
        import psutil
        usage = psutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent": usage.percent
        }
    except Exception as e:
        return {"error": str(e)}


def tool_check_service(name: str) -> Dict:
    """Check if a systemd service is running"""
    result = tool_exec_cmd(f"systemctl is-active {name}")
    if "error" in result:
        return result
    status = result.get("stdout", "").strip()
    return {"service": name, "status": status, "running": status == "active"}


def tool_kmkb_search(query: str) -> Dict:
    """Search KMKB knowledge base"""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "http://127.0.0.1:8765/search",
                json={"query": query, "top_k": 5}
            )
            if resp.status_code == 200:
                return {"query": query, "results": resp.json()}
    except:
        pass
    return {"error": "KMKB not available", "query": query}


def tool_kmkb_ask(question: str) -> Dict:
    """Ask KMKB a question"""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "http://127.0.0.1:8765/ask",
                json={"question": question, "top_k": 3}
            )
            if resp.status_code == 200:
                return {"question": question, "answer": resp.json()}
    except:
        pass
    return {"error": "KMKB not available", "question": question}


def tool_analyze_code(code: str) -> Dict:
    """Basic code analysis"""
    lines = code.split('\n')
    return {
        "lines": len(lines),
        "characters": len(code),
        "has_functions": "def " in code or "function " in code,
        "has_classes": "class " in code,
        "has_imports": "import " in code or "from " in code or "require(" in code,
        "has_comments": "#" in code or "//" in code or "/*" in code
    }


# ============ TOOL REGISTRY ============

TOOL_REGISTRY: Dict[str, Callable] = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
    "file_exists": tool_file_exists,
    "exec_cmd": tool_exec_cmd,
    "http_get": tool_http_get,
    "http_post": tool_http_post,
    "system_status": tool_system_status,
    "process_list": tool_process_list,
    "disk_usage": tool_disk_usage,
    "check_service": tool_check_service,
    "kmkb_search": tool_kmkb_search,
    "kmkb_ask": tool_kmkb_ask,
    "analyze_code": tool_analyze_code,
}

# Tool descriptions for the LLM
TOOL_DESCRIPTIONS = {
    "read_file": "read_file(path) - Read contents of a file",
    "write_file": "write_file(path, content) - Write content to a file. IMPORTANT: Quote content with spaces/commas. Example: write_file(\"/tmp/test.txt\", \"Hello, World!\")",
    "list_dir": "list_dir(path) - List files in a directory",
    "file_exists": "file_exists(path) - Check if a file exists",
    "exec_cmd": "exec_cmd(cmd) - Execute a shell command",
    "http_get": "http_get(url) - Make HTTP GET request",
    "http_post": "http_post(url, body) - Make HTTP POST request",
    "system_status": "system_status() - Get CPU, memory, disk status",
    "process_list": "process_list(sort_by) - List top processes (sort_by: cpu or memory)",
    "disk_usage": "disk_usage(path) - Get disk usage for a path",
    "check_service": "check_service(name) - Check if systemd service is running",
    "kmkb_search": "kmkb_search(query) - Search knowledge base",
    "kmkb_ask": "kmkb_ask(question) - Ask knowledge base a question",
    "analyze_code": "analyze_code(code) - Basic code structure analysis",
}


def get_tools_for_role(role: str) -> List[str]:
    """Get list of tools available to a role"""
    return ROLE_TOOLS.get(role, [])


def get_tool_descriptions_for_role(role: str) -> str:
    """Get formatted tool descriptions for a role"""
    tools = get_tools_for_role(role)
    descriptions = [TOOL_DESCRIPTIONS[t] for t in tools if t in TOOL_DESCRIPTIONS]
    return "\n".join(f"  - {d}" for d in descriptions)


def execute_tool(tool_name: str, role: str, **kwargs) -> Dict:
    """Execute a tool if the role has permission"""
    allowed_tools = get_tools_for_role(role)
    
    if tool_name not in allowed_tools:
        return {"error": f"Tool '{tool_name}' not permitted for role '{role}'"}
    
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        handler = TOOL_REGISTRY[tool_name]
        return handler(**kwargs)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def parse_tool_call(response: str) -> tuple:
    """
    Parse a tool call from bug response.
    Expected format: TOOL: tool_name(arg1, arg2, ...)
    Returns: (tool_name, kwargs) or (None, None) if no tool call
    """
    import re
    
    # Look for TOOL: pattern
    match = re.search(r'TOOL:\s*(\w+)\((.*?)\)', response, re.IGNORECASE)
    if not match:
        return None, None
    
    tool_name = match.group(1)
    args_str = match.group(2).strip()
    
    # Parse arguments
    kwargs = {}
    if args_str:
        # Handle simple cases: single arg or key=value pairs
        if '=' in args_str:
            # key=value format
            for part in args_str.split(','):
                if '=' in part:
                    key, val = part.split('=', 1)
                    kwargs[key.strip()] = val.strip().strip('"\'')
        else:
            # Positional args - try to map to first parameter
            # Get the tool's first parameter name
            handler = TOOL_REGISTRY.get(tool_name)
            if handler:
                import inspect
                sig = inspect.signature(handler)
                params = list(sig.parameters.keys())
                if params:
                    kwargs[params[0]] = args_str.strip('"\'')
    
    return tool_name, kwargs
