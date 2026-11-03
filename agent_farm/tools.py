"""
Agent Farm Tool System v3.4 - Role-Based Tool Access
Each bug role gets specific tools they can use.
Speed optimized with expanded permissions.
"""

import os
import re
import json
import shlex
import socket
import ipaddress
import subprocess
import httpx
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Callable, Optional
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

# Blocked paths for file operations.
# These are matched against the *canonical* (symlink-resolved) path, not the
# raw string, so "../" tricks, symlinks, and prefix look-alikes such as
# "/etc/passwd-backup" handling are not confused with the blocked entries.
BLOCKED_PATHS = ["/etc/shadow", "/etc/passwd", "/etc/gshadow", "/etc/sudoers",
                 "/root", "/boot", "/sys", "/proc/kcore"]

# Directories writes are allowed in (see tool_write_file). Matched with
# commonpath() against the canonical path, so "/tmpfoo" does NOT pass as
# "/tmp", and a symlink inside /tmp pointing elsewhere does not pass either.
ALLOWED_WRITE_DIRS = ["/tmp", "/home/kyle/repos", "/home/kyle/Desktop"]

# SECURITY NOTE -- READ BEFORE RELAXING THIS LIST:
# exec_cmd() runs with shell=True on LLM-generated commands. That is
# inherently risky: a substring blocklist can only reduce accidental damage,
# it cannot make shell execution safe against a determined or prompt-injected
# command (quoting, $(), backticks, alternate spellings, and new utilities
# can all evade it). This server must therefore only run locally, with
# trusted prompts, and must NOT be exposed to untrusted files, web pages,
# or other users' prompts as if this list were a sandbox. It is not one.
# Blocked command patterns (matched after lower-casing and collapsing
# whitespace, so "rm  -rf" and "rm\t-rf" are caught too).
BLOCKED_COMMANDS = [
    # destructive removal / overwrite
    "rm -rf", "rm -r", "rm --recursive", "rm --force", "rm -fr", "rm -f /",
    "rm -r /", "unlink", "shred", "wipe", "truncate -s 0 /",
    "mv /* ", "mv /", "cp /dev/null",
    # privilege escalation / ownership
    "sudo", "su -", "su ", "doas", "pkexec", "runuser",
    "chown", "chgrp", "chmod 777", "chmod -r 777", "chmod 000",
    # disk / device destruction
    "dd ", "dd if=", "mkfs", "mkswap", "fdisk", "parted",
    "> /dev/sd", "> /dev/mem", "> /dev/null 2", "of=/dev/sd",
    # process / system shutdown
    "shutdown", "reboot", "poweroff", "halt", "init 0", "init 6",
    "systemctl poweroff", "systemctl reboot",
    "kill -9 1", "kill -9", "killall", "pkill", "pkill -9",
    # fork bombs and pipe-to-shell (classic prompt-injection payloads)
    ":(){", ":(){ :|:& };:", "| sh", "| bash", "| zsh", "| dash",
    "curl|sh", "curl | sh", "wget|sh", "wget | sh",
    "iptables -f", "iptables --flush",
    # command substitution / evaluation -- these turn a "harmless" command
    # into an arbitrary second command, so they are blocked outright.
    "$(", "`", "${", "eval ", "exec ", "source ",
]


def _canonical_path(path: str) -> str:
    """Expand ~ / env vars and resolve symlinks and '..' components.

    realpath() resolves the parts that exist on disk (including a parent
    directory that is a symlink) even when the final file does not exist
    yet, which is exactly what write-path checks need.
    """
    return os.path.realpath(expand_path(path))


def _is_within(path: str, directory: str) -> bool:
    """True iff canonical `path` is `directory` itself or inside it."""
    try:
        return os.path.commonpath([path, directory]) == directory
    except ValueError:
        return False


def is_path_safe(path: str) -> bool:
    """Check if a canonical path is safe to access."""
    canonical = _canonical_path(path)
    for blocked in BLOCKED_PATHS:
        blocked_canonical = os.path.realpath(blocked)
        if canonical == blocked_canonical or _is_within(canonical, blocked_canonical):
            return False
    return True


def is_write_path_allowed(path: str) -> bool:
    """True iff the canonical path is inside an ALLOWED_WRITE_DIRS entry."""
    canonical = _canonical_path(path)
    return any(
        canonical == os.path.realpath(d) or _is_within(canonical, os.path.realpath(d))
        for d in ALLOWED_WRITE_DIRS
    )


def is_command_safe(cmd: str) -> bool:
    """Check if command is safe to execute (blocklist only -- see note above)."""
    # Collapse runs of whitespace so "rm  -rf" cannot evade "rm -rf".
    cmd_lower = re.sub(r"\s+", " ", cmd.lower())
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return False
    return True


# Characters that make a command need a real shell (pipes, redirection,
# chaining, substitution, globbing via the shell). A command without any
# of these is run with shell=False, so shell metacharacters in its
# arguments cannot be interpreted at all.
_SHELL_METACHARS = set(";|&$`><\n\r")


def _is_simple_command(cmd: str) -> bool:
    """True iff `cmd` is a single argv-style command with no shell syntax."""
    if not cmd or not cmd.strip():
        return False
    if any(ch in _SHELL_METACHARS for ch in cmd):
        return False
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return False
    return bool(parts)


# ============ HTTP / SSRF GUARDS ============
# http_get/http_post previously accepted any URL, including localhost and
# private-network addresses, which is a classic SSRF hole: a prompt-injected
# URL could reach internal services (metadata endpoints, local admin
# panels, the KMKB port, etc.) and return their responses to the model.
# These helpers fail closed: only public http/https URLs are allowed, and
# redirects are re-validated hop by hop, so a public URL cannot redirect
# the request to a private address.
#
# Escape hatch, off by default, for local testing only:
#   AGENT_FARM_ALLOW_PRIVATE_HTTP=1  -- allow private/loopback URLs
#   AGENT_FARM_ALLOWED_HTTP_HOSTS=host1,host2 -- allow these hosts even
#     if they resolve privately (e.g. an internal test service)

def _allowed_http_hosts() -> set:
    raw = os.environ.get("AGENT_FARM_ALLOWED_HTTP_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def is_url_public(url: str) -> bool:
    """True iff `url` is a public http/https URL safe to fetch.

    Fails closed on DNS failure, non-http(s) schemes, credentials in the
    URL, localhost names, and any resolved address that is private,
    loopback, link-local, multicast, reserved, or unspecified.
    """
    if os.environ.get("AGENT_FARM_ALLOW_PRIVATE_HTTP") == "1":
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost",) or host.endswith((".localhost", ".local", ".internal")):
        return False
    if host in _allowed_http_hosts():
        return True
    # IP literals must be checked directly, not via DNS (a mocked or
    # poisoned resolver must not be able to launder 127.0.0.1 into a
    # public address).
    try:
        literal = ipaddress.ip_address(host)
        return not (literal.is_private or literal.is_loopback or literal.is_link_local
                    or literal.is_multicast or literal.is_reserved or literal.is_unspecified)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except Exception:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


def _http_request(method: str, url: str, body: str = "") -> Dict:
    """Fetch a URL after SSRF validation, re-validating redirects."""
    if not is_url_public(url):
        return {"url": url, "error": "Blocked URL: only public http/https URLs are allowed"}
    try:
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            current_url = url
            for _ in range(4):
                if method == "POST":
                    resp = client.post(current_url, content=body)
                else:
                    resp = client.get(current_url)
                location = resp.headers.get("location")
                if (method == "GET" and resp.status_code in (301, 302, 303, 307, 308)
                        and location):
                    next_url = urljoin(current_url, location)
                    if not is_url_public(next_url):
                        return {"url": url, "error": "Blocked redirect to a non-public URL"}
                    current_url = next_url
                    continue
                return {"url": url, "status": resp.status_code,
                        "body": resp.text[:MAX_OUTPUT]}
            return {"url": url, "error": "Too many redirects"}
    except Exception as e:
        return {"error": str(e)}


# ============ TOOL IMPLEMENTATIONS ============

def expand_path(path: str) -> str:
    """Expand ~ and environment variables in path"""
    return os.path.expanduser(os.path.expandvars(path))


def tool_read_file(path: str) -> Dict:
    """Read a file (with safety limits)"""
    path = _canonical_path(path)
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
    path = _canonical_path(path)
    if not is_path_safe(path):
        return {"error": f"Blocked path: {path}"}
    # Only allow writing to safe directories (canonical/commonpath check:
    # "/tmpfoo" is not "/tmp", and symlinks out of the allowed dirs fail).
    if not is_write_path_allowed(path):
        return {"error": f"Write only allowed in: {ALLOWED_WRITE_DIRS}"}
    try:
        with open(path, 'w') as f:
            f.write(content)
        return {"path": path, "bytes_written": len(content), "status": "success"}
    except Exception as e:
        return {"error": str(e)}


def tool_list_dir(path: str) -> Dict:
    """List directory contents"""
    path = _canonical_path(path)
    if not is_path_safe(path):
        return {"error": f"Blocked path: {path}"}
    try:
        items = os.listdir(path)
        return {"path": path, "items": items[:100], "count": len(items)}
    except Exception as e:
        return {"error": str(e)}


def tool_file_exists(path: str) -> Dict:
    """Check if file exists (blocked paths are not probed)."""
    path = _canonical_path(path)
    if not is_path_safe(path):
        return {"path": path, "error": f"Blocked path: {path}"}
    return {"path": path, "exists": os.path.exists(path)}


def tool_exec_cmd(cmd: str) -> Dict:
    """Execute a command.

    Simple commands (no pipes, redirection, chaining, or substitution)
    now run with shell=False, so their arguments cannot be interpreted
    as shell syntax at all. Complex shell commands still use shell=True
    after the blocklist check, so this is still NOT a sandbox -- see the
    note above BLOCKED_COMMANDS. Run locally with trusted prompts only.
    Multi-line commands are rejected outright.
    """
    if "\n" in cmd or "\r" in cmd:
        return {"error": "Blocked multi-line command"}
    if not is_command_safe(cmd):
        return {"error": "Blocked dangerous command pattern"}
    try:
        if _is_simple_command(cmd):
            result = subprocess.run(
                shlex.split(cmd), shell=False, capture_output=True,
                text=True, timeout=EXEC_TIMEOUT
            )
        else:
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
    """HTTP GET request (public URLs only; see is_url_public)."""
    return _http_request("GET", url)


def tool_http_post(url: str, body: str = "") -> Dict:
    """HTTP POST request (public URLs only; see is_url_public)."""
    return _http_request("POST", url, body=body)


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
    path = _canonical_path(path)
    if not is_path_safe(path):
        return {"path": path, "error": f"Blocked path: {path}"}
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
    # A service name is not a shell fragment: reject anything that could
    # inject a second command (e.g. "x; rm ...") before it reaches the shell.
    if not re.fullmatch(r"[A-Za-z0-9_@.\-]+", name or ""):
        return {"service": name, "error": "Invalid service name"}
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


def _split_args_respecting_quotes(args_str: str) -> List[str]:
    """Split on commas that are not inside quotes or nested parentheses."""
    parts: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    for ch in args_str:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            continue
        if not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _unquote(value: str) -> str:
    """Strip one pair of matching surrounding quotes, if present."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _extract_tool_call(response: str):
    """Return (tool_name, args_str) for a TOOL: name(...) call, or (None, None).

    Unlike a non-greedy regex match, this scans for the matching
    closing parenthesis while respecting quotes, so arguments containing
    commas, parentheses, or quoted text survive intact.
    """
    match = re.search(r"TOOL:\s*(\w+)\s*\(", response, re.IGNORECASE)
    if not match:
        return None, None
    tool_name = match.group(1)
    start = match.end()
    in_single = False
    in_double = False
    escaped = False
    depth = 1
    for i in range(start, len(response)):
        ch = response[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return tool_name, response[start:i]
    return None, None


def parse_tool_call(response: str) -> tuple:
    """
    Parse a tool call from bug response.
    Expected format: TOOL: tool_name(arg1, arg2, ...)
    Returns: (tool_name, kwargs) or (None, None) if no tool call

    Parsing order (backwards compatible with the old format):
    1. If the argument string is a JSON object, use it verbatim.
    2. Otherwise split key=value arguments on commas that are *outside*
       quotes, so a value like "Hello, World!" stays whole.
    3. Otherwise map a single positional argument to the tool's first
       parameter, as before.
    """
    tool_name, args_str = _extract_tool_call(response)
    if not tool_name:
        return None, None
    args_str = args_str.strip()
    if not args_str:
        return tool_name, {}

    # 1. JSON object arguments: TOOL: write_file({"path": ..., "content": ...})
    if args_str.startswith("{"):
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                return tool_name, parsed
        except (json.JSONDecodeError, ValueError):
            pass

    kwargs = {}
    # 2. key=value arguments, split without breaking quoted commas.
    if "=" in args_str:
        for part in _split_args_respecting_quotes(args_str):
            if "=" in part:
                key, val = part.split("=", 1)
                kwargs[key.strip()] = _unquote(val)
        if kwargs:
            return tool_name, kwargs

    # 3. Positional argument -> first parameter, as before.
    handler = TOOL_REGISTRY.get(tool_name)
    if handler:
        import inspect
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        if params:
            kwargs[params[0]] = _unquote(args_str)
    return tool_name, kwargs
