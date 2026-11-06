"""
Agent Farm Tool System v3.4 - Role-Based Tool Access
Each bug role gets specific tools they can use.

Security model (summary -- see README "Security model" for the full
version): filesystem writes only inside an allowlist and reads
blocked for system/sensitive paths (canonical, symlink-resolved);
HTTP only to public http/https URLs, validated once and then
connected to that exact validated IP (IP pinning), with redirects
re-validated hop by hop; exec with no shell at all (simple commands
or safe pipelines only); api_swarm GET-only unless a call explicitly
opts into mutation; parallel work bounded by reliability.py limits.
Escape hatches for local testing are OFF by default, are ignored
entirely when AGENT_FARM_PRODUCTION=1, and their use is logged when
action logging is enabled. This server is for local, trusted use
only -- it is not a sandbox and must not be exposed to untrusted
prompts, files, web pages, or other users.
"""

import os
import re
import json
import shlex
import socket
import ssl
import http.client
import ipaddress
import subprocess
import httpx
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Callable, Optional
from .action_log import log_action, redact_url
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
# exec_cmd() never runs a shell. Simple commands run shell=False, and
# the only complex form allowed is a safe pipeline (see
# _parse_safe_pipeline): stages joined by a single '|', each stage a
# plain argument list, with no chaining, redirection, substitution,
# backgrounding, or shell expansion. A substring blocklist is kept as
# defence-in-depth only -- it is no longer the thing standing between
# a prompt-injected command and the shell, because there is no shell.
# Even so, exec_cmd still runs arbitrary binaries with the server's
# privileges, so this server must only run locally, with trusted
# prompts, and must NOT be exposed to untrusted files, web pages, or
# other users' prompts as if this were a sandbox. It is not one.
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


# Directories file_swarm and code_review_swarm must never read from, even
# though tool_read_file alone would allow some of them (it only blocks
# specific sensitive files, not whole system directories). This is the
# single list both swarm tools use, so a swarm tool cannot bypass it by
# keeping its own copy of the list.
FILE_SWARM_BLOCKED_DIRS = ["/etc", "/usr", "/bin", "/sbin", "/boot", "/root", "/sys", "/proc"]


def is_swarm_read_allowed(path: str) -> bool:
    """True iff a swarm tool may read this canonical path.

    Requires is_path_safe() AND not inside any FILE_SWARM_BLOCKED_DIRS
    entry. Canonical (symlink-resolved) paths are used, so a symlink in
    an allowed spot pointing at a blocked spot does not pass.
    """
    if not is_path_safe(path):
        return False
    canonical = _canonical_path(path)
    return not any(
        canonical == os.path.realpath(d) or _is_within(canonical, os.path.realpath(d))
        for d in FILE_SWARM_BLOCKED_DIRS
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


# Maximum stages in a safe pipeline. Pipelines like
# "find /tmp -type f | sort | head -20" need only a few stages, and a
# cap stops a request from spawning an unbounded process chain.
MAX_PIPELINE_STAGES = 5

# Shell syntax that is never allowed in exec_cmd, even inside a pipeline:
# chaining/backgrounding, redirection, and substitution. A single '|'
# between stages is the only operator a pipeline may contain.
_FORBIDDEN_SHELL_CHARS = set(";<>$`&\n\r")


def _parse_safe_pipeline(cmd: str) -> Optional[List[List[str]]]:
    """Parse `cmd` as a safe pipeline, or return None if it is not one.

    A safe pipeline is 2..MAX_PIPELINE_STAGES stages joined by a single
    '|'. Each stage must shlex-split into a non-empty argument list and
    must pass the command blocklist. No other shell syntax is allowed:
    '||', '|&', ';', '&&', '&', '>', '<', '$', backticks, and
    newlines are all rejected, even inside quotes, because quoting does
    not make substitution or redirection safe here. Glob characters
    ('*', '?') are passed through literally -- they are NOT expanded
    by a shell, so "ls *.py" will not behave like it does in a shell;
    use a tool that expands patterns itself, or list a directory.
    """
    if not cmd or not cmd.strip():
        return None
    # Split on unquoted '|' and reject every other shell operator.
    stages_raw: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if ch in _FORBIDDEN_SHELL_CHARS:
            return None
        if ch == "|":
            # '||' and '|&' are chaining, not a pipeline.
            if i + 1 < len(cmd) and cmd[i + 1] in ("|", "&"):
                return None
            stages_raw.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    if in_single or in_double:
        return None
    stages_raw.append("".join(current))
    if len(stages_raw) < 2 or len(stages_raw) > MAX_PIPELINE_STAGES:
        return None
    stages: List[List[str]] = []
    for raw in stages_raw:
        if not raw.strip():
            return None
        try:
            parts = shlex.split(raw)
        except ValueError:
            return None
        if not parts:
            return None
        # Per-stage blocklist check, in addition to the whole-command
        # check in tool_exec_cmd, so a blocked command cannot hide as
        # a later pipeline stage.
        if not is_command_safe(" ".join(parts)):
            return None
        stages.append(parts)
    return stages


def _run_pipeline(stages: List[List[str]]) -> Dict:
    """Run argument-array stages piped together, with no shell.

    The first stage reads from /dev/null (never the terminal), each
    later stage reads the previous stage's stdout, and only the last
    stage's stdout/stderr are returned. Earlier stages' stderr goes to
    /dev/null so a noisy early stage cannot deadlock the pipe.
    """
    procs = []
    try:
        prev_stdout = None
        for idx, argv in enumerate(stages):
            is_last = idx == len(stages) - 1
            proc = subprocess.Popen(
                argv,
                shell=False,
                stdin=prev_stdout if prev_stdout is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if is_last else subprocess.DEVNULL,
                text=True,
            )
            if prev_stdout is not None:
                prev_stdout.close()
            procs.append(proc)
            prev_stdout = proc.stdout
        try:
            stdout, stderr = procs[-1].communicate(timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            for proc in procs:
                try:
                    proc.kill()
                except Exception:
                    pass
            for proc in procs:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            return {"error": f"Command timed out ({EXEC_TIMEOUT}s)"}
        return {
            "stdout": (stdout or "")[:MAX_OUTPUT],
            "stderr": (stderr or "")[:1000],
            "returncode": procs[-1].returncode,
        }
    except Exception as e:
        for proc in procs:
            try:
                proc.kill()
            except Exception:
                pass
        return {"error": str(e)}


# ============ HTTP / SSRF GUARDS ============
# http_get/http_post previously accepted any URL, including localhost and
# private-network addresses, which is a classic SSRF hole: a prompt-injected
# URL could reach internal services (metadata endpoints, local admin
# panels, the KMKB port, etc.) and return their responses to the model.
# These helpers fail closed: only public http/https URLs are allowed, and
# redirects are re-validated hop by hop, so a public URL cannot redirect
# the request to a private address.
#
# Escape hatches, off by default, for local testing only:
#   AGENT_FARM_ALLOW_PRIVATE_HTTP=1  -- relax ONLY the private-IP rule.
#     Scheme validation and credential rejection still apply, and the
#     URL must still resolve. Previously this flag returned True
#     before any check, so it also bypassed scheme and credential
#     validation -- that bypass is fixed here.
#   AGENT_FARM_ALLOWED_HTTP_HOSTS=host1,host2 -- allow these hosts even
#     if they resolve privately (e.g. an internal test service).
#     Scheme and credential checks still apply to them, and they must
#     still resolve to an address (previously they were accepted
#     without any DNS check, which made IP pinning impossible).
#
# DNS rebinding defence: validation used to resolve a hostname with
# getaddrinfo() and then let httpx resolve it again on its own, so a
# hostile DNS answer could change between the check and the connection
# (time-of-check / time-of-use). _http_request now resolves once,
# validates every returned address, and then connects to that exact
# validated IP -- see _PinnedHTTPConnection / _PinnedHTTPSConnection.
# The Host header and, for https, the TLS server name stay as the
# original hostname, so certificate verification still checks the
# hostname, not the IP. Redirects go through the same validate-then-
# pin step on every hop, so a redirect cannot slip in a new,
# unvalidated address either.
#
# Remaining limitation, stated honestly: pinning closes the gap for
# the connection this code makes, but it cannot control DNS answers
# for other clients, and a hostname that legitimately rotates between
# several public IPs will be served by the single IP pinned at
# validation time (a failover IP will not be tried). If the pinned
# address stops answering, the request fails rather than falling back
# to a fresh, unvalidated lookup.

def _allowed_http_hosts() -> set:
    raw = os.environ.get("AGENT_FARM_ALLOWED_HTTP_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _is_non_public_ip(ip: "ipaddress._BaseAddress") -> bool:
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _validate_url(url: str) -> Optional[Dict]:
    """Validate `url` and return its pinned connection info, or None.

    Returns {"parsed": ..., "host": ..., "port": ..., "pinned_ip": ...,
    "private_allowed": ...} on success. Scheme and credential checks
    are mandatory and run before any escape hatch is consulted, so
    neither escape hatch can turn a non-http(s) URL or a
    credential-bearing URL into an allowed one. A hostname must
    resolve, and every address it resolves to must be public unless
    a private-IP escape hatch applies to this host -- otherwise a
    rotating answer could still hand the connection a private
    address. The pinned IP is the first resolved address.

    Safe defaults: both escape hatches are OFF by default (the
    allow-private flag only counts when it is exactly "1"), and
    when AGENT_FARM_PRODUCTION=1 is set, BOTH escape hatches are
    ignored entirely -- production mode cannot be talked out of
    the private-IP rule by a leftover test setting. Use
    escape_hatches_active() / get_security_config() to inspect
    the current configuration.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    # Mandatory, escape-hatch-proof checks first.
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.username or parsed.password:
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    production_mode = os.environ.get("AGENT_FARM_PRODUCTION") == "1"
    private_allowed = (
        not production_mode
        and (
            os.environ.get("AGENT_FARM_ALLOW_PRIVATE_HTTP") == "1"
            or host in _allowed_http_hosts()
        )
    )
    if (host in ("localhost",) or host.endswith((".localhost", ".local", ".internal"))) \
            and not private_allowed:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    # IP literals must be checked directly, not via DNS (a mocked or
    # poisoned resolver must not be able to launder 127.0.0.1 into a
    # public address).
    try:
        literal = ipaddress.ip_address(host)
        if _is_non_public_ip(literal) and not private_allowed:
            return None
        return {"parsed": parsed, "host": host, "port": port, "pinned_ip": str(literal),
                "private_allowed": private_allowed}
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return None
    if not infos:
        return None
    ips: List[str] = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except Exception:
            return None
        if _is_non_public_ip(ip) and not private_allowed:
            return None
        if str(ip) not in ips:
            ips.append(str(ip))
    if not ips:
        return None
    return {"parsed": parsed, "host": host, "port": port, "pinned_ip": ips[0],
            "private_allowed": private_allowed}


def escape_hatches_active():
    """Return the local-testing escape hatches currently in effect.

    Returns a list of names (possibly empty). In production mode
    (AGENT_FARM_PRODUCTION=1) this always returns [] because both
    hatches are ignored. Defaults are off, so an empty list is the
    normal, safe state -- a non-empty list here is a visible signal
    that this process is running in a relaxed local-testing mode
    and must never be used for production.
    """
    if os.environ.get("AGENT_FARM_PRODUCTION") == "1":
        return []
    active = []
    if os.environ.get("AGENT_FARM_ALLOW_PRIVATE_HTTP") == "1":
        active.append("AGENT_FARM_ALLOW_PRIVATE_HTTP")
    if _allowed_http_hosts():
        active.append("AGENT_FARM_ALLOWED_HTTP_HOSTS")
    return active


def get_security_config():
    """Return the current security configuration in plain terms."""
    return {
        "production_mode": os.environ.get("AGENT_FARM_PRODUCTION") == "1",
        "allow_private_http": os.environ.get("AGENT_FARM_ALLOW_PRIVATE_HTTP") == "1",
        "allowed_http_hosts": sorted(_allowed_http_hosts()),
        "escape_hatches_active": escape_hatches_active(),
        "http_rule": "public http/https only, IP-pinned, redirects re-validated",
        "exec_rule": "no shell; simple commands or safe pipelines only",
        "api_swarm_rule": "GET-only unless a call passes allow_mutating=True",
        "write_rule": "writes only inside the write allowlist",
        "trust": "local, trusted use only; not a sandbox",
    }


def is_url_public(url: str) -> bool:
    """True iff `url` is a public http/https URL safe to fetch.

    Fails closed on DNS failure, non-http(s) schemes, credentials in the
    URL, localhost names, and any resolved address that is private,
    loopback, link-local, multicast, reserved, or unspecified -- unless
    a private-IP escape hatch applies to this host, in which case only
    the private-IP rule is relaxed (scheme and credential checks
    still apply).
    """
    return _validate_url(url) is not None


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects to a validated IP, not a hostname.

    self.host stays as the original hostname, so the Host header is
    correct. connect() deliberately does NOT use socket.create_connection
    (which would call getaddrinfo again and re-open the rebinding gap);
    it opens a socket directly to the pinned IP.
    """

    pinned_ip: str = ""

    def connect(self) -> None:  # type: ignore[override]
        ip = ipaddress.ip_address(self.pinned_ip)
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        self.sock = socket.socket(family, socket.SOCK_STREAM)
        if self.timeout is not None:
            self.sock.settimeout(self.timeout)
        address = (str(ip), self.port, 0, 0) if ip.version == 6 else (str(ip), self.port)
        self.sock.connect(address)  # type: ignore[arg-type]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a validated IP, with hostname TLS.

    As above, the socket goes to the pinned IP, but the TLS handshake
    uses server_hostname=self.host, so the certificate is still
    verified against the original hostname, not the IP.
    """

    pinned_ip: str = ""

    def connect(self) -> None:  # type: ignore[override]
        ip = ipaddress.ip_address(self.pinned_ip)
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        address = (str(ip), self.port, 0, 0) if ip.version == 6 else (str(ip), self.port)
        sock.connect(address)  # type: ignore[arg-type]
        context = self._context or ssl._create_default_https_context()
        self.sock = context.wrap_socket(sock, server_hostname=self.host)


def _fetch_pinned(method: str, validated: Dict, body: str = "",
                  headers: Optional[Dict] = None) -> Dict:
    """Make one request to a validated, pinned address.

    Returns {"status": ..., "headers": ..., "body": ...} (body is
    decoded text, truncated by the caller). Raises on transport errors.
    No DNS lookup happens here: the connection class connects
    directly to validated["pinned_ip"].
    """
    parsed = validated["parsed"]
    scheme = parsed.scheme
    host = validated["host"]
    port = validated["port"]
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn_cls = _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
    conn = conn_cls(host, port, timeout=30.0)
    conn.pinned_ip = validated["pinned_ip"]  # type: ignore[attr-defined]
    try:
        conn.request(method, path, body=body.encode("utf-8") if body else None,
                     headers=dict(headers or {}))
        resp = conn.getresponse()
        raw = resp.read()
        try:
            charset = resp.headers.get_content_charset() or "utf-8"  # type: ignore[attr-defined]
        except Exception:
            charset = "utf-8"
        text = raw.decode(charset, errors="replace")
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return {"status": resp.status, "headers": resp_headers, "body": text}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _http_request(method: str, url: str, body: str = "", headers: Optional[Dict] = None) -> Dict:
    """Fetch a URL after SSRF validation, re-validating redirects.

    This is the single HTTP path for both `http_get`/`http_post` and
    `api_swarm`, so a swarm request cannot bypass the guard by calling
    a client directly. Only GET follows redirects, and each redirect
    target is re-validated and re-pinned. Other methods return the
    redirect response as-is rather than following it blindly.
    """
    method = (method or "GET").upper()
    if method not in ("GET", "POST", "PUT", "DELETE"):
        return {"url": url, "error": f"Unsupported method: {method}"}
    try:
        current_url = url
        for _ in range(4):
            validated = _validate_url(current_url)
            if validated is None:
                log_action("http_blocked", "blocked",
                           f"blocked URL/redirect: {redact_url(current_url)}")
                if current_url == url:
                    return {"url": url, "error": "Blocked URL: only public http/https URLs are allowed"}
                return {"url": url, "error": "Blocked redirect to a non-public URL"}
            if validated.get("private_allowed"):
                log_action("http_escape_hatch", "allowed",
                           "LOCAL TESTING ONLY: private-IP escape hatch used; "
                           "never set in production")
            resp = _fetch_pinned(method, validated, body=body, headers=headers)
            location = resp["headers"].get("location")
            if (method == "GET" and resp["status"] in (301, 302, 303, 307, 308)
                    and location):
                current_url = urljoin(current_url, location)
                continue
            result = {"url": url, "status": resp["status"],
                      "body": resp["body"][:MAX_OUTPUT]}
            if validated.get("private_allowed"):
                result["security_warning"] = (
                    "LOCAL TESTING ONLY: a private-IP escape hatch was "
                    "active for this request; never set it in production"
                )
            return result
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
        log_action("file_blocked", "blocked", f"blocked read path: {path}")
        return {"error": f"Blocked path: {path}"}
    try:
        # Explicit UTF-8: the platform default is UTF-8 on Linux but
        # cp1252 on Windows, so a file written on one side could not
        # be read back as text on the other (smart punctuation came
        # back as replacement characters / invalid UTF-8 bytes).
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(MAX_FILE_SIZE)
        return {"path": path, "content": content, "truncated": len(content) >= MAX_FILE_SIZE}
    except Exception as e:
        return {"error": str(e)}


def tool_write_file(path: str, content: str) -> Dict:
    """Write to a file (with safety limits)"""
    path = _canonical_path(path)
    if not is_path_safe(path):
        log_action("file_blocked", "blocked", f"blocked write path: {path}")
        return {"error": f"Blocked path: {path}"}
    # Only allow writing to safe directories (canonical/commonpath check:
    # "/tmpfoo" is not "/tmp", and symlinks out of the allowed dirs fail).
    if not is_write_path_allowed(path):
        log_action("file_blocked", "blocked", f"blocked write outside allowlist: {path}")
        return {"error": f"Write only allowed in: {ALLOWED_WRITE_DIRS}"}
    try:
        # Explicit UTF-8 (see tool_read_file): without this, Windows
        # writes text in cp1252, so smart quotes/dashes in model
        # output became bytes that are not valid UTF-8 on read-back.
        with open(path, 'w', encoding='utf-8') as f:
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
    """Execute a command, with no shell involved.

    Allowed forms, and only these:
    - A simple command: one argument list, run shell=False.
    - A safe pipeline: stages joined by a single '|', each stage an
      argument list, run as argument-array processes piped together
      (see _parse_safe_pipeline).

    Everything else is rejected: chaining (';', '&&', '||'),
    redirection ('>', '<'), substitution ('$()', backticks, '${}'),
    backgrounding ('&'), and multi-line commands. This is an
    intentional breaking change for commands that previously ran
    with shell=True: use separate exec_cmd calls instead of '&&' /
    '||', and use write_file instead of shell redirection. This is
    still NOT a sandbox -- the allowed binaries run with the server's
    privileges -- so run locally with trusted prompts only.
    """
    if "\n" in cmd or "\r" in cmd:
        log_action("exec_blocked", "blocked", "blocked multi-line command")
        return {"error": "Blocked multi-line command"}
    if not is_command_safe(cmd):
        # Log only the binary name, never the full command line --
        # arguments may contain private content.
        binary = (cmd.strip().split() or [""])[0][:80]
        log_action("exec_blocked", "blocked", f"blocked command binary: {binary}")
        return {"error": "Blocked dangerous command pattern"}
    try:
        if _is_simple_command(cmd):
            result = subprocess.run(
                shlex.split(cmd), shell=False, capture_output=True,
                text=True, timeout=EXEC_TIMEOUT
            )
            return {
                "command": cmd,
                "stdout": result.stdout[:MAX_OUTPUT],
                "stderr": result.stderr[:1000],
                "returncode": result.returncode
            }
        stages = _parse_safe_pipeline(cmd)
        if stages is not None:
            result = _run_pipeline(stages)
            if "error" in result:
                return {"error": result["error"]}
            return {
                "command": cmd,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
            }
        binary = (cmd.strip().split() or [""])[0][:80]
        log_action("exec_blocked", "blocked", f"blocked complex command binary: {binary}")
        return {"error": "Blocked complex shell command: only simple commands and "
                         "safe pipelines (stages joined by a single '|', no "
                         "redirection, chaining, substitution, or backgrounding) "
                         "are allowed"}
    except subprocess.TimeoutExpired:
        log_action("exec_timeout", "timeout", f"command timed out ({EXEC_TIMEOUT}s)")
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
