# Agent Farm v3.4 - Chunked Write Edition

AI organism evolution and parallel task execution with tool-enabled agents.
Now with **Chunked Write Pattern** for generating large documents and code files!

## What's New in v3.4

- **Chunked Write Pattern**: Bugs write sections in parallel, Python assembles directly
- **chunked_write**: Generate large markdown/text documents (unlimited size)
- **chunked_code_gen**: Generate multi-function code files in parallel
- **chunked_analysis**: Multi-perspective analysis with synthesis
- **Bypasses 500-char limit**: Each bug writes small chunks, combined output is unlimited

## Performance

- Parallel execution via ThreadPoolExecutor (no measured speedup claim is made)
- **1 iteration** per task (was 3-5)
- **Local synthesis** - qwen3:30b-a3b-q3k synthesizes results (no cloud tokens!)

## Security

This server runs commands (`exec_cmd`, `exec_swarm`) and file operations
on the local machine. It is **not a sandbox**: allowed programs still run
with the server's privileges, so it cannot make LLM-generated commands
safe against untrusted input. Run it locally, with trusted prompts only,
and do not expose it to untrusted files, web pages, or other users'
prompts.

## Security model

The guards in place today, in one place:

- **Filesystem**: reads are blocked for system/sensitive paths and
  writes are allowed only inside `/tmp`, `/home/kyle/repos`,
  `/home/kyle/Desktop`, and the platform's system temp directory
  (on Linux that is `/tmp` already; on Windows it is the user's Temp
  folder, and it is only added when it sits under a well-known temp
  location (`/tmp`, `/var/tmp`, or the Windows Temp folders), never a
  home dir, a drive root, or any other folder a TEMP setting points at).
  Paths are checked after resolving symlinks and `..` components, so
  prefix look-alikes and symlink tricks do not bypass them.
- **HTTP / SSRF**: only public http/https URLs are fetched, each
  validated once and then connected to that exact validated IP
  (IP pinning), with redirects re-validated hop by hop. URLs with
  login details in them are always rejected.
- **Exec**: no shell at all. Only simple commands (argument lists)
  or safe pipelines (stages joined by a single `|`, max 5 stages)
  run. Chaining, redirection, substitution, and backgrounding are
  rejected. A command blocklist remains only as defence-in-depth.
- **api_swarm**: GET-only by default. POST/PUT/DELETE need an
  explicit per-call `allow_mutating=True`; there is no server-wide
  flag that silently enables mutation for later calls. Making
  mutation structurally impossible would break this tool's intended
  purpose (calling APIs), so the safe default + per-call opt-in +
  action logging is the design instead.
- **Parallel work**: at most 32 items per request, 100,000-character
  payloads, 50,000-character aggregate output, per-task timeouts,
  and at most 4 simultaneous model calls.
- **Generated code**: `chunked_code_gen` for Python compiles and
  checks the assembled file and reports `validated` or
  `written_not_validated`; validation never runs the code, so it
  is not a claim the code was tested or is correct.

## Deployment: safe defaults

- Defaults are deny/off: private-HTTP escape hatches are off unless
  explicitly set, and `api_swarm` mutation is off unless explicitly
  passed per call.
- Local testing only, **never in production**:
  `AGENT_FARM_ALLOW_PRIVATE_HTTP=1` and
  `AGENT_FARM_ALLOWED_HTTP_HOSTS=host1,host2` relax *only* the
  private-IP rule; scheme and credential checks still apply, and
  allowed hosts must still resolve.
- Set `AGENT_FARM_PRODUCTION=1` in any production-like deployment:
  it ignores both escape hatches entirely, even if a test setting
  was left in the environment.
- Use `tools.get_security_config()` to inspect the current
  configuration, and `tools.escape_hatches_active()` to see
  whether any escape hatch is currently in effect.

## Action log (structured, opt-in)

Set `AGENT_FARM_ACTION_LOG=/path/to/actions.jsonl` (or `=1` for
`./agent-farm-actions.jsonl`) to record one JSON object per line
for enforcement events: blocked paths, blocked URLs, blocked
commands, blocked/allowed mutating API calls, timeouts, and code
validation outcomes. It never logs message bodies, file contents,
generated code, full command lines (binary name only), or URLs
with credentials (those are redacted). Each record carries a
SHA-256 hash chain (`prev_hash` / `hash`); `action_log.verify_action_log(path)`
detects a single record edited in place. That is **not**
tamper-proofing: anyone with write access to the log file could
rewrite the whole chain. The log is off by default.

Hardening in place:
- **HTTP SSRF guard**: `http_get`/`http_post` now allow only public
  http/https URLs. Private, loopback, link-local, and metadata addresses
  (including `127.0.0.1` and `169.254.169.254`) are blocked, DNS failures
  fail closed, and redirects are re-validated hop by hop. Each validated
  URL is then connected to the exact IP that was validated (IP pinning),
  with the original hostname kept for the Host header and, for https,
  TLS certificate verification, so a DNS answer cannot change between
  the check and the connection. For local testing only,
  `AGENT_FARM_ALLOW_PRIVATE_HTTP=1` or
  `AGENT_FARM_ALLOWED_HTTP_HOSTS=host1,host2` can relax ONLY the
  private-IP rule; scheme validation and credential rejection still
  apply, and allowed hosts must still resolve. The internal
  KMKB client (`127.0.0.1:8765`) does not go through these tools.
- **No shell at all**: `exec_cmd` never runs a shell. Simple commands
  run as argument lists, and the only complex form allowed is a safe
  pipeline (stages joined by a single `|`, each stage an argument list,
  max 5 stages). Chaining (`;`, `&&`, `||`), redirection (`>`, `<`),
  substitution (`$(...)`, backticks, `${...}`), backgrounding (`&`),
  and multi-line commands are rejected. This is a breaking change for
  commands that relied on shell chaining or redirection: use separate
  `exec_cmd` calls, and `write_file` instead of shell redirection.
- **Tests and CI**: see Testing below. CI runs compile checks and pytest
  on push and pull requests to `main`.
- **Swarm tools use the same guards**: `exec_swarm` now delegates to
  `tool_exec_cmd` and `api_swarm` now delegates to the shared HTTP
  helper, so neither can bypass the exec or SSRF guards above.
- **Parallel reliability limits**: parallel tools (`exec_swarm`,
  `api_swarm`, `file_swarm`, `kmkb_swarm`, `tool_swarm`,
  `deploy_swarm`, `quick_swarm`) accept at most 32 items per request
  and a 100,000-character payload; an empty or non-array list is
  rejected instead of creating a zero-worker pool. Each task has a
  per-task timeout, and total parallel output is capped at 50,000
  characters (`output_truncated: true` when the cap bites). Model
  calls are gated to at most 4 at once, shared across tools.
- **Blocking work stays off the async event loop**: async tools run
  blocking model, file, and swarm work in threads
  (`asyncio.to_thread`), so one long model call does not stall
  other requests. A per-task timeout means the caller stops waiting;
  it does not kill a thread already running.
- **Generated code is validated, not just written**:
  `chunked_code_gen` for Python now compiles the assembled file and
  reports duplicate top-level function names, syntax errors, and
  failed generation fragments before declaring it complete. A result
  of `validated` means it compiles cleanly; `written_not_validated`
  means the file was saved but did not pass validation (or, for
  non-Python languages, validation is not implemented). Validation
  never runs the generated code, so it is not a claim that the code
  was tested or is correct.
- **Server layout**: `server.py` is now a thin entrypoint. Tools live in
  `state.py` (model tiers, colonies, stats), `ollama.py`, `colony.py`,
  `colony_tools.py`, `swarm_tools.py`, and `chunked_tools.py`. The 25
  MCP tool names are unchanged, and `server.py` re-exports them.

## Testing

```bash
pip install .[dev]
python -m pytest -q
```

Tests in `tests/test_tools.py` cover path safety and symlink traversal,
write-allowlist prefix lookalikes, exec blocklist variants and
shell-free execution, HTTP SSRF guards (with mocked DNS), tool-call
parsing (JSON, quoted commas, parentheses), and ID uniqueness.
Tests in `tests/test_chatgpt_top_security.py` cover the ChatGPT-review
fixes: the private-HTTP override, DNS rebinding / IP pinning (including
a local-server test where DNS flips after validation), and safe
pipelines vs. rejected shell operators.
Tests in `tests/test_chatgpt_medium_reliability.py` cover the
ChatGPT-review medium findings: parallel input limits and output
caps, per-task timeouts, the shared model-concurrency gate, Python
code validation (syntax, duplicates, possible missing imports),
written-vs-validated code generation, and that a slow blocking
planner does not stall the async event loop.

## Models

| Role | Model | VRAM | Purpose |
|------|-------|------|---------|
| Scout | qwen3:4b | 2.5GB | Reconnaissance |
| Worker | qwen2.5-coder:7b-v8-review | 8.1GB | Task execution (code-tuned) |
| Memory | qwen3:4b | 2.5GB | Context retention |
| Guardian | qwen3:4b | 2.5GB | System monitoring |
| Learner | qwen2.5-coder:7b-v8-review | 8.1GB | Pattern acquisition (code-tuned) |
| **Synthesizer** | **qwen3:30b-a3b-q3k** | **14.7GB** | **Result synthesis** |

Code review, code generation, and KMKB synthesis use the synthesizer tier,
not a separate model.

## MCP Tools (25)

### Colony Management
- `spawn_colony` - Create bug colony (standard/fast/heavy/hybrid)
- `list_colonies` - List active colonies
- `colony_status` - Detailed colony info
- `quick_colony` - Quick health check
- `dissolve_colony` - Remove colony
- `cleanup_idle` - Remove idle colonies
- `farm_stats` - Comprehensive statistics

### Swarm Deployment
- `deploy_swarm` - Deploy tasks to colony
- `quick_swarm` - One-shot spawn + deploy

### Specialized Swarms
- `code_review_swarm` - 4-perspective code review
- `code_gen_swarm` - Generate code + tests + docs
- `file_swarm` - Parallel file operations
- `exec_swarm` - Parallel shell commands
- `api_swarm` - Parallel HTTP requests
- `kmkb_swarm` - Multi-angle knowledge queries

### Tool-Enabled Agents
- `tool_swarm` - Deploy bugs with real system tools
- `system_health_swarm` - Quick system health check
- `recon_swarm` - Directory/codebase reconnaissance
- `deep_analysis_swarm` - Deep disk/file analysis
- `worker_task` - Single worker with full tools

### Direct Operations
- `heavy_write` - Direct file write (bypasses LLM for large content)
- `synthesize` - Standalone synthesis of any JSON results

### Chunked Write Pattern (NEW)
- `chunked_write` - Generate large documents via parallel section writing
- `chunked_code_gen` - Generate code files with functions written in parallel
- `chunked_analysis` - Multi-perspective analysis with synthesis

## Bug Tool Permissions

| Role | Tools |
|------|-------|
| Scout | read_file, list_dir, file_exists, system_status, process_list, disk_usage, check_service, exec_cmd |
| Worker | read_file, write_file, list_dir, exec_cmd, http_get, http_post, system_status, disk_usage, check_service |
| Memory | read_file, kmkb_search, kmkb_ask, list_dir, system_status, process_list, disk_usage, check_service, exec_cmd |
| Guardian | system_status, process_list, disk_usage, check_service, read_file, list_dir, exec_cmd |
| Learner | read_file, analyze_code, list_dir, kmkb_search, system_status, process_list, disk_usage, check_service, exec_cmd |

## Structured Output Details

Agent Farm v3.3 uses Ollama's structured output feature to enforce JSON schemas on model responses:

```python
# Bug responds with guaranteed-valid JSON:
{"tool": "system_status", "arg": ""}
{"tool": "exec_cmd", "arg": "df -h"}
{"tool": "check_service", "arg": "ollama"}
```

The constrained decoding (GBNF grammar) masks invalid tokens during generation, ensuring:
- Always valid JSON
- Correct tool names
- Proper argument structure
- No parsing failures

Results now include a `mode` field showing which method was used:
- `structured` - JSON schema enforced
- `structured+autoformat` - JSON + simple result formatting
- `structured+deep` - JSON with multi-step reasoning
- `regex` - Fallback regex parsing
- `regex+autoformat` - Regex + simple result formatting

## Chunked Write Pattern

The chunked write pattern solves the ~500 char output limitation of small models by decomposing large tasks:

```
1. PLANNER BUG (qwen2.5-coder:7b-v8-review)
   |-- Creates structured JSON outline
   |-- {"sections": [{"title": "...", "description": "..."}]}

2. WORKER BUGS (qwen2.5-coder:7b-v8-review) - IN PARALLEL
   |-- Each writes one section (~300-500 chars)
   |-- 4 workers = 4 sections simultaneously

3. PYTHON CONCATENATION (NO LLM)
   |-- header + separator.join(sections)
   |-- Zero token cost, instant assembly

4. DIRECT FILE WRITE (NO LLM)
   |-- tool_write_file() saves result
   |-- Bypasses any output corruption
```

### Performance
| Tool | Output Size | Sections | Time |
|------|-------------|----------|------|
| chunked_write | 9.6 KB | 5 | 78s |
| chunked_code_gen | 1.9 KB | 4 functions | 88s |
| chunked_analysis | Varies | 4 perspectives | ~60s |

### Why It Works
- Small models excel at focused, short outputs
- Each section is within the "safe zone" (<500 chars)
- Python handles assembly (no LLM token cost)
- Parallel execution via ThreadPoolExecutor
- Structured output ensures reliable planning

## Usage Examples

### System Health Check
```
agent-farm:system_health_swarm
```

### Custom Task Swarm
```
agent-farm:tool_swarm
  colony_type: "heavy"
  tasks: [
    {"prompt": "Check CPU temperature"},
    {"prompt": "List top 5 memory processes"},
    {"prompt": "Check if docker is running"}
  ]
```

### Large File Write (Direct)
```
agent-farm:heavy_write
  path: "/tmp/large_output.txt"
  content: "... large content ..."
```

### Codebase Reconnaissance
```
agent-farm:recon_swarm
  target_path: "/home/kyle/repos/my-project"
```

### Generate Large Document (Chunked)
```
agent-farm:chunked_write
  output_path: "/tmp/security_guide.md"
  spec: "Linux server security hardening guide"
  num_sections: 5
  doc_type: "markdown"
```
Output: 9KB+ document with 5 coherent sections

### Generate Code File (Chunked)
```
agent-farm:chunked_code_gen
  output_path: "/tmp/utils.py"
  spec: "File utilities: read, write, copy, delete"
  language: "python"
  num_functions: 4
```
Output: Complete Python module with 4 functions

### Multi-Perspective Analysis
```
agent-farm:chunked_analysis
  target: "/home/kyle/repos/project"
  question: "What are the architectural patterns?"
  num_perspectives: 4
```
Output: Analysis from Structure, Patterns, Quality, Performance perspectives

## Installation

```bash
cd ~/repos/agent-farm
uv venv
uv pip install -e .
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "agent-farm": {
      "command": "/home/kyle/repos/agent-farm/.venv/bin/python",
      "args": ["-m", "agent_farm.server"]
    }
  }
}
```

## Changelog

### Changes included in this clean rebuild
- Removed a workflow script that hardcoded a credential (see the pull
  request description: removing it from a branch does NOT purge git history;
  rotate the key if it was real) and deleted the broken OpenAI workflow
  that ran a nonexistent root-level main.py. A replacement compile-check
  workflow could not be created through the GitHub App (403 on workflow
  files) and can be added manually after review if wanted.
- Path checks now use canonical, symlink-resolved paths with commonpath
  matching, applied to file reads, writes, listings, existence checks,
  disk usage, and file_swarm operations; write-dir matching no longer
  lets "/tmpfoo" pass as "/tmp".
- Expanded the exec-command blocklist and documented that it is not a
  sandbox; the server is for local, trusted use only.
- Colony and bug IDs are now uuid-based instead of second-resolution
  timestamps, so duplicate roles and same-second spawns cannot collide.
- Write tasks with path + content are direct-executed at any length;
  short writes are no longer silently dropped.
- file_swarm workers are capped at 8 and results are returned in input order.
- Removed the derived (circular) parallel-speedup calculation; no speedup
  is reported unless it is actually measured.
- Version aligned to 3.4.0 across pyproject, tools, and package init; the
  MCP tool count is corrected to 25 (counted from server.py).
- Tool-call parsing now handles quoted arguments containing commas and
  JSON object arguments, without breaking the TOOL: name(args) format.
- Dependencies in pyproject are now bounded (mcp <2, httpx <1, psutil <8);
  fastmcp stays pinned at 3.2.0.

### v3.4.0 (2026-01-23)
- **Chunked Write Pattern** - Bugs write sections in parallel, Python assembles
- **chunked_write** - Generate unlimited-size documents (tested: 9.6KB in 78s)
- **chunked_code_gen** - Generate multi-function code files in parallel
- **chunked_analysis** - Multi-perspective analysis with synthesis
- Bypasses 500-char bug limitation via task decomposition
- Planner uses structured JSON output for reliable outlines

### v3.3.0 (2026-01-23)
- **Ollama Structured Output** - JSON schema enforcement via constrained decoding
- **Reliable tool parsing** - No more regex failures
- **Mode tracking** - Results show parsing method used
- **Regex fallback** - Legacy parsing still available as backup
- All roles get exec_cmd for complex shell queries

### v3.2.0 (2026-01-22)
- **Synthesizer role** - qwen3:30b-a3b-q3k for accurate result synthesis
- **synthesize parameter** - Added to tool_swarm, system_health_swarm, recon_swarm, deep_analysis_swarm
- **synthesize tool** - Standalone synthesis of any JSON results
- **No more Claude synthesis tax** - bugs do ALL the work locally

### v3.1.0 (2026-01-20)
- 8.6x speed improvement (103s -> 12s)
- Auto-format results skip redundant LLM calls
- Reject invalid tools instantly
- Force tool usage before answers
- Complex shell command support fixed
- All roles upgraded to qwen3:4b minimum

### v3.0.0 (2026-01-19)
- Linux rebuild from Windows version
- Tool-enabled agents with role permissions
- System health, recon, worker swarms
- TRUE PARALLEL via ThreadPoolExecutor

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full text.

Copyright 2026 BossX429

## Clean rebuild note

This is a clean rebuild of agent-farm v3.4.0, based on the fixed code from BossX429/agent-farm main after PR #5 was merged. It contains no prior git history, no broken workflow files, and no credential files.
