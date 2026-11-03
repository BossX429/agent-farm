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
- **Local synthesis** - qwen2.5:14b synthesizes results (no cloud tokens!)

## Security

This server runs shell commands (`exec_cmd`, `exec_swarm`) and file operations
on the local machine. Its command/path checks are a blocklist and a set of
path guards, **not a sandbox**: they reduce accidental damage but cannot make
LLM-generated shell commands safe against untrusted input. Run it locally,
with trusted prompts only, and do not expose it to untrusted files, web pages,
or other users' prompts.

Hardening added in this PR:
- **HTTP SSRF guard**: `http_get`/`http_post` now allow only public
  http/https URLs. Private, loopback, link-local, and metadata addresses
  (including `127.0.0.1` and `169.254.169.254`) are blocked, DNS failures
  fail closed, and redirects are re-validated hop by hop. For local
  testing only, `AGENT_FARM_ALLOW_PRIVATE_HTTP=1` or
  `AGENT_FARM_ALLOWED_HTTP_HOSTS=host1,host2` can relax this. The internal
  KMKB client (`127.0.0.1:8765`) does not go through these tools.
- **Shell-free simple commands**: commands with no pipes, redirection,
  chaining, or substitution now run with `shell=False`, so their arguments
  cannot be interpreted as shell syntax. Complex shell commands still use
  `shell=True` after the blocklist, multi-line commands are rejected, and
  command substitution (`$(...)`, backticks, `${...}`) is blocked.
- **Tests and CI**: see Testing below. CI runs compile checks and pytest
  on push and pull requests to `main`.
- **Swarm tools use the same guards**: `exec_swarm` now delegates to
  `tool_exec_cmd` and `api_swarm` now delegates to the shared HTTP
  helper, so neither can bypass the exec or SSRF guards above.

## Testing

```bash
pip install .[dev]
python -m pytest -q
```

Tests in `tests/test_tools.py` cover path safety and symlink traversal,
write-allowlist prefix lookalikes, exec blocklist variants and
shell-free execution, HTTP SSRF guards (with mocked DNS), tool-call
parsing (JSON, quoted commas, parentheses), and ID uniqueness.

## Models

| Role | Model | VRAM | Purpose |
|------|-------|------|---------|
| Scout | qwen3:4b | 2.5GB | Reconnaissance |
| Worker | qwen3:4b | 2.5GB | Task execution |
| Memory | qwen3:4b | 2.5GB | Context retention |
| Guardian | qwen3:4b | 2.5GB | System monitoring |
| Learner | qwen3:4b | 2.5GB | Pattern acquisition |
| **Synthesizer** | **qwen2.5:14b** | **8.99GB** | **Result synthesis** |

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
1. PLANNER BUG (qwen2.5:14b)
   |-- Creates structured JSON outline
   |-- {"sections": [{"title": "...", "description": "..."}]}

2. WORKER BUGS (qwen3:4b) - IN PARALLEL
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
- **Synthesizer role** - qwen2.5:14b for accurate result synthesis
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
