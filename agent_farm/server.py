"""Agent Farm v3.4 - Chunked Write Edition
AI organism evolution and parallel task execution
Bug colonies with specialized roles, tool-enabled agents
TRUE PARALLEL via ThreadPoolExecutor

The MCP tools used to live in this single 1,900+ line file. They now
live in focused modules, and this file is only the entrypoint:

- state.py: model tiers, colonies, stats
- ollama.py: Ollama query helpers
- colony.py: colony core (creation, routing, parallel execution)
- colony_tools.py: colony management MCP tools
- swarm_tools.py: code/file/exec/api/kmkb and other swarm MCP tools
- chunked_tools.py: chunked write/code/analysis MCP tools

Importing the tool modules registers their tools on the shared
FastMCP instance in mcp_app.py. The tool functions are re-exported
here so `from agent_farm.server import exec_swarm` keeps working.
"""

from .mcp_app import mcp
from .state import MODEL_TIERS, OLLAMA_URL, colonies, stats
from .colony import _unique_id, create_bug, spawn_colony_internal
from .colony_tools import (
    spawn_colony,
    deploy_swarm,
    quick_swarm,
    list_colonies,
    colony_status,
    quick_colony,
    farm_stats,
    dissolve_colony,
    cleanup_idle,
)
from .swarm_tools import (
    review_pr,
    code_review_swarm,
    code_gen_swarm,
    file_swarm,
    exec_swarm,
    api_swarm,
    kmkb_swarm,
    tool_swarm,
    system_health_swarm,
    recon_swarm,
    worker_task,
    heavy_write,
    deep_analysis_swarm,
    synthesize,
)
from .chunked_tools import (
    chunked_write,
    chunked_code_gen,
    chunked_analysis,
)

__all__ = [
    "mcp",
    "MODEL_TIERS",
    "OLLAMA_URL",
    "colonies",
    "stats",
    "spawn_colony",
    "deploy_swarm",
    "quick_swarm",
    "list_colonies",
    "colony_status",
    "quick_colony",
    "farm_stats",
    "dissolve_colony",
    "cleanup_idle",
    "review_pr",
    "code_review_swarm",
    "code_gen_swarm",
    "file_swarm",
    "exec_swarm",
    "api_swarm",
    "kmkb_swarm",
    "tool_swarm",
    "system_health_swarm",
    "recon_swarm",
    "worker_task",
    "heavy_write",
    "deep_analysis_swarm",
    "synthesize",
    "chunked_write",
    "chunked_code_gen",
    "chunked_analysis",
    "main",
]


def main():
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
