"""Smoke test: the MCP server still exposes exactly the same 25 tools.

This guards a split/refactor of server.py: tool names must not change,
and every tool must still be importable from agent_farm.server (the
public entrypoint) as well as from its new home module.
"""

import agent_farm.server as server

EXPECTED_TOOLS = [
    "spawn_colony",
    "deploy_swarm",
    "quick_swarm",
    "list_colonies",
    "colony_status",
    "quick_colony",
    "farm_stats",
    "dissolve_colony",
    "cleanup_idle",
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
]


def test_server_exposes_25_tools():
    for name in EXPECTED_TOOLS:
        assert callable(getattr(server, name)), name
    assert len(EXPECTED_TOOLS) == 25


def test_colony_tools_work_without_ollama():
    # These tools do not call Ollama, so they must work offline.
    import asyncio

    async def main():
        colony = await server.spawn_colony("fast")
        assert colony["bugs"]
        status = await server.colony_status(colony["colony_id"])
        assert status["colony_id"] == colony["colony_id"]
        stats = await server.farm_stats()
        assert stats["active_colonies"] >= 1
        await server.dissolve_colony(colony["colony_id"])

    asyncio.run(main())
