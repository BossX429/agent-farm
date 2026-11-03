"""Single FastMCP instance, so tool modules can register without cycles."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-farm")
