"""
Entry point for Claude Desktop on Windows.

Windows Application Control policies block pip-generated .exe scripts, so
Claude Desktop cannot call `nlqueries-mcp.exe` directly.  This script is a
plain Python file that can be invoked as:

    python.exe path/to/mcp_entry.py

It bypasses the Click CLI entirely and starts the MCP server over stdio.
"""

from nlqueries.mcp_server.server import main

main(transport="stdio")
