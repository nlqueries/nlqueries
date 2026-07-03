"""
nlqueries.mcp_server.server
~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP server that exposes NLQueries agents as tools for Claude Desktop and
other MCP-compatible clients.

Transports
----------
stdio  (default) — Claude Desktop integration; the client launches the process
                   and communicates over stdin/stdout.
sse              — HTTP server-sent events; suitable for network clients.

Tools
-----
list_agents   Return the IDs of all agents available on this installation.
query         Ask a natural-language question to a named agent.
"""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.fastmcp import FastMCP

from nlqueries import config

_log = logging.getLogger(__name__)

_Transport = Literal["stdio", "sse", "streamable-http"]

_INSTRUCTIONS = (
    "NLQueries translates natural-language questions into SQL and retrieves "
    "answers from databases and documents. "
    "Use list_agents() to discover available agents, then query() to ask questions."
)


# ---------------------------------------------------------------------------
# Tool implementations (module-level so they are importable and testable)
# ---------------------------------------------------------------------------


def list_agents() -> list[str]:
    """Return the IDs of all agents available on this NLQueries installation.

    An agent is available when its YAML knowledge-base file exists in KB_PATH
    (default: ~/.nlqueries/knowledge_base/).  Run ``nlqueries export-kb`` to
    create an agent from a registered connector.

    Returns:
        Sorted list of agent ID strings, e.g. ["sales", "support"].
        Empty list when no agents have been exported yet.
    """
    if not config.KB_PATH.exists():
        return []
    return sorted(p.stem for p in config.KB_PATH.glob("*.yaml"))


def query(
    question: str,
    agent_id: str,
    dialect: str = "postgres",
) -> str:
    """Ask a natural-language question to an NLQueries agent.

    The agent translates the question into SQL (or searches indexed documents),
    and returns a natural-language answer together with the generated SQL and
    any source citations.

    Args:
        question:  The natural-language question, e.g. "Total revenue by region
                   last quarter?"
        agent_id:  Agent to query — use list_agents() to discover available IDs.
        dialect:   SQL dialect for the target database (default: postgres).
                   Other supported values: snowflake, bigquery, redshift,
                   mysql, mssql, duckdb.

    Returns:
        Formatted string containing the natural-language answer, generated SQL
        (for SQL/hybrid agents), and source citations (for document/hybrid agents).
        Includes per-query latency and agent-type metadata.
    """
    from nlqueries.orchestrator.sync_runner import run_query_sync  # noqa: PLC0415

    try:
        result = run_query_sync(question, agent_id, dialect=dialect)
    except FileNotFoundError:
        available = list_agents()
        hint = (
            f"Available agents: {available}"
            if available
            else "No agents found — run 'nlqueries export-kb' first."
        )
        return f"Agent '{agent_id}' not found. {hint}"
    except Exception as exc:  # noqa: BLE001
        _log.exception("MCP query tool error for agent %r", agent_id)
        return f"Query failed: {exc}"

    parts: list[str] = [result.answer]

    if result.sql:
        parts.append(f"\n\n**Generated SQL**\n```sql\n{result.sql}\n```")

    if result.sql_result and result.sql_result.error:
        parts.append(f"\n\n⚠ SQL execution error: {result.sql_result.error}")

    if result.citations:
        parts.append("\n\n**Sources**")
        for c in result.citations:
            label = c.source_name
            if c.page_number is not None:
                label += f", page {c.page_number}"
            parts.append(f"\n- {label}")
            if c.excerpt:
                excerpt = c.excerpt[:200] + "…" if len(c.excerpt) > 200 else c.excerpt
                parts.append(f'\n  > "{excerpt}"')

    parts.append(f"\n\n*{result.agent_type} agent · {result.latency_ms} ms*")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def _build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """Create a FastMCP instance with all tools registered."""
    server = FastMCP("nlqueries", instructions=_INSTRUCTIONS, host=host, port=port)
    server.add_tool(list_agents)
    server.add_tool(query)
    return server


# Module-level instance — used for tool introspection and tests.
mcp = _build_server()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(transport: _Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the MCP server.

    Args:
        transport: ``"stdio"`` (default, for Claude Desktop) or ``"sse"``
                   (HTTP server-sent events, for network clients).
        host:      Bind host for SSE transport (ignored for stdio).
        port:      Port for SSE transport (ignored for stdio).
    """
    server = _build_server(host=host, port=port) if transport == "sse" else mcp
    server.run(transport=transport)
