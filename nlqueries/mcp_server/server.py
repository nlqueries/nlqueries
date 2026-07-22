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
list_agents         Return agent IDs available on this installation.
query               Ask a natural-language question to a named agent.
get_agent_schema    Return tables, columns, and relationships for an agent.
submit_feedback     Record thumbs-up/down feedback for a query result.
health              Check LLM, Qdrant, embed daemon, and config status.
invalidate_cache    Drop the semantic cache for an agent.
list_connectors     List registered database connectors.
get_query_history   Return recent queries and ratings for an agent.
get_cache_stats     Return cache size and collection info for an agent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP

from nlqueries import config

_log = logging.getLogger(__name__)

_Transport = Literal["stdio", "sse", "streamable-http"]

_INSTRUCTIONS = (
    "NLQueries translates natural-language questions into SQL and retrieves "
    "answers from databases and documents. "
    "Typical workflow: list_agents() → get_agent_schema() → query() → submit_feedback(). "
    "Use health() to diagnose connectivity issues."
)


# ---------------------------------------------------------------------------
# Tool implementations
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


def get_agent_schema(agent_id: str) -> str:
    """Return the schema for an agent: tables, columns, types, and foreign keys.

    Reads the agent's YAML knowledge base and formats it compactly so you can
    inspect exactly which tables and columns are available before formulating
    a question.

    Args:
        agent_id: Agent ID from list_agents().

    Returns:
        Formatted schema string, or an error message if the agent is not found.
    """
    import re  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    kb_path = config.KB_PATH / f"{safe_id}.yaml"
    if not kb_path.exists():
        return f"Agent '{agent_id}' not found. Available: {list_agents()}"

    try:
        kb: dict[str, Any] = yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read KB for '{agent_id}': {exc}"

    schema = kb.get("schema", {})
    tables: list[dict[str, Any]] = schema.get("tables", [])
    if not tables:
        return f"Agent '{agent_id}' has no tables in its knowledge base."

    lines: list[str] = [f"## Schema — {agent_id}\n"]
    for tbl in tables:
        name = tbl.get("name", "?")
        desc = tbl.get("description", "")
        header = f"**{name}**" + (f" — {desc}" if desc else "")
        lines.append(header)

        cols = tbl.get("columns", [])
        col_parts: list[str] = []
        for col in cols:
            col_name = col.get("name", "?")
            col_type = col.get("type", "").upper() or "TEXT"
            flags: list[str] = []
            if col.get("primary_key"):
                flags.append("PK")
            if col.get("foreign_key"):
                flags.append(f"FK→{col['foreign_key']}")
            samples = col.get("samples", [])
            sample_str = f" samples: {samples[:3]}" if samples else ""
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            col_parts.append(f"  {col_name}: {col_type}{flag_str}{sample_str}")

        lines.append("\n".join(col_parts))

        fks = tbl.get("foreign_keys", [])
        if fks:
            lines.append("  FK: " + ", ".join(fks))

        lines.append("")

    return "\n".join(lines)


async def query(
    question: str,
    agent_id: str,
    dialect: str = "postgres",
    explain: bool = False,
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
        explain:   When true, append an answer-provenance summary (route,
                   cache hit/miss, knowledge injected, checks, timings).

    Returns:
        Formatted string containing the natural-language answer, generated SQL
        (for SQL/hybrid agents), and source citations (for document/hybrid agents).
        Includes per-query latency and agent-type metadata.
    """
    from nlqueries.orchestrator.sync_runner import run_query  # noqa: PLC0415

    try:
        with anyio.fail_after(45):
            result = await run_query(question, agent_id, dialect=dialect, explain=explain)
    except TimeoutError:
        return (
            f"⏱ Query timed out after 45 s for agent '{agent_id}'.\n\n"
            "Most likely cause: the embed daemon is not running, so the first query "
            "triggers a slow in-process model load.\n\n"
            "Fix: run `nlqueries embed-server start` in your terminal, then retry."
        )
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

    if result.sql_result and not result.sql_result.error and result.sql_result.rows:
        qr = result.sql_result
        header = "| " + " | ".join(str(c) for c in qr.columns) + " |"
        sep = "| " + " | ".join("---" for _ in qr.columns) + " |"
        rows = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in qr.rows)
        shown = len(qr.rows)
        suffix = f"\n*(showing {shown} of {qr.row_count} rows)*" if qr.row_count > shown else ""
        parts.append(f"\n\n**Results**\n\n{header}\n{sep}\n{rows}{suffix}")

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

    if result.provenance is not None:
        p = result.provenance
        cache = "off"
        if p.cache is not None:
            cache = (
                f"hit ({p.cache.tier}"
                + (f", {p.cache.similarity:.2f}" if p.cache.similarity is not None else "")
                + ")"
                if p.cache.hit
                else "miss"
            )
        checks = "no warnings" if not p.validator else f"{len(p.validator)} warning(s)"
        parts.append(
            "\n\n**Provenance**"
            f"\n- route: {p.route or 'unknown'}"
            f"\n- cache: {cache}"
            f"\n- knowledge injected: {len(p.prompt_sections)} section(s), "
            f"{len(p.capsules_used)} capsule(s)"
            f"\n- checks: {checks}"
        )

    parts.append(f"\n\n*{result.agent_type} agent · {result.latency_ms} ms*")

    return "".join(parts)


def submit_feedback(
    question: str,
    agent_id: str,
    generated_sql: str,
    rating: str,
    corrected_sql: str | None = None,
) -> str:
    """Record user feedback for a query result.

    Feedback is stored in ~/.nlqueries/feedback/<agent_id>.jsonl and is used
    by ``nlqueries promote-feedback`` to improve future query accuracy by
    seeding verified (question, SQL) examples into retrieval.

    Args:
        question:      The question that was asked.
        agent_id:      The agent that answered it.
        generated_sql: The SQL that was generated (copy from the query result).
        rating:        ``"up"`` (correct answer) or ``"down"`` (wrong answer).
        corrected_sql: Optional corrected SQL when rating is ``"down"``.

    Returns:
        Confirmation string, or an error message.
    """
    from nlqueries.feedback.models import QueryFeedback  # noqa: PLC0415
    from nlqueries.feedback.store import record_feedback  # noqa: PLC0415

    if rating not in ("up", "down"):
        return f"Invalid rating {rating!r} — must be 'up' or 'down'."

    try:
        fb = QueryFeedback(
            question=question,
            generated_sql=generated_sql,
            corrected_sql=corrected_sql or None,
            rating=rating,
            agent_id=agent_id,
        )
        record_feedback(fb)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to record feedback: {exc}"

    msg = f"Feedback recorded: {'👍' if rating == 'up' else '👎'} for agent '{agent_id}'."
    if corrected_sql:
        msg += " Corrected SQL saved — run 'nlqueries promote-feedback' to apply it."
    return msg


def health() -> str:
    """Check connectivity to all NLQueries dependencies.

    Runs four checks in sequence:
    - LLM: sends a minimal completion request to verify the API key and model.
    - Qdrant: hits the /healthz endpoint.
    - Embed daemon: probes the local embedding server.
    - Config: verifies KB_PATH exists and required env vars are set.

    Returns:
        A formatted status report string.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    lines: list[str] = ["## NLQueries Health Check\n"]

    # LLM
    llm_key = config.ANTHROPIC_API_KEY or ""
    if not llm_key:
        lines.append("❌ **LLM** — no ANTHROPIC_API_KEY set")
    else:
        try:
            from nlqueries.llm import get_llm_client  # noqa: PLC0415

            t0 = time.monotonic()
            get_llm_client().complete("You are a health check.", "Reply OK.", max_tokens=5)
            ms = int((time.monotonic() - t0) * 1000)
            lines.append(f"✅ **LLM** — {config.LLM_PROVIDER} / {config.LLM_MODEL} ({ms} ms)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"❌ **LLM** — {config.LLM_PROVIDER}: {exc}")

    # Qdrant
    try:
        req = urllib.request.Request(f"{config.QDRANT_URL}/healthz")
        urllib.request.urlopen(req, timeout=3)
        lines.append(f"✅ **Qdrant** — {config.QDRANT_URL}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"❌ **Qdrant** — {config.QDRANT_URL}: {exc}")

    # Embed daemon
    try:
        from nlqueries.embeddings.embedder import _try_daemon_single  # noqa: PLC0415

        vec = _try_daemon_single("health")
        if vec is not None:
            lines.append(f"✅ **Embed daemon** — port {config.EMBED_SERVER_PORT} (dim {len(vec)})")
        else:
            lines.append(
                f"⚠️  **Embed daemon** — not running on port {config.EMBED_SERVER_PORT} "
                "(embeddings fall back to in-process load, ~9 s)"
            )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"⚠️  **Embed daemon** — probe failed: {exc}")

    # Config
    agents = list_agents()
    if agents:
        lines.append(f"✅ **Config** — {len(agents)} agent(s): {', '.join(agents)}")
    else:
        lines.append(
            f"⚠️  **Config** — KB_PATH ({config.KB_PATH}) has no agents yet "
            "(run 'nlqueries export-kb' to create one)"
        )

    return "\n".join(lines)


def invalidate_cache(agent_id: str) -> str:
    """Drop the entire semantic cache for an agent.

    Use this after a schema change or data refresh to force fresh SQL
    generation on the next query.  The cache is rebuilt automatically
    as new queries arrive.

    Args:
        agent_id: Agent whose cache should be cleared.

    Returns:
        Confirmation string.
    """
    from nlqueries.cache.semantic_cache import SemanticCache  # noqa: PLC0415

    try:
        SemanticCache(agent_id).invalidate(agent_id)
        return f"Cache for agent '{agent_id}' cleared. Next queries will bypass the cache."
    except Exception as exc:  # noqa: BLE001
        return f"Cache invalidation failed for '{agent_id}': {exc}"


def list_connectors() -> str:
    """List registered database connectors and their types.

    Returns connectors registered via ``nlqueries connect``.  Passwords and
    full connection URLs are redacted for safety.

    Returns:
        Formatted list of connector IDs and their db-types, or a message
        when no connectors are registered.
    """
    import yaml  # noqa: PLC0415

    if not config.CONNECTORS_FILE.exists():
        return "No connectors registered. Use 'nlqueries connect' to add one."

    try:
        raw: dict[str, Any] = (
            yaml.safe_load(config.CONNECTORS_FILE.read_text(encoding="utf-8")) or {}
        )
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read connectors file: {exc}"

    if not raw:
        return "No connectors registered. Use 'nlqueries connect' to add one."

    lines = ["## Registered Connectors\n"]
    for connector_id, cfg in raw.items():
        db_type = cfg.get("db_type", cfg.get("type", "unknown"))
        host = cfg.get("host", "")
        database = cfg.get("database", "")
        detail = f"{host}/{database}".strip("/") if (host or database) else ""
        lines.append(f"- **{connector_id}** ({db_type})" + (f" — {detail}" if detail else ""))

    return "\n".join(lines)


def get_query_history(agent_id: str, limit: int = 20) -> str:
    """Return recent queries and their feedback ratings for an agent.

    Args:
        agent_id: Agent to look up.
        limit:    Maximum number of records to return (default: 20, max: 200).

    Returns:
        Formatted history string, newest first.
    """
    from nlqueries.feedback.store import load_feedback  # noqa: PLC0415

    limit = min(max(1, limit), 200)
    try:
        records = load_feedback(agent_id)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to load history for '{agent_id}': {exc}"

    if not records:
        return f"No query history found for agent '{agent_id}'."

    records = sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]

    lines = [f"## Query History — {agent_id} (last {len(records)})\n"]
    for r in records:
        icon = "👍" if r.rating == "up" else "👎"
        ts = r.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{icon} [{ts}] {r.question}")
        if r.generated_sql:
            sql_preview = r.generated_sql[:120].replace("\n", " ")
            lines.append(f"   SQL: `{sql_preview}`" + ("…" if len(r.generated_sql) > 120 else ""))
        if r.corrected_sql:
            lines.append(f"   Corrected: `{r.corrected_sql[:120]}`")

    return "\n".join(lines)


def get_cache_stats(agent_id: str) -> str:
    """Return cache statistics for an agent.

    Reports the number of cached entries and the Qdrant collection name
    used for this agent's semantic cache.

    Args:
        agent_id: Agent to inspect.

    Returns:
        Formatted stats string.
    """
    from nlqueries.cache.semantic_cache import SemanticCache  # noqa: PLC0415

    try:
        stats = SemanticCache(agent_id).stats()
    except Exception as exc:  # noqa: BLE001
        return f"Could not retrieve cache stats for '{agent_id}': {exc}"

    total = stats.get("total_entries", 0)
    collection = stats.get("collection", "?")
    return (
        f"## Cache Stats — {agent_id}\n\n"
        f"- Entries: **{total}**\n"
        f"- Collection: `{collection}`\n"
        f"- Qdrant: {config.QDRANT_URL}"
    )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

_ALL_TOOLS: list[Callable[..., Any]] = [
    list_agents,
    get_agent_schema,
    query,
    submit_feedback,
    health,
    invalidate_cache,
    list_connectors,
    get_query_history,
    get_cache_stats,
]


def _build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """Create a FastMCP instance with all tools registered."""
    server = FastMCP("nlqueries", instructions=_INSTRUCTIONS, host=host, port=port)
    for fn in _ALL_TOOLS:
        server.add_tool(fn)
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
