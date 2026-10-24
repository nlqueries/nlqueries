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

import functools
import getpass
import inspect
import ipaddress
import logging
import os
import secrets
import socket
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import (
    DEFAULT_MAX_REQUEST_BODY_SIZE,
    RequestBodyLimitMiddleware,
)
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from nlqueries import config
from nlqueries.auth.principal import Principal, Source, local_principal
from nlqueries.state_files import agent_storage_id

_log = logging.getLogger(__name__)

_Transport = Literal["stdio", "sse", "streamable-http"]

_INSTRUCTIONS = (
    "NLQueries translates natural-language questions into SQL and retrieves "
    "answers from databases and documents. "
    "Typical workflow: list_agents() → get_agent_schema() → query() → submit_feedback(). "
    "Use health() to diagnose connectivity issues."
)


#: Which subjects may perform which actions on which agents. Required once
#: authentication is on: without it every authenticated call would be denied,
#: which is a misconfiguration rather than a policy.
_GRANTS_FILE_ENV = "NLQ_MCP_GRANTS_FILE"


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
    inspect which tables and columns are available before formulating a
    question. A table whose column list is partial carries a note saying so.

    Args:
        agent_id: Agent ID from list_agents().

    Returns:
        Formatted schema string, or an error message if the agent is not found.
    """

    import yaml  # noqa: PLC0415

    # The same note the prompt renderers emit, from the same constant, so the
    # four renderings of this column list cannot drift apart in what they say
    # about a partial one. Imported here rather than at module scope, matching
    # how this function defers its other imports.
    #
    # Why the note is needed: a deployment can restrict which columns an agent
    # may read, and the knowledge base then lists only the permitted ones, so a
    # withheld column is indistinguishable from one the table does not have. A
    # reader -- person or client model -- concludes the data is not there rather
    # than that it was withheld. Nothing on this path writes SQL, so it is
    # milder than in the prompt renderers, but it is the same false impression.
    #
    # That reasoning lives here and not in the docstring because FastMCP
    # publishes the docstring verbatim as the tool description every MCP client
    # model receives: `_build_server` registers `guard(fn, ...)` and `guard`
    # keeps `__doc__` through `functools.wraps`. Measured on the published
    # description: 403 characters before this change, 951 with the rationale in
    # the docstring, 458 with the one sentence above -- and every one of those
    # characters is spent on every client, for a distinction no client can act
    # on.
    from nlqueries.orchestrator.prompt_assembly import (  # noqa: PLC0415
        _PARTIAL_COLUMNS_NOTE,
        _columns_omitted,
    )

    safe_id = agent_storage_id(agent_id)
    kb_path = config.KB_PATH / f"{safe_id}.yaml"
    if not kb_path.exists():
        return _agent_not_found(agent_id)

    try:
        kb: dict[str, Any] = yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read KB for '{agent_id}': {exc}"

    schema = kb.get("schema", {})
    tables: list[dict[str, Any]] = schema.get("tables", [])
    if not tables:
        return f"Agent '{agent_id}' has no tables in its knowledge base."

    # The fourth rendering of this list, and the one shown to an MCP
    # client's model. It composes no SQL itself, but these are the names
    # that model will use when it asks for some -- so it names tables the
    # way the three prompt renderers do. Deferred like the imports above.
    from nlqueries.orchestrator.prompt_assembly import _table_ref  # noqa: PLC0415

    lines: list[str] = [f"## Schema — {agent_id}\n"]
    for tbl in tables:
        name = _table_ref(tbl) or tbl.get("name", "?")
        desc = tbl.get("description", "")
        header = f"**{name}**" + (f" — {desc}" if desc else "")
        lines.append(header)

        cols = tbl.get("columns", [])
        col_parts: list[str] = []
        for col in cols:
            col_name = col.get("name", "?")
            col_type = col.get("type", "").upper() or "TEXT"
            flags: list[str] = []
            # The knowledge base's own field names, as `_render_m_schema` reads
            # them. This read `primary_key` and `foreign_key`, which no
            # knowledge base this function can load has ever carried: the
            # generator wrote no key fields before `kb_version: 2` and these
            # three since, so no column ever showed a key flag here.
            if col.get("is_primary_key"):
                flags.append("PK")
            if col.get("is_foreign_key") and col.get("references"):
                flags.append(f"FK→{col['references']}")
            samples = col.get("samples", [])
            if samples and col.get("values_complete"):
                sample_str = f" values: {samples}"
            else:
                sample_str = f" samples: {samples[:3]}" if samples else ""
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            col_parts.append(f"  {col_name}: {col_type}{flag_str}{sample_str}")

        lines.append("\n".join(col_parts))

        if _columns_omitted(tbl):
            lines.append(f"  Note: {_PARTIAL_COLUMNS_NOTE}.")

        # No per-table foreign-key line: each foreign key shows as its column's
        # FK flag above. This used to read a `foreign_keys` list of strings
        # from the table, which the generator never writes (it writes one list
        # of {from, to} mappings under `schema`), so it never showed anything,
        # and a list in the generator's shape copied onto a table raised
        # TypeError in the join.
        lines.append("")

    return "\n".join(lines)


def _agent_not_found(agent_id: str) -> str:
    """The reply for an agent that does not exist.

    It names no other agent. Which agents exist is what `list_agents` answers,
    under its own permission; a mistyped ID is not a way round that.
    """
    return f"Agent '{agent_id}' not found. Use list_agents() to see the agents you can use."


def _reference() -> str:
    """A short id tying a failure in a tool reply to its full entry in the log."""
    return secrets.token_hex(4)


#: How long the MCP `query` tool waits for an answer.
_QUERY_BUDGET_SECONDS = 45
#: (agent, dialect) pairs a client asked for that disagreed with the agent's
#: connector and have already been logged. Both values come from the client, so
#: the set is capped, and each value is kept only as far as it is logged
#: (:data:`_DIALECT_LOGGED_CHARS`); once full, further disagreements go unlogged.
_DIALECT_WARNED: set[tuple[str, str]] = set()
_DIALECT_WARNED_MAX = 256
_DIALECT_LOGGED_CHARS = 40
#: The database's own limit on the statement, at or just under the tool's wait
#: (as `run_query` asks), so it stops when the tool gives up rather than
#: running on after the client has been told the query timed out.
_STATEMENT_BUDGET_SECONDS = 40


async def query(
    question: str,
    agent_id: str,
    dialect: str | None = None,
    explain: bool = False,
    paraphrase: bool = False,
) -> str:
    """Ask a natural-language question to an NLQueries agent.

    The agent translates the question into SQL (or searches indexed documents),
    and returns a natural-language answer together with the generated SQL and
    any source citations.

    Args:
        question:  The natural-language question, e.g. "Total revenue by region
                   last quarter?"
        agent_id:  Agent to query — use list_agents() to discover available IDs.
        dialect:   Ignored, and kept so clients that send it keep working. The
                   SQL dialect is the one of the database the agent's
                   connector points at.
        explain:   When true, append an answer-provenance summary (route,
                   cache hit/miss, knowledge injected, checks, timings).
        paraphrase: When true, append a deterministic plain-English rendering of
                   the generated SQL (no LLM), so you can confirm what it does.

    Returns:
        Formatted string containing the natural-language answer, generated SQL
        (for SQL/hybrid agents), and source citations (for document/hybrid agents).
        Includes per-query latency and agent-type metadata.
    """
    from nlqueries.connectors.base import short_cause, short_error_text  # noqa: PLC0415
    from nlqueries.connectors.loader import dialect_for_agent  # noqa: PLC0415
    from nlqueries.execution import ExecutionPolicy  # noqa: PLC0415
    from nlqueries.orchestrator.sync_runner import run_query  # noqa: PLC0415
    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    # This tool exists to answer from the database: its action is
    # `query:execute`, and `guard` has authorised that before this body runs.
    # Permission to execute is minted here, at the entry point, as the CLI's
    # `query` does. Without it the run fell back to the generate-only default
    # and every caller got SQL with no rows (the default makes a forgotten
    # grant visible rather than unsafe).
    #
    # Cancelling the wait does not stop a statement already running in its
    # worker thread, so the database gets a budget of its own, just under the
    # tool's, and the turn gets the tool's deadline: no correction starts on a
    # turn the client has already been told timed out.
    deadline = time.monotonic() + _QUERY_BUDGET_SECONDS
    try:
        # The SQL policy vouches for a statement only when it parses it with
        # the grammar of the engine that runs it, so that grammar comes from
        # the agent's connector and never from the caller. A client's `dialect`
        # is ignored; one that disagrees is logged, since it usually means the
        # client was configured for a different database. Inside the `try`:
        # this is the first read of the connectors file, and a hand-edited one
        # that does not parse must come back as a message like any other fault.
        engine_dialect = dialect_for_agent(agent_id)
        asked = _sqlglot_dialect(dialect) if dialect else None
        # Once per agent and dialect for the life of the process: a client that
        # always sends `postgres` would otherwise put this on every query.
        if asked and asked != engine_dialect:
            # Cut to what is logged: the key is what the line says, and a client
            # cannot park 256 strings of megabytes each in the set.
            seen = (agent_id[:_DIALECT_LOGGED_CHARS], asked[:_DIALECT_LOGGED_CHARS])
            if seen not in _DIALECT_WARNED and len(_DIALECT_WARNED) < _DIALECT_WARNED_MAX:
                _DIALECT_WARNED.add(seen)
                _log.warning(
                    "MCP query for agent %.40r asked for dialect %.40r; using %r, the "
                    "dialect of the agent's connector",
                    agent_id,
                    dialect,
                    engine_dialect,
                )
        with anyio.fail_after(_QUERY_BUDGET_SECONDS):
            result = await run_query(
                question,
                agent_id,
                dialect=engine_dialect,
                explain=explain,
                execution=ExecutionPolicy.execute_read_only(),
                timeout_seconds=_STATEMENT_BUDGET_SECONDS,
                deadline=deadline,
            )
    except TimeoutError:
        return (
            f"⏱ Query timed out after {_QUERY_BUDGET_SECONDS} s for agent '{agent_id}'.\n\n"
            "Likely causes:\n"
            "- the embed daemon is not running, so the first query triggers a slow "
            "in-process model load: run `nlqueries embed-server start`, then retry;\n"
            "- the query itself is slow: ask for less (a narrower filter or time "
            "range, an aggregate rather than raw rows). The database stops it after "
            f"{_STATEMENT_BUDGET_SECONDS} s."
        )
    except FileNotFoundError:
        return _agent_not_found(agent_id)
    except Exception as exc:  # noqa: BLE001
        reference = _reference()
        # The first line at ERROR, the traceback at DEBUG under the same
        # reference: as for a failed statement below, the lines after a
        # driver's first are where it quotes what it was sent, a connection
        # string included, and a log usually reaches more people than
        # connectors.yaml.
        _log.error(
            "MCP query tool error for agent %r [ref %s]: %s",
            agent_id,
            reference,
            short_cause(exc),
        )
        _log.debug("MCP query tool error [ref %s], in full:", reference, exc_info=True)
        return f"Query failed: {short_cause(exc)} (reference {reference})"

    parts: list[str] = [result.answer]

    if result.sql:
        parts.append(f"\n\n**Generated SQL**\n```sql\n{result.sql}\n```")

    if paraphrase and result.sql:
        from nlqueries.verbalizer import verbalize  # noqa: PLC0415

        described = verbalize(result.sql, engine_dialect)
        if described.text:
            hint = "" if described.complete else " *(described in part)*"
            parts.append(f"\n\n**In plain English**\n{described.text}{hint}")

    if result.sql_result and result.sql_result.error:
        # `sql_result.error` itself stays raw, as the correction loop needs it.
        # The full text is logged at DEBUG only: a log usually reaches a wider
        # audience than connectors.yaml, and the lines after the first are
        # where a driver quotes the DSN. The reference ties the two together.
        reference = _reference()
        summary = short_error_text(result.sql_result.error)
        _log.warning("SQL execution error for agent %r [ref %s]: %s", agent_id, reference, summary)
        _log.debug(
            "SQL execution error for agent %r [ref %s], in full: %s",
            agent_id,
            reference,
            result.sql_result.error,
        )
        parts.append(f"\n\n⚠ SQL execution error: {summary} (reference {reference})")

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

    Feedback is stored in ~/.nlqueries/feedback/<agent_id>.jsonl. A thumbs-up
    can become a verified example through ``nlqueries promote-feedback``, and a
    correction an example query through ``nlqueries export-kb``; feedback sent
    here is used only when an operator includes it with ``--include-anonymous``.

    Args:
        question:      The question that was asked.
        agent_id:      The agent that answered it.
        generated_sql: The SQL that was generated (copy from the query result).
        rating:        ``"up"`` (correct answer) or ``"down"`` (wrong answer).
        corrected_sql: Optional corrected SQL when rating is ``"down"``.

    Returns:
        Confirmation string, or an error message.
    """
    from nlqueries.feedback.models import SOURCE_MCP, QueryFeedback  # noqa: PLC0415
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
            # This transport has no authentication (SEC-05), so the record
            # identifies nobody and does not qualify for promotion on its own.
            source=SOURCE_MCP,
        )
        record_feedback(fb)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to record feedback: {exc}"

    msg = f"Feedback recorded: {'👍' if rating == 'up' else '👎'} for agent '{agent_id}'."
    if corrected_sql:
        # Not `promote-feedback`: that command handles thumbs-up only, so this
        # used to send operators to a step that ignores corrections.
        msg += (
            " Corrected SQL saved. Feedback sent over MCP does not identify who gave"
            " it, so an operator has to review it and run"
            " 'nlqueries export-kb --include-anonymous' to use the correction."
        )
    return msg


#: Whether the caller of the `health` call being served may see agent IDs: set
#: per call by the registered tool (see `_health_tool`) from the caller's
#: `agents:list` grant. False by default, so the report gives a count unless a
#: grant says otherwise.
_AGENT_IDS_VISIBLE: ContextVar[bool] = ContextVar("nlq_agent_ids_visible", default=False)


def health(probe_llm: bool = False) -> str:
    """Check connectivity to all NLQueries dependencies.

    Runs four checks in sequence:
    - LLM: reports the configured provider and model. Sends a real completion
      only when *probe_llm* is true.
    - Qdrant: hits the /healthz endpoint.
    - Embed daemon: probes the local embedding server.
    - Config: verifies KB_PATH exists and required env vars are set.

    Args:
        probe_llm: Spend a token on a live round-trip to the model. Off by
            default: this tool is reachable without credentials, and a health
            check that costs money is a way to bill someone for asking whether
            their server is up. Off, the LLM line reports configuration; on, it
            reports latency and a working key.

    Returns:
        A formatted status report string.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    lines: list[str] = ["## NLQueries Health Check\n"]

    # LLM
    # Not the Anthropic key by name: that was already wrong for an
    # OpenAI-only deployment, and Bedrock has no key at all.
    if not config.llm_credentials_available():
        lines.append(
            "❌ **LLM** — no credentials (ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or a Bedrock configuration)"
        )
    elif not probe_llm:
        lines.append(
            f"✅ **LLM** — {config.LLM_PROVIDER} / {config.LLM_MODEL} "
            "(configured; pass probe_llm=true to test the key)"
        )
    else:
        try:
            from nlqueries.llm import (  # noqa: PLC0415
                PROBE_MAX_TOKENS,
                get_llm_client,
                probe_round_trip,
            )

            t0 = time.monotonic()
            # See the CLI doctor: the probe asks for a small budget to stay
            # cheap, and a reasoning model can spend it all reasoning. The round
            # trip succeeded, which is what is being checked -- but it is said
            # out loud, so a caller diagnosing empty answers is not told the LLM
            # is simply fine.
            spent_on_reasoning = probe_round_trip(get_llm_client())
            ms = int((time.monotonic() - t0) * 1000)
            note = (
                f" — spent the {PROBE_MAX_TOKENS}-token probe budget before writing anything "
                "(normal for a reasoning model; raise LLM_MAX_OUTPUT_TOKENS if answers come "
                "back empty)"
                if spent_on_reasoning
                else ""
            )
            lines.append(f"✅ **LLM** — {config.LLM_PROVIDER} / {config.LLM_MODEL} ({ms} ms){note}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"❌ **LLM** — {config.LLM_PROVIDER}: {exc}")

    # Qdrant
    try:
        from nlqueries.http_urls import http_url  # noqa: PLC0415

        req = urllib.request.Request(http_url(f"{config.QDRANT_URL}/healthz"))  # noqa: S310
        urllib.request.urlopen(req, timeout=3)  # noqa: S310 - scheme checked
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
    if agents and _AGENT_IDS_VISIBLE.get():
        lines.append(f"✅ **Config** — {len(agents)} agent(s): {', '.join(agents)}")
    elif agents:
        lines.append(f"✅ **Config** — {len(agents)} agent(s)")
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
    from nlqueries.connectors.loader import (  # noqa: PLC0415
        ConnectorsFileError,
        _read_connectors_root,
    )

    if not config.CONNECTORS_FILE.exists():
        return "No connectors registered. Use 'nlqueries connect' to add one."

    try:
        # The loader's reader, so this tool reads the file as every other path
        # does. `object`: YAML can hold any shape, and the checks below are for that.
        raw: object = _read_connectors_root() or {}
    except ConnectorsFileError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read connectors file: {exc}"

    if not raw:
        return "No connectors registered. Use 'nlqueries connect' to add one."
    if not isinstance(raw, dict):
        return (
            f"Failed to read connectors file: its top level is a {type(raw).__name__}, "
            "not a mapping of connector ids to settings."
        )

    lines = ["## Registered Connectors\n"]
    for connector_id, cfg in raw.items():
        if not isinstance(cfg, dict):
            # Listed, so the entry to fix is named; reading it raised.
            shown = (
                "empty entry"
                if cfg is None
                else f"not a mapping of connection settings: {type(cfg).__name__}"
            )
            lines.append(f"- **{connector_id}** ({shown})")
            continue
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


#: The longest question a tool accepts, in characters.
_MAX_QUESTION_CHARS_ENV = "NLQ_MCP_MAX_QUESTION_CHARS"
_DEFAULT_MAX_QUESTION_CHARS = 4_000

#: The longest SQL a tool accepts, in characters.
_MAX_SQL_CHARS_ENV = "NLQ_MCP_MAX_SQL_CHARS"
_DEFAULT_MAX_SQL_CHARS = 8_000


def _char_limit(name: str, default: int) -> int:
    """A character limit from the environment, or *default*.

    Anything but a positive whole number falls back to the default with a
    warning. There is no value that turns the limit off: a typo should not
    remove it, and an operator who needs more can set a larger number.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        _log.warning("%s is not a positive number (%r); using %d.", name, raw, default)
        return default
    return value


def _input_limits() -> dict[str, int]:
    """The character limit for each tool argument that has one, by name."""
    question = _char_limit(_MAX_QUESTION_CHARS_ENV, _DEFAULT_MAX_QUESTION_CHARS)
    sql = _char_limit(_MAX_SQL_CHARS_ENV, _DEFAULT_MAX_SQL_CHARS)
    return {"question": question, "generated_sql": sql, "corrected_sql": sql}


def _bounded(func: Callable[..., Any], limits: dict[str, int]) -> Callable[..., Any]:
    """Wrap *func* so an over-long argument is refused before it runs.

    A question is sent to the model and embedded, and feedback is stored and can
    be promoted into retrieval, so their size is what a call costs. The check
    comes before the tool's body (`guard` has already authorised and admitted
    the call): an over-limit call gets a short refusal, and no model is called
    and nothing is stored.

    The signature is kept (`functools.wraps`), for the reason `guard` keeps it:
    FastMCP publishes each tool's schema from it.
    """
    signature = inspect.signature(func)
    names = [name for name in limits if name in signature.parameters]
    if not names:
        return func

    def refusal(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
        try:
            arguments = signature.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return None
        for name in names:
            value = arguments.get(name)
            if isinstance(value, str) and len(value) > limits[name]:
                return (
                    f"Refused: {name} is {len(value):,} characters, over the limit of "
                    f"{limits[name]:,}. Nothing was run."
                )
        return None

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_bounded(*args: Any, **kwargs: Any) -> Any:
            refused = refusal(args, kwargs)
            if refused is not None:
                return refused
            return await func(*args, **kwargs)

        return async_bounded

    @functools.wraps(func)
    def bounded(*args: Any, **kwargs: Any) -> Any:
        refused = refusal(args, kwargs)
        if refused is not None:
            return refused
        return func(*args, **kwargs)

    return bounded


def _health_tool(authorizer: Any, resolve: Callable[[], Principal]) -> Callable[..., Any]:
    """`health` as served: agent IDs only for a caller granted `agents:list`.

    `health:read` is the grant a monitor gets, and a monitor needs to know the
    server has agents, not what they are called. Listing them is what
    `agents:list` grants, so the IDs follow that grant and everyone else gets
    the count.
    """
    from nlqueries.auth.principal import Action  # noqa: PLC0415

    @functools.wraps(health)
    def health_tool(*args: Any, **kwargs: Any) -> str:
        visible = authorizer.authorize(resolve(), Action.AGENTS_LIST, None).allowed
        token = _AGENT_IDS_VISIBLE.set(visible)
        try:
            return health(*args, **kwargs)
        finally:
            _AGENT_IDS_VISIBLE.reset(token)

    return health_tool


class _Server(FastMCP):
    """FastMCP with the SSE transport's request bodies capped.

    The SDK caps a streamable-HTTP request body at 4 MiB and answers 413 above
    that. In 1.29, the oldest release this supports, its SSE transport reads a
    POSTed message whole, with no cap. The same middleware, with the same
    limit, goes in front of the SSE app here so the two networked transports
    accept the same sizes. Later releases cap SSE themselves; the check is then
    made twice, at the same limit, which costs nothing.
    """

    def sse_app(self, mount_path: str | None = None) -> Starlette:
        app = super().sse_app(mount_path)
        app.add_middleware(RequestBodyLimitMiddleware, max_body_size=DEFAULT_MAX_REQUEST_BODY_SIZE)
        return app


def _build_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    authenticated: bool = False,
    networked: bool = False,
) -> FastMCP:
    """Create a FastMCP instance with all tools registered.

    With *authenticated*, the configured verifier is attached and the SDK
    rejects any request whose bearer token it does not accept. Without it the
    server behaves as before, which is correct for stdio and refused for a
    networked transport (see :func:`_require_authentication`).
    """
    auth_args: dict[str, Any] = {}
    if authenticated:
        from nlqueries.auth.mcp_verifier import build_verifier  # noqa: PLC0415

        configured = build_verifier()
        if configured is None:  # pragma: no cover - guarded by the caller
            raise SystemExit("MCP authentication was requested but none is configured.")
        verifier, settings = configured
        auth_args = {"token_verifier": verifier, "auth": settings}

    server = _Server(
        "nlqueries",
        instructions=_INSTRUCTIONS,
        host=host,
        port=port,
        transport_security=_transport_security(host) if networked else None,
        **auth_args,
    )

    # One guard, at the one place every tool is registered. Nine call sites
    # would be nine chances for the tenth tool to be added without one.
    from nlqueries.audit import Surface, audited_tool  # noqa: PLC0415
    from nlqueries.auth.admission import from_config  # noqa: PLC0415
    from nlqueries.auth.enforcement import guard  # noqa: PLC0415

    authorizer = _resolve_authorizer(authenticated, networked=networked)
    # One control for the process, shared by every tool, so a caller cannot get
    # their allowance again by switching tool.
    #
    # Only where callers are told apart. On a transport running without
    # authentication every request resolves to the same anonymous subject, so
    # these would stop being per-caller limits and become one budget for the
    # whole deployment -- a cap the operator never had before, which any single
    # client could exhaust and thereby starve the others. The switch that turns
    # authentication off is a request for the previous behaviour, and this is
    # part of what it gives up.
    admission = from_config() if (networked and authenticated) else None
    resolve = _principal_for_call(networked=networked)
    limits = _input_limits()
    for fn in _ALL_TOOLS:
        if fn is health:
            fn = _health_tool(authorizer, resolve)
        # Outermost, so the authorisation decision and the execution it lets
        # through are recorded under one request id.
        guarded = guard(_bounded(fn, limits), authorizer, resolve, admission)
        server.add_tool(audited_tool(guarded, Surface.MCP, resolve))
    return server


def _transport_security(host: str) -> TransportSecuritySettings | None:
    """Host and Origin checks for a listener bound beyond loopback.

    On a loopback bind the SDK turns these on itself, allowing the loopback
    names. On any other bind it turns them off, so a page in a browser on the
    same network could reach the server through a DNS name that resolves to it.
    Here the server answers only to the name it is published under -- the host
    of `NLQ_MCP_RESOURCE_URL` -- and to `localhost`, `127.0.0.1` and `[::1]`,
    which is how a published container port is reached from its host. A request
    with any other `Host` gets 421. `Origin` follows the same names: the
    resource URL's scheme on its host, in the same spellings as the host, and
    `http://` on each loopback name, as the SDK
    allows on a loopback bind, so a client is not refused for naming as its
    origin a host it was let in on. Any other `Origin` gets 403; a client that
    sends none is not affected.

    The SDK matches a `Host` or `Origin` exactly, or against `name:*` as "name,
    a colon and a port", so `localhost:*` does not match a bare `localhost`.
    Each name is therefore listed in every form a client or proxy sends: a proxy
    on a scheme's default port sends no port at all, some clients send the
    default port explicitly, and a browser leaves the default port out of
    `Origin`.
    """
    if _is_loopback(host):
        return None

    from nlqueries.auth.mcp_verifier import RESOURCE_URL_ENV  # noqa: PLC0415

    allowed_hosts = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "[::1]", "[::1]:*"]
    allowed_origins = [
        "http://localhost",
        "http://localhost:*",
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://[::1]",
        "http://[::1]:*",
    ]
    resource = urlsplit(os.getenv(RESOURCE_URL_ENV, "").strip())
    scheme = resource.scheme.lower()
    try:
        explicit = resource.port
    except ValueError:
        # Not a port number. Nothing about the resource host can be trusted,
        # so it is left out, as an unparseable URL is.
        scheme = ""
    if scheme and resource.hostname:
        netloc = resource.netloc.rpartition("@")[2].lower()
        name = netloc[: -len(f":{explicit}")] if explicit is not None else netloc
        default = _DEFAULT_PORTS.get(scheme)
        effective = explicit if explicit is not None else default
        # The same spellings for the origin as for the host: a browser leaves
        # the default port out of `Origin`, and other clients may write it.
        for prefix, entries in (("", allowed_hosts), (f"{scheme}://", allowed_origins)):
            entries.append(f"{prefix}{name}")
            if effective is not None:
                entries.append(f"{prefix}{name}:{effective}")
            if explicit is not None and explicit != default:
                entries.append(f"{prefix}{name}:*")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(dict.fromkeys(allowed_hosts)),
        allowed_origins=list(dict.fromkeys(allowed_origins)),
    )


#: The port a client leaves out of `Host` for each scheme.
_DEFAULT_PORTS = {"https": 443, "http": 80}


def _principal_for_call(*, networked: bool) -> Callable[[], Principal]:
    """How to name the caller, decided by transport rather than by request.

    Three cases, not two. Over a networked transport with a verifier the SDK has
    put the verified token on the request context. Over stdio there is no token
    because the caller launched the process. And a networked transport running
    without a verifier — which an operator has to ask for — has callers about
    whom nothing is known.

    That last case used to fall through to the process owner, so every remote
    call was recorded against the account the server runs as. An audit trail
    that names the wrong party is worse than one that says it does not know.

    The transport is also checked per call, for a server built as stdio and
    then served over HTTP anyway (the module-level `mcp` mounted in an app, or
    run with another transport). The local principal stands for whoever
    launched the process, and a request that arrived over HTTP is not that, so
    it is anonymous too.
    """

    def resolve() -> Principal:
        from mcp.server.auth.middleware.auth_context import get_access_token  # noqa: PLC0415

        from nlqueries.auth.mcp_verifier import principal_from  # noqa: PLC0415

        token = get_access_token()
        if token is not None:
            return principal_from(token)
        if networked or _over_http():
            return Principal(subject="anonymous", source=Source.ANONYMOUS)
        return local_principal(getpass.getuser())

    return resolve


def _over_http() -> bool:
    """Whether the call being served arrived as an HTTP request.

    The SDK puts the HTTP request on the request context for its HTTP
    transports and leaves it empty for stdio.
    """
    from mcp.server.lowlevel.server import request_ctx  # noqa: PLC0415

    context = request_ctx.get(None)
    return context is not None and context.request is not None


def _resolve_authorizer(authenticated: bool, *, networked: bool = False) -> Any:
    """The authorizer for this server, chosen by transport and configuration.

    stdio gets the local profile: the caller owns the process and already has
    what the process has. A networked transport with a verifier gets the
    allowlist. A networked transport without one gets an authorizer that grants
    everything and says that is what it is doing — that is the behaviour before
    SEC-05, and an operator has to set a switch to have it.

    Deny-by-default remains the code path; the case of authentication with no
    grants file is refused at startup instead, because a server that
    authenticates correctly and then denies every call is a support case rather
    than a policy.
    """
    from nlqueries.auth.authorizer import (  # noqa: PLC0415
        AllowAllUnauthenticated,
        ConfigAllowlistAuthorizer,
        LocalAuthorizer,
        load_grants,
    )

    if not authenticated:
        return AllowAllUnauthenticated() if networked else LocalAuthorizer()

    path = os.getenv(_GRANTS_FILE_ENV, "").strip()
    if not path:
        raise SystemExit(
            f"MCP authentication is configured but {_GRANTS_FILE_ENV} is not set, "
            "so every authenticated call would be denied. Write a grants file "
            "naming which subjects may perform which actions on which agents."
        )
    return ConfigAllowlistAuthorizer(load_grants(Path(path)))


# Module-level instance — used for tool introspection and tests.
mcp = _build_server()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Hosts that expose the listener beyond this machine. `0.0.0.0` and `::` are
#: every interface; an empty host is how some stacks spell the same thing. Any
#: other spelling of an unspecified address counts too: see `_is_wildcard`.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})  # noqa: S104


def _address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """*host* as an IP address, or None for a name.

    Read the way the socket layer reads it, not only the way `ipaddress` does:
    `0`, `0.0` and `0x0` are all `0.0.0.0` to `bind`, though `ipaddress`
    rejects them. An IPv4-mapped IPv6 address is returned as the IPv4 address
    it maps.
    """
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    address: ipaddress.IPv4Address | ipaddress.IPv6Address
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        try:
            address = ipaddress.IPv4Address(socket.inet_aton(text))
        except (OSError, ValueError):
            return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _is_wildcard(host: str) -> bool:
    """Whether binding *host* listens on every interface."""
    if host.strip() in _WILDCARD_HOSTS:
        return True
    address = _address(host)
    return address is not None and address.is_unspecified


def _is_loopback(host: str) -> bool:
    """Whether binding *host* listens on this machine only."""
    if host.strip().lower() == "localhost":
        return True
    address = _address(host)
    return address is not None and address.is_loopback


#: Set to `1` to bind a network interface anyway. Named for what it does.
_INSECURE_BIND_ENV = "NLQ_ALLOW_INSECURE_BIND"


def _refuse_unauthenticated_exposure(host: str, *, authenticated: bool = False) -> None:
    """Refuse a wildcard bind unless the operator has said they mean it.

    Every tool on this server is reachable without credentials — including
    `query`, which runs SQL against a configured database, and
    `invalidate_cache`. Binding to every interface therefore publishes a
    database-query API to whatever can route to the host, and the CLI used to
    default to exactly that.

    Unlike the Qdrant rule in `embeddings/qdrant_client`, this one has an escape
    hatch, and the difference is deliberate: talking to a remote vector store
    anonymously has no legitimate use, whereas a network-reachable MCP transport
    is a supported deployment that simply is not safe *yet*. The switch is the
    honest way to say "I know, and I am doing it anyway" — and it should be
    deleted, along with this function, once authentication lands.

    Authentication has since landed (a networked transport now refuses to start
    without it), but what this allows and refuses is unchanged: *authenticated*
    changes only what it says. Told "no authentication" on a server that
    requires a token, an operator reasonably concludes the token is not in
    force -- the shipped compose file, which sets the switch and configures a
    token, logged exactly that on every start.
    """
    if not _is_wildcard(host):
        return
    if os.getenv(_INSECURE_BIND_ENV, "").strip().lower() in {"1", "true", "yes"}:
        if authenticated:
            _log.info(
                "MCP server bound to %s: reachable from other hosts, and every "
                "caller must authenticate.",
                host,
            )
        else:
            _log.warning(
                "MCP server bound to %s with no authentication: every tool, "
                "including query and invalidate_cache, is reachable by anything "
                "that can route here.",
                host,
            )
        return

    exposure = (
        "authentication is on, but this still serves a database-query API on "
        "every interface of this host"
        if authenticated
        else "it has no authentication, so every tool — including query, which "
        "runs SQL against your database — would be reachable by anything that "
        "can route to this host"
    )
    raise SystemExit(
        f"Refusing to bind the MCP server to {host!r}: {exposure}. "
        "Use --host 127.0.0.1 (the default) and put a proxy in front if you need "
        f"remote access, or set {_INSECURE_BIND_ENV}=1 to bind anyway."
    )


#: Transports that listen on a socket. `streamable-http` was absent from the
#: exposure check and from the server construction below, so it neither refused
#: a wildcard bind nor honoured --host: it silently served on the module-level
#: instance's defaults.
_NETWORK_TRANSPORTS = frozenset({"sse", "streamable-http"})

#: Set to `1` to serve a networked transport with no authentication.
#:
#: Deliberately not `NLQ_ALLOW_INSECURE_BIND`. That one says "I know this port
#: is reachable"; this one says "I know anyone who reaches it may run SQL
#: against my database". They are different admissions and a deployment that
#: made the first should not be taken to have made the second.
_ALLOW_UNAUTHENTICATED_ENV = "NLQ_ALLOW_UNAUTHENTICATED_MCP"


def _require_authentication(transport: str) -> bool:
    """Whether to attach a verifier, refusing if one is needed and absent.

    Returns True when the server should be built with authentication.
    """
    if transport not in _NETWORK_TRANSPORTS:
        # stdio: the caller launched this process and already has whatever it
        # has. The local profile is authorised in step 3 through the same
        # interface rather than bypassing it.
        return False

    from nlqueries.auth.mcp_verifier import build_verifier  # noqa: PLC0415

    if build_verifier() is not None:
        return True

    if os.getenv(_ALLOW_UNAUTHENTICATED_ENV, "").strip().lower() in {"1", "true", "yes"}:
        _log.warning(
            "MCP %s transport serving with no authentication: every tool, "
            "including query, is reachable by anything that can connect.",
            transport,
        )
        return False

    raise SystemExit(
        f"Refusing to start the MCP {transport} transport with no authentication: "
        "every tool — including query, which runs SQL against your database — "
        "would be reachable by anything that can connect. Configure an identity "
        "provider (NLQ_MCP_OIDC_DISCOVERY_URL and NLQ_MCP_OIDC_CLIENT_ID) or a "
        "pre-shared token (NLQ_MCP_STATIC_TOKEN), both with NLQ_MCP_RESOURCE_URL. "
        f"Set {_ALLOW_UNAUTHENTICATED_ENV}=1 to serve anyway."
    )


def main(transport: _Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the MCP server.

    Args:
        transport: ``"stdio"`` (default, for Claude Desktop), ``"sse"`` or
                   ``"streamable-http"`` (both for network clients).
        host:      Bind host for the networked transports (ignored for stdio).
        port:      Port for the networked transports (ignored for stdio).
    """
    # A signing key too short to protect the cache stops the server here, with
    # the reason. Found later, on a query, it can only turn the cache off.
    from nlqueries.cache.envelope import SigningKeyError, check_signing_key  # noqa: PLC0415

    try:
        check_signing_key()
    except SigningKeyError as exc:
        # As the other startup refusals here: the reason, not a traceback.
        raise SystemExit(str(exc)) from exc

    from nlqueries.audit import Surface, bind_process  # noqa: PLC0415

    # For work a tool call starts on a thread that does not inherit its
    # context: recorded as the server's, not the library's or the CLI's.
    bind_process(Surface.MCP)
    # Authentication first, so the bind guard below can say whether it is on.
    authenticated = _require_authentication(transport)
    if transport in _NETWORK_TRANSPORTS:
        # In `main`, not in the CLI: `mcp_entry.py` and `nlqueries-mcp` both
        # reach the server without passing through Click.
        _refuse_unauthenticated_exposure(host, authenticated=authenticated)
    server = (
        _build_server(host=host, port=port, authenticated=authenticated, networked=True)
        if transport in _NETWORK_TRANSPORTS
        else mcp
    )
    server.run(transport=transport)
