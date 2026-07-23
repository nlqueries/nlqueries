"""
nlqueries.orchestrator.sync_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synchronous (non-streaming) query runner that drives MultiAgentOrchestrator
to completion and returns a single structured AgentQueryResult.

This is the adapter layer between the streaming orchestrator and the REST API
/ SDK callers who want a complete result in one call.

Task 26.5 (Sprint 26) — Timeout cancellation propagation. ``timeout_seconds``
threads down to the SQL sub-agent's ``execute_query`` call, applied as
``SET LOCAL statement_timeout`` on Postgres connections (see
``nlqueries.connectors.postgres.PostgresConnector.execute_query``) so a
runaway query is aborted by the database itself rather than left running
orphaned after the caller has given up. It does not (yet) bound the LLM
client call — see the enterprise-side coordination note for why that half
is scoped as a separate, larger core change.

Public API
----------
``AgentQueryResult``
    Structured result of a completed agent query.
``run_query``
    Async function that drives :class:`MultiAgentOrchestrator` to completion.
``run_query_sync``
    Synchronous wrapper around ``run_query`` via ``asyncio.run()``; for SDK
    users who do not have an async runtime.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.conversation import ConversationTurn
from nlqueries.orchestrator.document_retrieval import Citation
from nlqueries.orchestrator.followup_resolver import resolve_followup
from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
from nlqueries.orchestrator.provenance import Provenance, use_provenance


@dataclass
class AgentQueryResult:
    """Structured result returned by :func:`run_query`."""

    question: str
    resolved_question: str  # after follow-up resolution
    agent_type: str  # "sql" | "document" | "hybrid" | "unclear"
    answer: str  # natural-language answer text (all streamed tokens joined)
    sql: str | None  # generated SQL if agent_type in ("sql", "hybrid")
    sql_result: QueryResult | None  # raw SQL rows if executed
    citations: list[Citation]  # empty list if agent_type == "sql"
    merged_answer: str | None  # set only for hybrid
    latency_ms: int  # total time from question to result
    session_id: str | None
    from_cache: bool = False  # True when served from the semantic cache (Task 27.3)
    # Advisory findings attached by a caller after generation (e.g. the
    # enterprise Nexus join/PII validator). Core never populates this itself —
    # it is a carrier so a wrapping layer and the CLI can surface the same
    # findings on the canonical result object.
    nexus_warnings: list[str] = field(default_factory=list)
    # Answer provenance (SYL-1.1): how the answer was produced — route, capsules,
    # KB parts injected, cache hit/miss, validator warnings, timings. Populated
    # only when ``explain=True`` is passed to :func:`run_query`; ``None`` otherwise,
    # so default behaviour is unchanged.
    provenance: Provenance | None = None


def _parse_final_chunk(
    tokens: list[str],
) -> tuple[list[str], str, str | None, list[Citation], str | None, QueryResult | None, bool]:
    """Parse the last token as a structured JSON chunk.

    Returns:
        (text_tokens, agent_type, sql, citations, merged_answer, sql_result, from_cache)
    """
    agent_type = "unclear"
    sql: str | None = None
    citations: list[Citation] = []
    merged_answer: str | None = None
    sql_result: QueryResult | None = None
    text_tokens = tokens

    if not tokens:
        return text_tokens, agent_type, sql, citations, merged_answer, sql_result, False

    try:
        parsed: Any = json.loads(tokens[-1])
    except (json.JSONDecodeError, TypeError):
        return text_tokens, agent_type, sql, citations, merged_answer, sql_result, False

    if not isinstance(parsed, dict) or "agent_type" not in parsed:
        return text_tokens, agent_type, sql, citations, merged_answer, sql_result, False

    from_cache = bool(parsed.get("from_cache", False))
    agent_type = str(parsed.get("agent_type", "unclear"))
    text_tokens = tokens[:-1]

    sql_table: dict[str, Any] | None = None
    if agent_type == "sql":
        sql = parsed.get("sql") or None
        sql_table = parsed.get("sql_table")
        if sql_table and isinstance(sql_table, dict):
            sql_result = QueryResult(
                columns=sql_table.get("columns") or [],
                rows=sql_table.get("rows") or [],
                row_count=int(sql_table.get("row_count", 0)),
                execution_time_ms=float(sql_table.get("execution_time_ms", 0.0)),
                error=sql_table.get("error"),
            )

    elif agent_type == "document":
        citations = [
            Citation(
                chunk_id="",
                source_name=str(c.get("source_name", "")),
                page_number=c.get("page_number"),
                chunk_index=0,
                excerpt=str(c.get("excerpt", "")),
                relevance_score=0.0,
            )
            for c in parsed.get("citations", [])
        ]

    elif agent_type == "hybrid":
        merged_answer = parsed.get("merged_answer") or None
        text_tokens = []  # hybrid: NL answer IS the merged_answer field
        citations = [
            Citation(
                chunk_id="",
                source_name=str(c.get("source_name", "")),
                page_number=c.get("page_number"),
                chunk_index=0,
                excerpt=str(c.get("excerpt", "")),
                relevance_score=0.0,
            )
            for c in parsed.get("citations", [])
        ]
        sql_table = parsed.get("sql_table")
        if sql_table and isinstance(sql_table, dict):
            sql_result = QueryResult(
                columns=sql_table.get("columns") or [],
                rows=sql_table.get("rows") or [],
                row_count=int(sql_table.get("row_count", 0)),
                execution_time_ms=float(sql_table.get("execution_time_ms", 0.0)),
                error=sql_table.get("error"),
            )

    return text_tokens, agent_type, sql, citations, merged_answer, sql_result, from_cache


async def run_query(
    question: str,
    agent_id: str,
    available_types: Sequence[str] = ("sql",),
    dialect: str = "postgres",
    session_id: str | None = None,
    history: list[ConversationTurn] | None = None,
    timeout_seconds: float | None = None,
    extra_dynamic_context: str | None = None,
    explain: bool = False,
) -> AgentQueryResult:
    """Drive MultiAgentOrchestrator to completion, collecting all yielded
    tokens and the final structured chunk, then return an AgentQueryResult.

    Follow-up references in *question* are resolved via
    :func:`~nlqueries.orchestrator.followup_resolver.resolve_followup` before
    the question is passed to the orchestrator.

    Args:
        question:        Natural-language question from the user.
        agent_id:        Agent identifier (used for KB lookup and Qdrant collection).
        available_types: Agent types enabled for this agent (default: ``("sql",)``).
        dialect:         SQL dialect forwarded to the SQL sub-agent.
        session_id:      Optional session identifier — passed through to the result.
        history:         Prior conversation turns for follow-up resolution.
        timeout_seconds: Cancellation budget (Task 26.5 — Sprint 26), forwarded
                         to the SQL sub-agent's ``execute_query`` call as a
                         ``SET LOCAL statement_timeout`` on Postgres, so a
                         runaway query is aborted server-side instead of
                         continuing to run — and hold locks/connections —
                         after the caller (e.g. an API request that already
                         timed out and returned) has given up waiting on it.
                         Callers should pass a value at or slightly below
                         their own outer wait budget. Does **not** currently
                         bound the LLM call itself — see the connectors
                         referenced in ``DatabaseConnector.execute_query``'s
                         docstring for which dialects honor this.
        extra_dynamic_context: Optional guidance appended to the SQL prompt's
                         dynamic (non-cached) block. The enterprise layer uses
                         this to inject a Nexus join-paths / Beacons section;
                         core stays agnostic to its content.

    Returns:
        :class:`AgentQueryResult` with all streamed tokens joined and structured
        fields extracted from the final chunk.  ``latency_ms`` is the wall-clock
        time from function entry to return. ``from_cache`` (Task 27.3) is
        ``True`` when the answer was served from the semantic cache rather
        than dispatched to an LLM/SQL/document sub-agent.
    """
    start = time.monotonic()

    # Resolve follow-up references before routing.
    resolved = resolve_followup(question, history or [])
    resolved_question = resolved.resolved

    # Drive the orchestrator to completion; collect every yielded token. When
    # ``explain`` is set, bind a provenance collector for the run so each
    # orchestrator site records what it contributed (no-op / None otherwise, so
    # behaviour is unchanged when not asked for).
    orchestrator = MultiAgentOrchestrator()
    tokens: list[str] = []
    prov: Provenance | None = Provenance() if explain else None
    with use_provenance(prov):
        async for token in orchestrator.handle_question(
            resolved_question,
            agent_id,
            available_types=list(available_types),
            dialect=dialect,
            history=None,  # already resolved above; avoids double LLM call
            cache_key=question,  # original question — consistent key regardless of LLM rewrite
            timeout_seconds=timeout_seconds,
            extra_dynamic_context=extra_dynamic_context,
        ):
            tokens.append(token)

    # Split text tokens from the final structured chunk.
    text_tokens, agent_type, sql, citations, merged_answer, sql_result, from_cache = (
        _parse_final_chunk(tokens)
    )

    # Natural-language answer: join text tokens; for hybrid use merged_answer.
    answer = merged_answer if agent_type == "hybrid" and merged_answer else "".join(text_tokens)

    latency_ms = int((time.monotonic() - start) * 1000)
    if prov is not None:
        # The route is authoritative from the final chunk's agent_type; record the
        # end-to-end wall-clock so timings has at least the total even when no
        # sub-phase reported one.
        if prov.route is None:
            prov.route = agent_type
        prov.timings.setdefault("total_ms", float(latency_ms))

    return AgentQueryResult(
        question=question,
        resolved_question=resolved_question,
        agent_type=agent_type,
        answer=answer,
        sql=sql,
        sql_result=sql_result,
        citations=citations,
        merged_answer=merged_answer,
        latency_ms=latency_ms,
        session_id=session_id,
        from_cache=from_cache,
        provenance=prov,
    )


def run_query_sync(
    question: str,
    agent_id: str,
    **kwargs: Any,
) -> AgentQueryResult:
    """Synchronous wrapper: calls ``asyncio.run(run_query(...))``.

    Convenience for SDK users who do not have an async runtime.  Cannot be
    called from within a running event loop — use :func:`run_query` there.
    """
    return asyncio.run(run_query(question, agent_id, **kwargs))
