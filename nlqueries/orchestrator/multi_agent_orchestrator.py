"""
nlqueries.orchestrator.multi_agent_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Routes questions to the SQL agent, Document agent, or Hybrid based on
LLM-classified intent.

Sprint 13 update: the hybrid branch runs SQL and Document agents
**concurrently** via ``asyncio.gather()`` and merges their results via
:func:`~nlqueries.orchestrator.result_merger.merge_results`.

Sprint 21 update: a semantic cache (backed by Qdrant) is checked before
dispatching. Cache hits bypass the LLM entirely and return the stored
answer word-by-word.  Cache misses run normally and store the result so
subsequent similar questions can be served from cache.

2026-07-01: previously implemented as a LangGraph ``StateGraph``. Replaced
with plain async dispatch — the graph was a straightforward linear/branching
flow (classify → dispatch → merge) with no persistence, subgraphs, or
LangChain LLM wrappers, so it didn't need the dependency. This removes the
`langgraph`/`langchain_core` import that broke on Python 3.14 (pydantic.v1
compatibility shim) — see docs/troubleshooting.md#w6 for background. Document
ingestion has since also dropped its LangChain dependency (``langchain_text_splitters``
replaced with ``nlqueries.document_connectors.chunker``), so that same
Python-version constraint no longer applies anywhere in the project; see
docs/connectors.md#document-connectors.

Public API
----------
``MultiAgentOrchestrator``
    Call ``handle_question(question, agent_id, available_types, dialect)``
    to get an async token stream.  The final chunk includes ``"agent_type"``
    so the UI can display which agent answered.

Routing::

    classify_intent()
        ├─ sql      → _run_sql()
        ├─ document → _run_document()
        ├─ hybrid   → _run_hybrid()   (SQL + Document concurrently, then merged)
        └─ unclear  → error chunk
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any

from nlqueries.cache.envelope import binding_for_agent
from nlqueries.cache.semantic_cache import CacheEntry, SemanticCache
from nlqueries.connectors.base import QueryResult
from nlqueries.execution import DEFAULT_POLICY, ExecutionPolicy
from nlqueries.orchestrator.conversation import ConversationTurn
from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator
from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult
from nlqueries.orchestrator.followup_resolver import aresolve_followup
from nlqueries.orchestrator.intent_classifier import IntentType, aclassify_intent, coerce_intent
from nlqueries.orchestrator.orchestrator import Orchestrator, _json_default, sql_table_chunk
from nlqueries.orchestrator.provenance import (
    record_cache,
    record_intent_confidence,
    record_route,
)
from nlqueries.orchestrator.result_merger import HybridQueryResult, merge_results
from nlqueries.sql_policy import evaluate

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal carrier used to pass result fields to SemanticCache.put()
# without importing AgentQueryResult (which would create a circular import).
# ---------------------------------------------------------------------------


@dataclass
class _CacheData:
    """Minimal carrier whose attributes match the AgentQueryResult Protocol."""

    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_citations_from_tokens(tokens: list[str]) -> list[Citation] | None:
    """Parse Citation objects from the final JSON chunk in a document token list."""
    if not tokens:
        return None
    try:
        last = json.loads(tokens[-1])
        if last.get("type") == "citations":
            return [
                Citation(
                    chunk_id="",
                    source_name=str(c.get("source_name", "")),
                    page_number=c.get("page_number"),
                    chunk_index=0,
                    excerpt=str(c.get("excerpt", "")),
                    relevance_score=0.0,
                )
                for c in last.get("citations", [])
            ]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None


def _final_sql_chunk(tokens: list[str]) -> dict[str, Any] | None:
    """The SQL sub-agent's final JSON chunk, when its last token is one."""
    if not tokens:
        return None
    try:
        last = json.loads(tokens[-1])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(last, dict) or last.get("type") != "sql" or not last.get("sql"):
        return None
    return last


def _executed_table(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """The executed result inside *chunk*, or ``None`` when nothing ran.

    One definition of "executed", because two callers depend on it and must not
    drift: the extraction below, and the cache-write decision in
    ``handle_question``, which declines to store prose built from live figures.
    Asked of the chunk rather than of the resulting ``QueryResult``, whose
    fallback is identified only by its column being named ``sql_query`` -- a
    query that genuinely selected a column of that name would be misread.
    """
    table = chunk.get("sql_table")
    if isinstance(table, dict) and table.get("columns"):
        return table
    return None


def _carries_executed_data(tokens: list[str]) -> bool:
    """Whether the SQL sub-agent ran a query and returned columns for it."""
    chunk = _final_sql_chunk(tokens)
    return chunk is not None and _executed_table(chunk) is not None


def _extract_sql_query_result(tokens: list[str]) -> QueryResult | None:
    """Return the SQL sub-agent's executed result, or the statement if none ran.

    This used to read only the ``sql`` key and build a one-column, one-row table
    from the statement text, discarding the rows the sub-agent had already
    executed and reported in ``sql_table``. A hybrid answer therefore asserted
    things about data it never showed, and could not report truncation at all,
    because ``truncated`` and ``truncation_reason`` live on the frame it threw
    away.

    The SQL-only table is kept for the cases where nothing ran -- generate-only
    mode, an execution the policy refused, or a statement that failed validation
    -- because the synthesis prompt is still better with the statement in it
    than with nothing. An execution error is carried into that fallback rather
    than dropped: ``sql_table`` is ``{"error": ...}`` with no columns when the
    connector raised, and reporting that as "nothing ran" would be a lie.
    """
    chunk = _final_sql_chunk(tokens)
    if chunk is None:
        return None

    table = _executed_table(chunk)
    if table is not None:
        return QueryResult(
            columns=list(table["columns"]),
            rows=[list(row) for row in table.get("rows") or []],
            row_count=int(table.get("row_count") or 0),
            execution_time_ms=float(table.get("execution_time_ms") or 0.0),
            error=table.get("error"),
            truncated=bool(table.get("truncated")),
            truncation_reason=table.get("truncation_reason"),
        )

    # The raw value, not `_executed_table`'s: that returns None precisely when
    # the table carries an error and no columns, which is the case this reads.
    raw_table = chunk.get("sql_table")
    if isinstance(raw_table, dict) and raw_table.get("error"):
        error = str(raw_table["error"])
    elif not chunk.get("is_valid"):
        error = str(chunk.get("validation_error", ""))
    else:
        error = None
    return QueryResult(
        columns=["sql_query"],
        rows=[[chunk["sql"]]],
        row_count=1,
        execution_time_ms=0.0,
        error=error,
    )


# ---------------------------------------------------------------------------
# Background-task registry
# Used for fire-and-forget cache writes so they don't block the response stream.
# ---------------------------------------------------------------------------

_background_tasks: set[asyncio.Task[None]] = set()


async def drain_background_tasks() -> None:
    """Await all in-flight background cache writes.

    Call this in tests after consuming the generator to ensure cache writes
    have completed before making assertions.  Also useful for graceful
    shutdown in the MCP server.
    """
    if _background_tasks:
        await asyncio.gather(*list(_background_tasks), return_exceptions=True)


def _schedule_cache_write(
    cache: SemanticCache,
    key: str,
    resolved_q: str,
    agent_type_str: str,
    tokens: list[str],
    payload_extra: dict[str, str] | None = None,
) -> None:
    """Extract result fields from *tokens* and write to *cache* in a background task.

    *payload_extra* (when given) is stored in the cache-entry payload so a later
    ``get(..., payload_filter=...)`` can scope the hit — see seam S2.
    """
    # Extract answer (all text tokens) and sql (from final JSON chunk).
    #
    # Only SQL that PASSED validation is stored. The frame carries both `sql` and
    # `is_valid`, and taking the former without the latter caches a statement the
    # orchestrator itself refused to run — which is how a single prose answer
    # ("I can't help with that.") poisoned an agent's cache, after which every
    # cache hit re-executed that prose against the customer's database.
    answer = ""
    sql: str | None = None
    if tokens:
        text_parts: list[str] = []
        for t in tokens:
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict) and "type" in parsed:
                    sql = (parsed.get("sql") or None) if parsed.get("is_valid") else None
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            text_parts.append(t)
        answer = "".join(text_parts)

    data = _CacheData(
        resolved_question=resolved_q,
        agent_type=agent_type_str,
        answer=answer,
        sql=sql,
    )

    async def _write() -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, functools.partial(cache.put, key, data, payload_extra=payload_extra)
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("Semantic cache write failed: %s", exc)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop (e.g. in sync test context)

    task = loop.create_task(_write())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _is_final_chunk(token: str) -> bool:
    """Return True if *token* is the orchestrator's terminal structured JSON chunk."""
    try:
        parsed = json.loads(token)
        return isinstance(parsed, dict) and "type" in parsed
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Agent runners (previously LangGraph nodes; now plain async helpers)
# ---------------------------------------------------------------------------


async def _run_sql(
    question: str, agent_id: str, dialect: str, execution: ExecutionPolicy = DEFAULT_POLICY
) -> list[str]:
    # Passed down, never constructed here. A sub-agent that could decide its own
    # permission would be a way to launder one: route to it, and the answer to
    # "may this run" changes without the caller's intent changing.
    orch = Orchestrator()
    tokens: list[str] = []
    async for token in orch.handle_question(
        question, agent_id, dialect=dialect, execution=execution
    ):
        tokens.append(token)
    return tokens


async def _run_document(question: str, agent_id: str) -> tuple[list[str], list[Citation] | None]:
    orch = DocumentOrchestrator()
    collection = f"doc_{agent_id}_chunks"
    tokens: list[str] = []
    async for token in orch.handle_question(question, collection):
        tokens.append(token)
    citations = _extract_citations_from_tokens(tokens)
    return tokens, citations


async def _run_hybrid(
    question: str,
    agent_id: str,
    dialect: str,
    timeout_seconds: float | None = None,
    extra_dynamic_context: str | None = None,
    execution: ExecutionPolicy = DEFAULT_POLICY,
) -> tuple[list[str], list[str], list[Citation] | None]:
    """Run SQL and Document agents concurrently via ``asyncio.gather``."""
    sql_orch = Orchestrator()
    doc_orch = DocumentOrchestrator()
    collection = f"doc_{agent_id}_chunks"

    async def _collect_sql() -> list[str]:
        tokens: list[str] = []
        async for token in sql_orch.handle_question(
            question,
            agent_id,
            dialect=dialect,
            timeout_seconds=timeout_seconds,
            extra_dynamic_context=extra_dynamic_context,
            execution=execution,
        ):
            tokens.append(token)
        return tokens

    async def _collect_doc() -> list[str]:
        tokens: list[str] = []
        async for token in doc_orch.handle_question(question, collection):
            tokens.append(token)
        return tokens

    sql_tokens, doc_tokens = await asyncio.gather(_collect_sql(), _collect_doc())
    citations = _extract_citations_from_tokens(doc_tokens)
    return sql_tokens, doc_tokens, citations


def _merge_hybrid(
    question: str,
    agent_id: str,
    sql_tokens: list[str],
    citations: list[Citation] | None,
) -> HybridQueryResult:
    sql_query_result = _extract_sql_query_result(sql_tokens) if sql_tokens else None

    doc_retrieval: DocumentRetrievalResult | None = None
    if citations:
        doc_retrieval = DocumentRetrievalResult(
            chunks=[],
            citations=citations,
            collection=f"doc_{agent_id}_chunks",
        )

    return merge_results(question, sql_result=sql_query_result, document_result=doc_retrieval)


def _is_executable_select(sql: str, dialect: str) -> bool:
    """Return ``True`` when the SQL policy permits *sql* for *dialect*.

    A last gate in front of someone else's database, and the only one on the
    cache-replay path: a hit reaches this with a stored statement and no model
    in front of it.

    Delegates to :func:`nlqueries.sql_policy.evaluate`, which replaced the
    root-node check performed here. That check asked only whether the statement
    parsed as a ``Select``, which every payload in ``tests/security/payloads``
    satisfies. It also used ``parse_one``, which returns the first statement and
    discards the rest.

    The policy fails closed on an unrecognised dialect name for the reason this
    function already recorded: ``mssql`` is a registered ``db_type`` and is not
    a sqlglot dialect name, so parsing with a fallback grammar was reachable
    through a supported connector. It is logged at ERROR because an unknown
    dialect is a configuration problem an operator can fix, and should not be
    diagnosed by noticing that cached queries quietly stopped running.
    """
    if not sql or not sql.strip():
        return False

    decision = evaluate(sql, dialect)
    if not decision.allowed:
        _log.error(
            "Cached SQL was not executed. Policy %s refused it: %s",
            decision.policy_version,
            decision.summary(),
        )
    return decision.allowed


async def _execute_cached_sql(
    agent_id: str,
    sql: str,
    timeout_seconds: float | None,
    dialect: str = "postgres",
    execution: ExecutionPolicy = DEFAULT_POLICY,
) -> dict[str, Any] | None:
    """Execute *sql* for *agent_id* and return a ``sql_table`` dict (or ``None``).

    The semantic cache stores the generated SQL and the answer text but not the
    result rows (which would go stale), so a cache hit must re-run the query to
    return current data. This mirrors ``Orchestrator.handle_question``'s
    execution block so a cache-hit ``sql`` frame carries the same ``sql_table``
    shape as a fresh one — the expensive LLM SQL-generation is still skipped;
    only the cheap query execution runs. Best-effort: any failure is surfaced as
    an ``{"error": ...}`` table rather than dropping the result entirely.

    Cached SQL is re-checked before it runs. The write side now stores only
    validated SQL, but entries written before that fix are still live in
    deployed caches, and this is the one path that sends a stored statement to a
    customer's database with no generation step in front of it. Requiring a
    parseable ``SELECT`` keeps a poisoned entry — prose, or anything that is not
    a read — from being executed on their server.
    """
    from nlqueries.connectors.loader import open_connector_for_agent  # noqa: PLC0415

    # A cache hit is a shortcut past generation, not past permission. This path
    # is the one that sends a stored statement to a customer's database with no
    # model in front of it, so it asks the same question the fresh path asks.
    if not execution.may_execute:
        return None

    if not _is_executable_select(sql, dialect):
        return {"error": "Cached SQL failed revalidation and was not executed"}

    try:
        connector = await asyncio.to_thread(open_connector_for_agent, agent_id, execution)
        if connector is None:
            return None
        qr = await asyncio.to_thread(connector.execute_query, sql, timeout_seconds)
        return sql_table_chunk(qr)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """Orchestrator for NLQueries multi-agent routing.

    Routes a question to the SQL agent, Document agent, or both (hybrid)
    based on LLM-classified intent, then streams the response with an
    ``"agent_type"`` field in the final JSON chunk.

    Sprint 13: the hybrid branch runs both agents concurrently via
    ``asyncio.gather()`` and synthesises a unified answer via
    :func:`~nlqueries.orchestrator.result_merger.merge_results`.

    Sprint 21: a semantic cache is consulted before dispatching. Cache hits
    bypass the LLM and return the stored answer immediately.

    Exposes the same async-generator interface as :class:`Orchestrator`.

    Routing::

        classify_intent()
            ├─ sql      → _run_sql()
            ├─ document → _run_document()
            ├─ hybrid   → _run_hybrid()   (SQL + Document concurrently, merged)
            └─ unclear  → error chunk
    """

    def __init__(self) -> None:
        pass

    async def handle_question(
        self,
        question: str,
        agent_id: str,
        available_types: Sequence[str] = ("sql",),
        dialect: str = "postgres",
        history: list[ConversationTurn] | None = None,
        cache_key: str | None = None,
        timeout_seconds: float | None = None,
        extra_dynamic_context: str | None = None,
        intent_override: str | None = None,
        cache_context: dict[str, str] | None = None,
        execution: ExecutionPolicy = DEFAULT_POLICY,
    ) -> AsyncGenerator[str, None]:
        """Route *question* to the appropriate agent and stream the response.

        Yields LLM reasoning tokens followed by a single structured JSON chunk.
        The final chunk includes an ``"agent_type"`` field so the UI can display
        which agent answered.

        For hybrid queries both agents run concurrently via ``asyncio.gather()``.
        The final chunk has ``"type": "hybrid"``::

            {"type": "hybrid", "agent_type": "hybrid",
             "merged_answer": "...", "sql_table": {...}, "citations": [...]}

        When *history* is provided, the question is first passed through
        :func:`~nlqueries.orchestrator.followup_resolver.resolve_followup` so
        that pronoun and contextual references are resolved into a fully
        self-contained question before intent classification.

        **Extension-point guarantees (seam S3, relied on by embedders).** These
        are contractual, not incidental — see ``docs`` and the pinned regression
        tests in ``test_multi_agent_orchestrator.py``:

        * Passing ``history=None`` (or an empty list) bypasses
          ``resolve_followup`` entirely — no follow-up LLM call is made and the
          question is used verbatim. An embedder that resolves follow-ups
          out-of-band (e.g. the enterprise Conversation Context Engine) sets
          ``history=None`` to suppress the built-in resolver.
        * ``extra_dynamic_context`` is injected into the SQL agent's *dynamic*
          prompt block only (never the cached static schema block), so it never
          invalidates the prompt cache. Used to feed caller context (Nexus,
          conversation context) into generation.
        * ``intent_override`` (seam S1) short-circuits ``classify_intent``: when
          it is a valid :class:`IntentType` value it is coerced against
          *available_types* (via ``coerce_intent``) and used directly, saving the
          classifier LLM call. An invalid value falls back to normal
          classification (fail-open). ``None`` leaves behaviour unchanged.
        * ``cache_context`` (seam S2) is stored in the cache entry on write and
          required as an exact payload match on read, so a caller can scope an
          entry (e.g. a follow-up to one conversation context). ``None`` leaves
          caching unchanged.

        Sprint 21: the resolved question is looked up in the semantic cache
        before dispatching. On a cache hit the stored answer is yielded
        word-by-word and the final JSON chunk includes ``"from_cache": True``.
        On a miss the result is stored in the cache after dispatch completes.

        Args:
            question:        Natural-language question from the user.
            agent_id:        Identifier of the agent (used for KB and Qdrant collection).
            available_types: Agent types enabled for this agent.
                             Defaults to ``("sql",)`` for backward compatibility.
            dialect:         SQL dialect forwarded to the SQL agent.
            history:         Prior conversation turns for follow-up resolution.
                             ``None`` (default) disables follow-up resolution.
            timeout_seconds: Forwarded to the SQL agent's ``execute_query``
                             call (Task 26.5 — Sprint 26) so a runaway query
                             is aborted server-side rather than left running
                             orphaned after the caller has given up. Not
                             applied to the LLM call itself.

        Yields:
            String tokens from the agent response, then a final JSON chunk
            with ``"agent_type"`` set to ``"sql"``, ``"document"``, or ``"hybrid"``.
        """
        resolved = await aresolve_followup(question, history or [])
        effective_question = resolved.resolved

        # ------------------------------------------------------------------
        # Embed the question and check the semantic cache — off the event loop.
        #
        # Both are synchronous and both do network I/O: embed_text makes a
        # urllib call to the embedding daemon, and SemanticCache.get performs up
        # to three Qdrant round trips plus a second embed on a full miss. Called
        # bare inside this async generator they froze the entire uvicorn worker,
        # so every other WebSocket on it — every other user's chat turn — stopped
        # for the duration. That is the difference between roughly 3 concurrent
        # turns per worker and roughly 15.
        #
        # One thread hop, not two: the embed's result is the cache lookup's
        # input, so splitting them into separate to_thread calls would pay the
        # hop twice for work that is strictly sequential anyway.
        #
        # The vector is reused for the cache lookup AND passed through to
        # assemble_prompt() inside the SQL orchestrator, saving a second
        # embed_text() round-trip on every cache miss (Phase 1C).
        # ------------------------------------------------------------------
        # Bound to this agent, connector, dialect, schema and policy version, so a
        # forged or stale entry does not verify and is treated as a miss.
        _cache = SemanticCache(agent_id, binding=binding_for_agent(agent_id, dialect))
        # Use caller-supplied cache_key when provided (e.g. run_query passes
        # the original pre-resolution question so repeated identical queries
        # always hit the same cache entry regardless of LLM rewrite variance).
        _cache_lookup_key = cache_key if cache_key is not None else effective_question

        def _embed_and_lookup() -> tuple[list[float] | None, CacheEntry | None]:
            vector: list[float] | None = None
            try:
                from nlqueries.embeddings.embedder import (  # noqa: PLC0415
                    embed_text as _embed_text,
                )

                vector = _embed_text(effective_question)
            except Exception:  # noqa: BLE001 — a missing embedder degrades to a text-only lookup
                pass
            return vector, _cache.get(
                _cache_lookup_key, vector=vector, payload_filter=cache_context
            )

        _question_vector, _cached = await asyncio.to_thread(_embed_and_lookup)
        if _cached is None:
            record_cache(hit=False)  # provenance (SYL-1.1); a hit records tier+score itself
        if _cached is not None:
            # Serve cached answer word-by-word to simulate streaming.
            for _word in _cached.answer.split():
                yield _word + " "
            # Build a final structured chunk that mirrors the live format.
            _hit_chunk: dict[str, Any] = {
                "agent_type": _cached.agent_type,
                "from_cache": True,
            }
            if _cached.agent_type == "sql":
                _hit_chunk["type"] = "sql"
                if _cached.sql:
                    _hit_chunk["sql"] = _cached.sql
                    # Re-execute the cached SQL so the replayed frame carries a
                    # fresh result table (the cache stores SQL + answer text but
                    # not rows). Without this, a cache hit streams an answer with
                    # no ``sql_table``, and table-rendering callers (enterprise
                    # chat, CLI, MCP) show no data for repeated queries.
                    _hit_chunk["sql_table"] = await _execute_cached_sql(
                        agent_id, _cached.sql, timeout_seconds, dialect, execution
                    )
            elif _cached.agent_type == "document":
                _hit_chunk["type"] = "citations"
                _hit_chunk["citations"] = []
            elif _cached.agent_type == "hybrid":
                _hit_chunk["type"] = "hybrid"
                _hit_chunk["merged_answer"] = _cached.answer
            # default=_json_default coerces Decimal/date values in the freshly
            # re-executed sql_table (raw DB-driver types) the same way the live
            # SQL path does — a plain json.dumps would raise on them.
            yield json.dumps(_hit_chunk, default=_json_default)
            return

        # ------------------------------------------------------------------
        # Cache miss: classify intent and dispatch
        # ------------------------------------------------------------------
        # Seam S1: a caller-supplied intent_override skips classify_intent when it
        # names a valid IntentType (coerced to an available type the same way the
        # classifier's own output is). An unparseable value falls through to the
        # normal path (fail-open). The single-agent fast path still wins so a
        # one-type agent never pays for classification.
        override_intent: IntentType | None = None
        if intent_override is not None:
            try:
                override_intent = coerce_intent(IntentType(intent_override), list(available_types))
            except ValueError:
                override_intent = None

        if len(available_types) == 1 and available_types[0] in ("sql", "document"):
            intent = IntentType(available_types[0])
            _log.debug("Intent classification skipped: single agent type %s", intent)
        elif override_intent is not None:
            intent = override_intent
            _log.debug("Intent classification skipped: caller override %s", intent)
        else:
            classification = await aclassify_intent(effective_question, list(available_types))
            intent = classification.intent
            record_intent_confidence(classification.confidence)  # provenance (SYL-1.1)
        record_route(intent.value)  # provenance (SYL-1.1)

        # ------------------------------------------------------------------
        # Dispatch: single-agent branches stream tokens through immediately;
        # hybrid must buffer both sub-streams before merging.
        # Cache writes are scheduled as background tasks (fire-and-forget).
        # ------------------------------------------------------------------

        if intent == IntentType.sql:
            orch = Orchestrator()
            seen: list[str] = []
            async for token in orch.handle_question(
                effective_question,
                agent_id,
                dialect=dialect,
                question_vector=_question_vector,
                timeout_seconds=timeout_seconds,
                extra_dynamic_context=extra_dynamic_context,
                execution=execution,
            ):
                seen.append(token)
                if _is_final_chunk(token):
                    try:
                        last = json.loads(token)
                        last["agent_type"] = "sql"
                        yield json.dumps(last)
                    except (json.JSONDecodeError, ValueError):
                        yield token
                else:
                    yield token
            _schedule_cache_write(
                _cache, _cache_lookup_key, effective_question, "sql", seen, cache_context
            )
            return

        if intent == IntentType.document:
            doc_orch = DocumentOrchestrator()
            collection = f"doc_{agent_id}_chunks"
            seen = []
            async for token in doc_orch.handle_question(effective_question, collection):
                seen.append(token)
                if _is_final_chunk(token):
                    try:
                        last = json.loads(token)
                        last["agent_type"] = "document"
                        yield json.dumps(last)
                    except (json.JSONDecodeError, ValueError):
                        yield token
                else:
                    yield token
            _schedule_cache_write(
                _cache, _cache_lookup_key, effective_question, "document", seen, cache_context
            )
            return

        hybrid_result: HybridQueryResult | None = None
        citations: list[Citation] | None = None

        if intent == IntentType.hybrid:
            sql_tokens, document_tokens, citations = await _run_hybrid(
                effective_question,
                agent_id,
                dialect,
                timeout_seconds,
                extra_dynamic_context,
                execution,
            )
            hybrid_result = _merge_hybrid(effective_question, agent_id, sql_tokens, citations)

            # Cache write for hybrid (background task), but only when the
            # prose carries no live figures.
            #
            # A hybrid entry stores the merged answer and nothing else -- no
            # statement, no citations -- and a hit replays it verbatim for the
            # whole TTL. The SQL branch re-runs its stored statement instead,
            # for the reason `_execute_cached_sql` records: stored prose cannot
            # carry fresh rows. That was harmless while a hybrid answer was
            # synthesised from the statement text alone. Now that it is
            # synthesised from executed rows, caching it would answer a
            # semantically similar question hours later with figures from the
            # first run, stated as current -- the same shape of defect this
            # change exists to fix, and not worth trading for a cache hit.
            #
            # Answers where nothing ran still cache, and they are the ones the
            # cache was helping: generate-only mode, a refused execution, a
            # statement that failed validation. Re-synthesising on a hit would
            # keep the rest, but it needs the statement and the citations stored
            # too, and a synthesis call on every hit -- a larger change, and one
            # to make deliberately rather than as a side effect of this one.
            if hybrid_result is not None and not _carries_executed_data(sql_tokens):
                _schedule_cache_write(
                    _cache,
                    _cache_lookup_key,
                    effective_question,
                    "hybrid",
                    [hybrid_result.merged_answer or ""],
                    cache_context,
                )

        if intent == IntentType.hybrid and hybrid_result is not None:
            sql_table_dict = None
            if hybrid_result.sql_table is not None:
                # cap=False: this branch has never applied the row cap, and
                # starting to would be a different change from reporting
                # truncation.
                #
                # This frame now holds the sub-agent's executed table --
                # `_extract_sql_query_result` returns it rather than the
                # one-cell table of statement text it used to synthesise -- so
                # the truncation flags here describe the query the answer was
                # built from, which is what they read as. The statement-only
                # table survives for the cases where nothing ran, and it reports
                # no truncation because there was none.
                #
                # This comment described the opposite until the change that made
                # it untrue; a reader of this branch reaches it before the
                # function it describes.
                sql_table_dict = sql_table_chunk(hybrid_result.sql_table, cap=False)
            citations_list = [
                {
                    "source_name": c.source_name,
                    "page_number": c.page_number,
                    "excerpt": c.excerpt,
                }
                for c in hybrid_result.citations
            ]
            # The statement this answer was built from, under the same key the
            # `sql` frame uses.
            #
            # It used to be recoverable from `sql_table`, which held one cell
            # containing the statement text. Returning the executed table
            # instead is what a hybrid answer should show, and it removed the
            # only route to the statement at the same time -- so a consumer that
            # scope-checks or audits the query has nothing to read, while the
            # frame now carries real rows. Both halves of that are new; the
            # combination is the one worth avoiding.
            #
            # Taken from the sub-agent's own final chunk rather than from
            # `sql_table`, so it is present whether the query ran or not, and
            # `None` only when the sub-agent produced no statement at all.
            final_sql_chunk = _final_sql_chunk(sql_tokens)
            yield json.dumps(
                {
                    "type": "hybrid",
                    "agent_type": "hybrid",
                    "merged_answer": hybrid_result.merged_answer,
                    "sql": final_sql_chunk.get("sql") if final_sql_chunk else None,
                    "sql_table": sql_table_dict,
                    "citations": citations_list,
                }
            )

        else:
            yield json.dumps(
                {
                    "type": "error",
                    "error": "Intent unclear or unavailable",
                    "agent_type": "unclear",
                }
            )
