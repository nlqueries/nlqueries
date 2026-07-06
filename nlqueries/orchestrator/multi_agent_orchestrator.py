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
import json
import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any

from nlqueries.cache.semantic_cache import SemanticCache
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.conversation import ConversationTurn
from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator
from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult
from nlqueries.orchestrator.followup_resolver import resolve_followup
from nlqueries.orchestrator.intent_classifier import IntentType, classify_intent
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.result_merger import HybridQueryResult, merge_results

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


def _extract_sql_query_result(tokens: list[str]) -> QueryResult | None:
    """Parse the SQL final chunk from a token list and return a minimal QueryResult.

    Creates a single-column, single-row table so that ``merge_results`` can
    include the generated SQL in its synthesis prompt.
    """
    if not tokens:
        return None
    try:
        last = json.loads(tokens[-1])
        if last.get("type") == "sql" and last.get("sql"):
            return QueryResult(
                columns=["sql_query"],
                rows=[[last["sql"]]],
                row_count=1,
                execution_time_ms=0.0,
                error=(None if last.get("is_valid") else str(last.get("validation_error", ""))),
            )
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None


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
) -> None:
    """Extract result fields from *tokens* and write to *cache* in a background task."""
    # Extract answer (all text tokens) and sql (from final JSON chunk).
    answer = ""
    sql: str | None = None
    if tokens:
        text_parts: list[str] = []
        for t in tokens:
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict) and "type" in parsed:
                    sql = parsed.get("sql") or None
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
            await loop.run_in_executor(None, cache.put, key, data)
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


async def _run_sql(question: str, agent_id: str, dialect: str) -> list[str]:
    orch = Orchestrator()
    tokens: list[str] = []
    async for token in orch.handle_question(question, agent_id, dialect=dialect):
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
    question: str, agent_id: str, dialect: str
) -> tuple[list[str], list[str], list[Citation] | None]:
    """Run SQL and Document agents concurrently via ``asyncio.gather``."""
    sql_orch = Orchestrator()
    doc_orch = DocumentOrchestrator()
    collection = f"doc_{agent_id}_chunks"

    async def _collect_sql() -> list[str]:
        tokens: list[str] = []
        async for token in sql_orch.handle_question(question, agent_id, dialect=dialect):
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

        Yields:
            String tokens from the agent response, then a final JSON chunk
            with ``"agent_type"`` set to ``"sql"``, ``"document"``, or ``"hybrid"``.
        """
        resolved = resolve_followup(question, history or [])
        effective_question = resolved.resolved

        # ------------------------------------------------------------------
        # Pre-compute the question embedding once.
        # The same vector is reused by the semantic cache lookup AND passed
        # through to assemble_prompt() inside the SQL orchestrator, saving a
        # second embed_text() round-trip on every cache miss (Phase 1C).
        # ------------------------------------------------------------------
        _question_vector: list[float] | None = None
        try:
            from nlqueries.embeddings.embedder import embed_text as _embed_text  # noqa: PLC0415

            _question_vector = _embed_text(effective_question)
        except Exception:  # noqa: BLE001
            pass

        # ------------------------------------------------------------------
        # Semantic cache check (Sprint 21)
        # ------------------------------------------------------------------
        _cache = SemanticCache(agent_id)
        # Use caller-supplied cache_key when provided (e.g. run_query passes
        # the original pre-resolution question so repeated identical queries
        # always hit the same cache entry regardless of LLM rewrite variance).
        _cache_lookup_key = cache_key if cache_key is not None else effective_question
        _cached = _cache.get(_cache_lookup_key, vector=_question_vector)
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
            elif _cached.agent_type == "document":
                _hit_chunk["type"] = "citations"
                _hit_chunk["citations"] = []
            elif _cached.agent_type == "hybrid":
                _hit_chunk["type"] = "hybrid"
                _hit_chunk["merged_answer"] = _cached.answer
            yield json.dumps(_hit_chunk)
            return

        # ------------------------------------------------------------------
        # Cache miss: classify intent and dispatch
        # ------------------------------------------------------------------
        # Fast path: exactly one agent type available — no LLM call needed.
        if len(available_types) == 1 and available_types[0] in ("sql", "document"):
            intent = IntentType(available_types[0])
            _log.debug("Intent classification skipped: single agent type %s", intent)
        else:
            classification = classify_intent(effective_question, list(available_types))
            intent = classification.intent

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
            _schedule_cache_write(_cache, _cache_lookup_key, effective_question, "sql", seen)
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
            _schedule_cache_write(_cache, _cache_lookup_key, effective_question, "document", seen)
            return

        hybrid_result: HybridQueryResult | None = None
        citations: list[Citation] | None = None

        if intent == IntentType.hybrid:
            sql_tokens, document_tokens, citations = await _run_hybrid(
                effective_question, agent_id, dialect
            )
            hybrid_result = _merge_hybrid(effective_question, agent_id, sql_tokens, citations)

            # Cache write for hybrid (background task)
            if hybrid_result is not None:
                _schedule_cache_write(
                    _cache,
                    _cache_lookup_key,
                    effective_question,
                    "hybrid",
                    [hybrid_result.merged_answer or ""],
                )

        if intent == IntentType.hybrid and hybrid_result is not None:
            sql_table_dict = None
            if hybrid_result.sql_table is not None:
                qt = hybrid_result.sql_table
                sql_table_dict = {
                    "columns": qt.columns,
                    "rows": qt.rows,
                    "row_count": qt.row_count,
                    "execution_time_ms": qt.execution_time_ms,
                    "error": qt.error,
                }
            citations_list = [
                {
                    "source_name": c.source_name,
                    "page_number": c.page_number,
                    "excerpt": c.excerpt,
                }
                for c in hybrid_result.citations
            ]
            yield json.dumps(
                {
                    "type": "hybrid",
                    "agent_type": "hybrid",
                    "merged_answer": hybrid_result.merged_answer,
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
