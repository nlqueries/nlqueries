"""
nlqueries.orchestrator.multi_agent_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph-based multi-agent orchestrator — routes questions to the SQL agent,
Document agent, or Hybrid based on LLM-classified intent.

Sprint 13 update: the hybrid branch now runs SQL and Document agents
**concurrently** via ``asyncio.gather()`` (replacing the Sprint 12 stub) and
merges their results via :func:`~nlqueries.orchestrator.result_merger.merge_results`.

Public API
----------
``MultiAgentOrchestrator``
    Call ``handle_question(question, agent_id, available_types, dialect)``
    to get an async token stream.  The final chunk includes ``"agent_type"``
    so the UI can display which agent answered.

Graph topology::

    START → classify_intent_node
        ├─ sql      → sql_node              → merge_node → END
        ├─ document → document_node         → merge_node → END
        ├─ hybrid   → parallel_hybrid_node  → merge_node → END
        └─ unclear  → merge_node → END
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.conversation import ConversationTurn
from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator
from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult
from nlqueries.orchestrator.followup_resolver import resolve_followup
from nlqueries.orchestrator.intent_classifier import IntentType, classify_intent
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.result_merger import HybridQueryResult, merge_results


class AgentState(TypedDict):
    question: str
    agent_id: str
    available_types: list[str]
    dialect: str
    intent: IntentType | None
    sql_result: str | None  # JSON-encoded list[str] of collected tokens
    document_result: str | None  # JSON-encoded list[str] of collected tokens
    citations: list[Citation] | None
    final_answer: str | None
    error: str | None
    hybrid_result: HybridQueryResult | None  # populated for hybrid intent (Sprint 13)


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
# Node functions
# ---------------------------------------------------------------------------


async def _classify_intent_node(state: AgentState) -> dict[str, Any]:
    result = classify_intent(state["question"], state["available_types"])
    return {"intent": result.intent}


async def _sql_node(state: AgentState) -> dict[str, Any]:
    orch = Orchestrator()
    tokens: list[str] = []
    async for token in orch.handle_question(
        state["question"],
        state["agent_id"],
        dialect=state["dialect"],
    ):
        tokens.append(token)
    return {"sql_result": json.dumps(tokens)}


async def _document_node(state: AgentState) -> dict[str, Any]:
    orch = DocumentOrchestrator()
    collection = f"doc_{state['agent_id']}_chunks"
    tokens: list[str] = []
    async for token in orch.handle_question(state["question"], collection):
        tokens.append(token)
    citations = _extract_citations_from_tokens(tokens)
    return {"document_result": json.dumps(tokens), "citations": citations}


async def _parallel_hybrid_node(state: AgentState) -> dict[str, Any]:
    """Run SQL and Document agents concurrently via ``asyncio.gather``."""
    sql_orch = Orchestrator()
    doc_orch = DocumentOrchestrator()
    collection = f"doc_{state['agent_id']}_chunks"

    async def _collect_sql() -> str:
        tokens: list[str] = []
        async for token in sql_orch.handle_question(
            state["question"],
            state["agent_id"],
            dialect=state["dialect"],
        ):
            tokens.append(token)
        return json.dumps(tokens)

    async def _collect_doc() -> str:
        tokens: list[str] = []
        async for token in doc_orch.handle_question(state["question"], collection):
            tokens.append(token)
        return json.dumps(tokens)

    sql_json, doc_json = await asyncio.gather(_collect_sql(), _collect_doc())

    doc_tokens: list[str] = json.loads(doc_json)
    citations = _extract_citations_from_tokens(doc_tokens)

    return {
        "sql_result": sql_json,
        "document_result": doc_json,
        "citations": citations,
    }


async def _merge_node(state: AgentState) -> dict[str, Any]:
    """Merge results; for hybrid intent, call ``merge_results`` for LLM synthesis."""
    intent = state.get("intent")

    if intent == IntentType.hybrid:
        sql_raw = state.get("sql_result")
        sql_query_result: QueryResult | None = None
        if sql_raw:
            sql_query_result = _extract_sql_query_result(json.loads(sql_raw))

        doc_retrieval: DocumentRetrievalResult | None = None
        citations = state.get("citations")
        if citations:
            doc_retrieval = DocumentRetrievalResult(
                chunks=[],
                citations=citations,
                collection=f"doc_{state['agent_id']}_chunks",
            )

        hybrid = merge_results(
            state["question"],
            sql_result=sql_query_result,
            document_result=doc_retrieval,
        )
        return {"hybrid_result": hybrid, "final_answer": None}

    if intent == IntentType.sql:
        return {"final_answer": state.get("sql_result"), "hybrid_result": None}
    if intent == IntentType.document:
        return {"final_answer": state.get("document_result"), "hybrid_result": None}
    return {"final_answer": None, "hybrid_result": None}


# ---------------------------------------------------------------------------
# Routing function (conditional edge)
# ---------------------------------------------------------------------------


def _route_after_classify(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == IntentType.document:
        return "document_node"
    if intent == IntentType.hybrid:
        return "parallel_hybrid_node"
    if intent == IntentType.sql:
        return "sql_node"
    return "merge_node"  # unclear → skip agents, merge with no result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:  # type: ignore[type-arg]
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    builder.add_node("classify_intent_node", _classify_intent_node)
    builder.add_node("sql_node", _sql_node)
    builder.add_node("document_node", _document_node)
    builder.add_node("parallel_hybrid_node", _parallel_hybrid_node)
    builder.add_node("merge_node", _merge_node)

    builder.set_entry_point("classify_intent_node")

    builder.add_conditional_edges(
        "classify_intent_node",
        _route_after_classify,
        {
            "sql_node": "sql_node",
            "document_node": "document_node",
            "parallel_hybrid_node": "parallel_hybrid_node",
            "merge_node": "merge_node",
        },
    )
    builder.add_edge("sql_node", "merge_node")
    builder.add_edge("document_node", "merge_node")
    builder.add_edge("parallel_hybrid_node", "merge_node")
    builder.add_edge("merge_node", END)

    return builder


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """LangGraph-based multi-agent orchestrator for NLQueries.

    Routes a question to the SQL agent, Document agent, or both (hybrid)
    based on LLM-classified intent, then streams the response with an
    ``"agent_type"`` field in the final JSON chunk.

    Sprint 13: the hybrid branch now runs both agents concurrently via
    ``asyncio.gather()`` and synthesises a unified answer via
    :func:`~nlqueries.orchestrator.result_merger.merge_results`.

    Exposes the same async-generator interface as :class:`Orchestrator`.

    Graph topology::

        START → classify_intent_node
            ├─ sql      → sql_node              → merge_node → END
            ├─ document → document_node         → merge_node → END
            ├─ hybrid   → parallel_hybrid_node  → merge_node → END
            └─ unclear  → merge_node → END
    """

    def __init__(self) -> None:
        self._graph = _build_graph().compile()

    async def handle_question(
        self,
        question: str,
        agent_id: str,
        available_types: Sequence[str] = ("sql",),
        dialect: str = "postgres",
        history: list[ConversationTurn] | None = None,
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

        initial_state: AgentState = {
            "question": effective_question,
            "agent_id": agent_id,
            "available_types": list(available_types),
            "dialect": dialect,
            "intent": None,
            "sql_result": None,
            "document_result": None,
            "citations": None,
            "final_answer": None,
            "error": None,
            "hybrid_result": None,
        }

        final_state: AgentState = await self._graph.ainvoke(initial_state)  # type: ignore[assignment]

        intent = final_state.get("intent")

        if intent == IntentType.sql:
            raw = final_state.get("sql_result")
            if raw:
                tokens: list[str] = json.loads(raw)
                for token in tokens[:-1]:
                    yield token
                if tokens:
                    last = json.loads(tokens[-1])
                    last["agent_type"] = "sql"
                    yield json.dumps(last)

        elif intent == IntentType.document:
            raw = final_state.get("document_result")
            if raw:
                tokens = json.loads(raw)
                for token in tokens[:-1]:
                    yield token
                if tokens:
                    last = json.loads(tokens[-1])
                    last["agent_type"] = "document"
                    yield json.dumps(last)

        elif intent == IntentType.hybrid:
            hybrid_result = final_state.get("hybrid_result")
            if hybrid_result is not None:
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
