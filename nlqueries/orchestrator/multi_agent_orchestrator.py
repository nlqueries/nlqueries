"""
nlqueries.orchestrator.multi_agent_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph-based multi-agent orchestrator — routes questions to the SQL agent,
Document agent, or Hybrid (stub) based on LLM-classified intent.

Public API
----------
``MultiAgentOrchestrator``
    Call ``handle_question(question, agent_id, available_types, dialect)``
    to get an async token stream.  The final chunk includes ``"agent_type"``
    so the UI can display which agent answered.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator
from nlqueries.orchestrator.document_retrieval import Citation
from nlqueries.orchestrator.intent_classifier import IntentType, classify_intent
from nlqueries.orchestrator.orchestrator import Orchestrator


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

    citations: list[Citation] | None = None
    if tokens:
        try:
            last = json.loads(tokens[-1])
            if last.get("type") == "citations":
                citations = [
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

    return {"document_result": json.dumps(tokens), "citations": citations}


async def _merge_node(state: AgentState) -> dict[str, Any]:
    """Stub merge for Sprint 12; real cross-source merging added in Sprint 13."""
    intent = state.get("intent")
    if intent == IntentType.hybrid:
        # Sprint 12 stub: prefer SQL result when both agents ran
        return {"final_answer": state.get("sql_result")}
    if intent == IntentType.sql:
        return {"final_answer": state.get("sql_result")}
    if intent == IntentType.document:
        return {"final_answer": state.get("document_result")}
    return {"final_answer": None}


# ---------------------------------------------------------------------------
# Routing functions (used as conditional edges)
# ---------------------------------------------------------------------------


def _route_after_classify(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == IntentType.document:
        return "document_node"
    if intent in (IntentType.sql, IntentType.hybrid):
        return "sql_node"
    return "merge_node"  # unclear → skip agents, merge with no result


def _route_after_sql(state: AgentState) -> str:
    """After sql_node: also run document_node for hybrid intent."""
    if state.get("intent") == IntentType.hybrid:
        return "document_node"
    return "merge_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:  # type: ignore[type-arg]
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    builder.add_node("classify_intent_node", _classify_intent_node)
    builder.add_node("sql_node", _sql_node)
    builder.add_node("document_node", _document_node)
    builder.add_node("merge_node", _merge_node)

    builder.set_entry_point("classify_intent_node")

    builder.add_conditional_edges(
        "classify_intent_node",
        _route_after_classify,
        {
            "sql_node": "sql_node",
            "document_node": "document_node",
            "merge_node": "merge_node",
        },
    )
    builder.add_conditional_edges(
        "sql_node",
        _route_after_sql,
        {
            "document_node": "document_node",
            "merge_node": "merge_node",
        },
    )
    builder.add_edge("document_node", "merge_node")
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

    Exposes the same async-generator interface as :class:`Orchestrator`.

    Graph topology::

        START → classify_intent_node
            ├─ sql      → sql_node → merge_node → END
            ├─ document → document_node → merge_node → END
            ├─ hybrid   → sql_node → document_node → merge_node → END
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
    ) -> AsyncGenerator[str, None]:
        """Route *question* to the appropriate agent and stream the response.

        Yields LLM reasoning tokens followed by a single structured JSON
        chunk.  The final chunk includes an ``"agent_type"`` field so the
        UI can display which agent answered::

            {"type": "sql", "sql": "...", "agent_type": "sql", ...}
            {"type": "citations", "citations": [...], "agent_type": "document"}

        Args:
            question:        Natural-language question from the user.
            agent_id:        Identifier of the agent (used for KB and Qdrant collection).
            available_types: Agent types enabled for this agent.
                             Defaults to ``("sql",)`` for backward compatibility.
            dialect:         SQL dialect forwarded to the SQL agent.

        Yields:
            String tokens from the agent response, then a final JSON chunk
            with ``"agent_type"`` set to ``"sql"``, ``"document"``, or ``"hybrid"``.
        """
        initial_state: AgentState = {
            "question": question,
            "agent_id": agent_id,
            "available_types": list(available_types),
            "dialect": dialect,
            "intent": None,
            "sql_result": None,
            "document_result": None,
            "citations": None,
            "final_answer": None,
            "error": None,
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
            # Sprint 12 stub: yield SQL result with hybrid agent_type
            # Sprint 13 will replace this with real cross-source merging
            raw = final_state.get("sql_result")
            if raw:
                tokens = json.loads(raw)
                for token in tokens[:-1]:
                    yield token
                if tokens:
                    last = json.loads(tokens[-1])
                    last["agent_type"] = "hybrid"
                    yield json.dumps(last)

        else:
            yield json.dumps(
                {
                    "type": "error",
                    "error": "Intent unclear or unavailable",
                    "agent_type": "unclear",
                }
            )
