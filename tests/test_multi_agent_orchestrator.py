"""Tests for nlqueries.orchestrator.multi_agent_orchestrator."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from nlqueries.orchestrator.intent_classifier import IntentClassificationResult, IntentType
from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQL_TOKENS = [
    "Let me query the database.",
    json.dumps(
        {
            "type": "sql",
            "sql": "SELECT count(*) FROM orders",
            "is_valid": True,
            "validation_error": None,
            "dialect": "postgres",
            "attempt_count": 1,
        }
    ),
]

_DOC_TOKENS = [
    "The refund policy states that...",
    json.dumps(
        {
            "type": "citations",
            "citations": [
                {
                    "source_name": "policy.pdf",
                    "page_number": 3,
                    "excerpt": "Full refund within 30 days.",
                }
            ],
        }
    ),
]


def _classify_result(intent: IntentType) -> IntentClassificationResult:
    return IntentClassificationResult(intent=intent, confidence=0.95, reasoning="Mock.")


def _async_gen_factory(tokens: list[str]):  # type: ignore[return]
    """Return an async generator function that yields *tokens*."""

    async def _gen(*_args: object, **_kwargs: object):  # type: ignore[return]
        for t in tokens:
            yield t

    return _gen


# ---------------------------------------------------------------------------
# Spec-required tests
# ---------------------------------------------------------------------------


class TestMultiAgentOrchestratorSpec:
    """The three tests explicitly required by the Task 12.2 spec."""

    def test_sql_question_routes_to_sql_node(self) -> None:
        """When intent=sql, SQL orchestrator is called; document orchestrator is not."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> tuple[list[str], MagicMock]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.sql),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"
                ) as MockDocOrch,
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "How many orders did we get last month?",
                    "agent1",
                    available_types=["sql"],
                ):
                    tokens.append(token)
                return tokens, MockDocOrch

        tokens, MockDocOrch = asyncio.run(run())

        # SQL tokens must appear in output
        assert any("sql" in t for t in tokens)
        # Document orchestrator must NOT have been instantiated or called
        MockDocOrch.assert_not_called()

    def test_document_question_routes_to_document_node(self) -> None:
        """When intent=document, document orchestrator is called; SQL orchestrator is not."""

        doc_instance = MagicMock()
        doc_instance.handle_question = _async_gen_factory(_DOC_TOKENS)

        async def run() -> tuple[list[str], MagicMock]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.document),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                    return_value=doc_instance,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"
                ) as MockSqlOrch,
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "What does the refund policy say?",
                    "agent1",
                    available_types=["document"],
                ):
                    tokens.append(token)
                return tokens, MockSqlOrch

        tokens, MockSqlOrch = asyncio.run(run())

        # Citations chunk must appear in output
        assert any('"citations"' in t for t in tokens)
        # SQL orchestrator must NOT have been instantiated or called
        MockSqlOrch.assert_not_called()

    def test_final_chunk_includes_agent_type(self) -> None:
        """The final yielded JSON chunk includes an 'agent_type' field."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.sql),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "Count orders", "agent1", available_types=["sql"]
                ):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        final = json.loads(tokens[-1])
        assert "agent_type" in final
        assert final["agent_type"] == "sql"


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestMultiAgentOrchestratorExtra:
    def test_sql_agent_type_in_final_chunk(self) -> None:
        """Final SQL chunk carries agent_type='sql' and preserves SQL fields."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.sql),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question("Count orders", "agent1"):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        final = json.loads(tokens[-1])
        assert final["agent_type"] == "sql"
        assert final["type"] == "sql"
        assert "sql" in final

    def test_document_agent_type_in_final_chunk(self) -> None:
        """Final document chunk carries agent_type='document'."""

        doc_instance = MagicMock()
        doc_instance.handle_question = _async_gen_factory(_DOC_TOKENS)

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.document),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                    return_value=doc_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "What is the policy?", "agent1", available_types=["document"]
                ):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        final = json.loads(tokens[-1])
        assert final["agent_type"] == "document"
        assert final["type"] == "citations"

    def test_text_tokens_yielded_before_final_chunk(self) -> None:
        """LLM reasoning tokens are yielded before the structured final chunk."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.sql),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question("Count orders", "agent1"):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        # First token is the reasoning text, last token is the JSON chunk
        assert tokens[0] == "Let me query the database."
        final = json.loads(tokens[-1])
        assert final["type"] == "sql"

    def test_unclear_intent_yields_error_chunk(self) -> None:
        """When intent is unclear, an error JSON chunk is yielded."""

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.unclear),
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                # Use multiple types so the fast-path (single-type bypass) is
                # not taken and classify_intent is actually called.
                async for token in orch.handle_question(
                    "...", "agent1", available_types=["sql", "document"]
                ):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        assert len(tokens) == 1
        final = json.loads(tokens[0])
        assert final["type"] == "error"
        assert final["agent_type"] == "unclear"

    def test_hybrid_routes_to_sql_then_document(self) -> None:
        """Hybrid intent runs both SQL and Document agents concurrently (Sprint 13)."""
        from nlqueries.orchestrator.result_merger import HybridQueryResult

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        doc_instance = MagicMock()
        doc_instance.handle_question = _async_gen_factory(_DOC_TOKENS)

        sql_called: list[bool] = []
        doc_called: list[bool] = []

        mock_hybrid = HybridQueryResult(
            sql_answer="SQL rows",
            sql_table=None,
            document_answer="Doc excerpts",
            citations=[],
            merged_answer="Merged hybrid answer.",
        )

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.hybrid),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    side_effect=lambda: (sql_called.append(True), sql_instance)[1],
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                    side_effect=lambda: (doc_called.append(True), doc_instance)[1],
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.merge_results",
                    return_value=mock_hybrid,
                ),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "Which customers from Q3 report haven't ordered yet?",
                    "agent1",
                    available_types=["sql", "document", "hybrid"],
                ):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())
        assert sql_called, "SQL orchestrator must be called for hybrid intent"
        assert doc_called, "Document orchestrator must be called for hybrid intent"
        # Sprint 13: hybrid returns a unified hybrid chunk, not the SQL stub
        final = json.loads(tokens[-1])
        assert final["agent_type"] == "hybrid"
        assert final["type"] == "hybrid"

    def test_handle_question_returns_async_generator(self) -> None:
        """handle_question returns an async generator."""
        from collections.abc import AsyncGenerator

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> bool:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                    return_value=_classify_result(IntentType.sql),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                orch = MultiAgentOrchestrator()
                gen = orch.handle_question("Count orders", "agent1")
                return isinstance(gen, AsyncGenerator)

        assert asyncio.run(run())

    def test_multiagentorchestrator_exported_from_package(self) -> None:
        """MultiAgentOrchestrator is importable from the top-level orchestrator package."""
        from nlqueries.orchestrator import MultiAgentOrchestrator as MAO

        assert MAO is MultiAgentOrchestrator
