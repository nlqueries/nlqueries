"""Tests for nlqueries.orchestrator.multi_agent_orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
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


# ---------------------------------------------------------------------------
# New coverage: fast-path classification, semantic cache, history, embedding
# ---------------------------------------------------------------------------


class TestMultiAgentOrchestratorNewPaths:
    def test_fast_path_skips_classify_intent_for_single_type(self) -> None:
        """classify_intent is skipped when exactly one agent type is available."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        doc_instance = MagicMock()
        doc_instance.handle_question = _async_gen_factory(_DOC_TOKENS)

        async def run_sql_only() -> MagicMock:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent"
                ) as mock_classify,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Count orders", "agent1", available_types=["sql"]
                ):
                    pass
            return mock_classify

        asyncio.run(run_sql_only()).assert_not_called()

        async def run_document_only() -> MagicMock:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent"
                ) as mock_classify,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                    return_value=doc_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "What is the policy?", "agent1", available_types=["document"]
                ):
                    pass
            return mock_classify

        asyncio.run(run_document_only()).assert_not_called()

        async def run_multi_type() -> MagicMock:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                    return_value=_classify_result(IntentType.sql),
                ) as mock_classify,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Count orders", "agent1", available_types=["sql", "document"]
                ):
                    pass
            return mock_classify

        asyncio.run(run_multi_type()).assert_called_once()

    def test_cache_hit_sql_bypasses_llm(self) -> None:
        """A semantic-cache hit for a sql entry yields the cached answer without
        instantiating the SQL Orchestrator."""
        from datetime import UTC, datetime

        from nlqueries.cache.semantic_cache import CacheEntry

        cache_entry = CacheEntry(
            question="Total revenue?",
            resolved_question="Total revenue?",
            agent_type="sql",
            answer="Total revenue was $1.2M",
            sql="SELECT sum(revenue) FROM orders",
            created_at=datetime.now(UTC),
            hit_count=1,
        )

        async def run() -> tuple[list[str], MagicMock]:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator") as MockOrch,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                MockCache.return_value.get.return_value = cache_entry
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question("Total revenue?", "agent1"):
                    tokens.append(token)
                return tokens, MockOrch

        tokens, MockOrch = asyncio.run(run())

        MockOrch.assert_not_called()
        answer_text = "".join(tokens[:-1])
        assert "revenue" in answer_text
        assert "$1.2M" in answer_text
        final = json.loads(tokens[-1])
        assert final["from_cache"] is True
        assert final["agent_type"] == "sql"

    def test_cache_hit_sql_reexecutes_and_attaches_fresh_result_table(self) -> None:
        """A sql cache hit re-runs the cached SQL and attaches a fresh sql_table
        (the cache stores SQL + answer text but not the rows), while still
        skipping the LLM SQL-generation."""
        from datetime import UTC, datetime
        from decimal import Decimal

        from nlqueries.cache.semantic_cache import CacheEntry
        from nlqueries.connectors.base import QueryResult

        cache_entry = CacheEntry(
            question="Total revenue?",
            resolved_question="Total revenue?",
            agent_type="sql",
            answer="Total revenue was $1.2M",
            sql="SELECT sum(revenue) FROM orders",
            created_at=datetime.now(UTC),
            hit_count=1,
        )
        # Decimal mirrors a Postgres numeric column — the frame must still
        # serialize (via _json_default), exactly like the live SQL path.
        qr = QueryResult(
            columns=["sum"],
            rows=[[Decimal("1200000.5")]],
            row_count=1,
            execution_time_ms=3.0,
            error=None,
        )
        fake_connector = MagicMock()
        fake_connector.execute_query.return_value = qr

        async def run() -> dict[str, object]:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator") as MockOrch,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch(
                    "nlqueries.connectors.loader.open_connector_for_agent",
                    return_value=fake_connector,
                ),
            ):
                MockCache.return_value.get.return_value = cache_entry
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question("Total revenue?", "agent1"):
                    tokens.append(token)
                MockOrch.assert_not_called()  # LLM SQL-generation is still skipped
                return json.loads(tokens[-1])

        final = asyncio.run(run())
        assert final["from_cache"] is True
        assert final["sql_table"] is not None
        assert final["sql_table"]["columns"] == ["sum"]
        assert final["sql_table"]["rows"] == [[1200000.5]]  # Decimal → float via _json_default
        assert final["sql_table"]["row_count"] == 1
        assert final["sql_table"]["error"] is None
        fake_connector.execute_query.assert_called_once_with(
            "SELECT sum(revenue) FROM orders", None
        )

    def test_cache_miss_writes_cache_in_background_non_blocking(self) -> None:
        """A cache write on miss must not block the caller from receiving tokens."""
        import time as _time

        from nlqueries.orchestrator.multi_agent_orchestrator import drain_background_tasks

        delay = 0.3
        put_calls: list[tuple[str, Any]] = []

        def slow_put(key: str, data: Any, *, payload_extra: Any = None) -> None:
            _time.sleep(delay)
            put_calls.append((key, data))

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> float:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                MockCache.return_value.get.return_value = None
                MockCache.return_value.put.side_effect = slow_put
                orch = MultiAgentOrchestrator()
                start = _time.perf_counter()
                async for _token in orch.handle_question("Count orders", "agent1"):
                    pass
                elapsed = _time.perf_counter() - start
                await drain_background_tasks()
            return elapsed

        elapsed = asyncio.run(run())

        # Tokens must have been fully yielded well before the slow put() finishes.
        assert elapsed < delay
        assert put_calls, "cache.put should eventually run once drained"

    def test_history_passed_to_resolve_followup(self) -> None:
        """resolve_followup receives *history* when given, and [] when history=None."""
        from datetime import UTC, datetime

        from nlqueries.orchestrator.conversation import ConversationTurn
        from nlqueries.orchestrator.followup_resolver import ResolvedQuestion

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        history = [
            ConversationTurn(
                role="user",
                content="How many orders last month?",
                agent_type=None,
                sql=None,
                timestamp=datetime.now(UTC),
            )
        ]

        async def run(history_arg: list[ConversationTurn] | None) -> MagicMock:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aresolve_followup",
                    return_value=ResolvedQuestion(
                        original="And this month?",
                        resolved="How many orders this month?",
                        is_followup=True,
                        reasoning="Mock.",
                    ),
                ) as mock_resolve,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "And this month?", "agent1", history=history_arg
                ):
                    pass
            return mock_resolve

        mock_resolve_with_history = asyncio.run(run(history))
        mock_resolve_with_history.assert_called_once_with("And this month?", history)

        mock_resolve_default = asyncio.run(run(None))
        mock_resolve_default.assert_called_once_with("And this month?", [])

    def test_question_embedded_exactly_once_per_request(self) -> None:
        """embed_text is called exactly once per handle_question invocation."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> MagicMock:
            with (
                patch(
                    "nlqueries.embeddings.embedder.embed_text",
                    return_value=[0.1] * 384,
                ) as mock_embed,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Count orders", "agent1", available_types=["sql"]
                ):
                    pass
            return mock_embed

        mock_embed = asyncio.run(run())

        mock_embed.assert_called_once()
        assert mock_embed.return_value == [0.1] * 384

    def test_cache_key_override_used_for_cache_lookup(self) -> None:
        """A caller-supplied cache_key is used for the cache lookup, not the question text."""

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> MagicMock:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Count orders", "agent1", cache_key="my-override-key"
                ):
                    pass
            return MockCache.return_value.get

        mock_get = asyncio.run(run())

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "my-override-key"

    def test_cache_hit_document_bypasses_document_orchestrator(self) -> None:
        """A semantic-cache hit for a document entry yields citations-shaped output
        without instantiating the DocumentOrchestrator."""
        from datetime import UTC, datetime

        from nlqueries.cache.semantic_cache import CacheEntry

        cache_entry = CacheEntry(
            question="What is the refund policy?",
            resolved_question="What is the refund policy?",
            agent_type="document",
            answer="Full refund within 30 days.",
            sql=None,
            created_at=datetime.now(UTC),
            hit_count=1,
        )

        async def run() -> tuple[list[str], MagicMock]:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator"),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"
                ) as MockDocOrch,
            ):
                MockCache.return_value.get.return_value = cache_entry
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "What is the refund policy?", "agent1", available_types=["document"]
                ):
                    tokens.append(token)
                return tokens, MockDocOrch

        tokens, MockDocOrch = asyncio.run(run())

        MockDocOrch.assert_not_called()
        final = json.loads(tokens[-1])
        assert final["type"] == "citations"
        assert final["from_cache"] is True
        assert final["agent_type"] == "document"

    def test_cache_hit_hybrid_bypasses_both_orchestrators(self) -> None:
        """A semantic-cache hit for a hybrid entry yields a hybrid-shaped final chunk
        without instantiating either sub-orchestrator."""
        from datetime import UTC, datetime

        from nlqueries.cache.semantic_cache import CacheEntry

        cache_entry = CacheEntry(
            question="Which customers haven't ordered?",
            resolved_question="Which customers haven't ordered?",
            agent_type="hybrid",
            answer="Merged answer from hybrid",
            sql=None,
            created_at=datetime.now(UTC),
            hit_count=1,
        )

        async def run() -> tuple[list[str], MagicMock, MagicMock]:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator") as MockOrch,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"
                ) as MockDocOrch,
            ):
                MockCache.return_value.get.return_value = cache_entry
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "Which customers haven't ordered?",
                    "agent1",
                    available_types=["sql", "document", "hybrid"],
                ):
                    tokens.append(token)
                return tokens, MockOrch, MockDocOrch

        tokens, MockOrch, MockDocOrch = asyncio.run(run())

        MockOrch.assert_not_called()
        MockDocOrch.assert_not_called()
        final = json.loads(tokens[-1])
        assert final["type"] == "hybrid"
        assert final["from_cache"] is True
        assert final["agent_type"] == "hybrid"


# ---------------------------------------------------------------------------
# Extension-point seams for embedders (S1 intent_override, S2 cache_context,
# S3 documented guarantees) — the enterprise Conversation Context Engine relies
# on these, so they are pinned here against a future refactor.
# ---------------------------------------------------------------------------


def _capturing_gen(sink: dict[str, Any], tokens: list[str]):  # type: ignore[return]
    """Async-gen factory that records the args/kwargs it was called with."""

    async def _gen(*args: Any, **kwargs: Any):  # type: ignore[return]
        sink["args"] = args
        sink["kwargs"] = kwargs
        for t in tokens:
            yield t

    return _gen


class TestExtensionPointSeams:
    def test_intent_override_skips_classify_intent(self) -> None:
        """S1: a valid intent_override is used directly; classify_intent isn't called."""
        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> tuple[list[str], MagicMock]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent"
                ) as mock_classify,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch(
                    "nlqueries.embeddings.embedder.embed_text",
                    return_value=[0.0] * 384,
                ),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "Revenue by region",
                    "agent1",
                    available_types=["sql", "document"],  # multi-type: would classify
                    intent_override="sql",
                ):
                    tokens.append(token)
                return tokens, mock_classify

        tokens, mock_classify = asyncio.run(run())
        mock_classify.assert_not_called()
        final = json.loads(tokens[-1])
        assert final["agent_type"] == "sql"

    def test_invalid_intent_override_falls_back_to_classify(self) -> None:
        """S1: an unparseable override fails open — classify_intent runs as usual."""
        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)

        async def run() -> MagicMock:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                    return_value=_classify_result(IntentType.sql),
                ) as mock_classify,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Revenue by region",
                    "agent1",
                    available_types=["sql", "document"],
                    intent_override="not-a-real-intent",
                ):
                    pass
                return mock_classify

        mock_classify = asyncio.run(run())
        mock_classify.assert_called_once()

    def test_history_none_uses_question_verbatim(self) -> None:
        """S3: history=None bypasses resolve_followup — the SQL agent sees the
        original question even when it contains a follow-up signal word."""
        sink: dict[str, Any] = {}
        sql_instance = MagicMock()
        sql_instance.handle_question = _capturing_gen(sink, _SQL_TOKENS)
        question = "show me that instead"  # contains signal words 'that'/'instead'

        async def run() -> None:
            # resolve_followup is NOT patched — the real one must no-op on empty history.
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    question, "agent1", available_types=["sql"], history=None
                ):
                    pass

        asyncio.run(run())
        assert sink["args"][0] == question

    def test_extra_dynamic_context_forwarded_to_sql_agent(self) -> None:
        """S3: extra_dynamic_context reaches the SQL orchestrator unchanged."""
        sink: dict[str, Any] = {}
        sql_instance = MagicMock()
        sql_instance.handle_question = _capturing_gen(sink, _SQL_TOKENS)

        async def run() -> None:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "Count orders",
                    "agent1",
                    available_types=["sql"],
                    extra_dynamic_context="CONVERSATION CONTEXT BLOCK",
                ):
                    pass

        asyncio.run(run())
        assert sink["kwargs"]["extra_dynamic_context"] == "CONVERSATION CONTEXT BLOCK"

    def test_cache_context_scopes_get_and_put(self) -> None:
        """S2: cache_context is used as the get() payload_filter and the put()
        payload_extra, scoping the entry to one conversation context."""
        from nlqueries.orchestrator.multi_agent_orchestrator import drain_background_tasks

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        ctx = {"context_fingerprint": "fp-123"}

        async def run() -> MagicMock:
            with (
                patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as MockCache,
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            ):
                MockCache.return_value.get.return_value = None
                orch = MultiAgentOrchestrator()
                async for _token in orch.handle_question(
                    "only completed", "agent1", available_types=["sql"], cache_context=ctx
                ):
                    pass
                await drain_background_tasks()
                return MockCache.return_value

        cache = asyncio.run(run())
        assert cache.get.call_args.kwargs["payload_filter"] == ctx
        assert cache.put.call_args.kwargs["payload_extra"] == ctx


# ---------------------------------------------------------------------------
# W-2: the cache lookup must not block the event loop
#
# embed_text and SemanticCache.get are both synchronous and both do network
# I/O — a urllib call to the embedding daemon, and up to three Qdrant round
# trips plus a second embed on a full miss. Called bare inside the async
# generator they froze the whole uvicorn worker, so one user's cache lookup
# stopped every other user's chat turn on that worker.
# ---------------------------------------------------------------------------


class TestCacheLookupDoesNotBlockTheEventLoop:
    def _slow_cache(self, delay: float) -> MagicMock:
        """A cache whose get() sleeps synchronously, as a real one does on a miss."""
        import time

        cache = MagicMock()
        cache.get.side_effect = lambda *_a, **_kw: (time.sleep(delay), None)[1]
        return cache

    @contextlib.contextmanager
    def _patched(self, sql_instance: MagicMock, cache: MagicMock):
        """Patch once around the whole run.

        Patching per turn breaks under concurrency: two `patch` contexts on the
        same target form a stack, so the first to exit restores the real object
        while the second is still relying on the mock.
        """
        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        ):
            yield

    async def _turn(self, question: str) -> list[str]:
        tokens: list[str] = []
        async for token in MultiAgentOrchestrator().handle_question(
            question, "agent1", available_types=["sql"]
        ):
            tokens.append(token)
        return tokens

    def test_two_turns_overlap_instead_of_queueing(self) -> None:
        """Two 300 ms lookups must take about 300 ms together, not 600.

        This is the measurement that matters: a blocking lookup does not slow
        the turn that pays for it, it stops every other turn on the worker.
        """
        import time

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        cache = self._slow_cache(0.3)

        async def run() -> float:
            started = time.monotonic()
            await asyncio.gather(self._turn("first question"), self._turn("second question"))
            return time.monotonic() - started

        with self._patched(sql_instance, cache):
            elapsed = asyncio.run(run())

        assert elapsed < 0.5, f"two concurrent 300ms lookups took {elapsed:.3f}s — they serialised"

    def test_an_unrelated_task_keeps_running_during_a_lookup(self) -> None:
        """The direct symptom: while one turn is in its cache lookup, everything
        else on the loop must continue to be scheduled."""
        import time

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        cache = self._slow_cache(0.4)

        async def run() -> int:
            ticks = 0

            async def _heartbeat() -> None:
                nonlocal ticks
                deadline = time.monotonic() + 0.35
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                    ticks += 1

            await asyncio.gather(self._turn("a question"), _heartbeat())
            return ticks

        with self._patched(sql_instance, cache):
            ticks = asyncio.run(run())

        # A blocked loop yields roughly zero ticks; a free one yields ~17.
        assert ticks > 5, f"the event loop only ticked {ticks} times during a 400ms lookup"

    def test_the_vector_still_reaches_the_cache_and_the_prompt(self) -> None:
        """The Phase 1C optimisation must survive the move: the vector computed
        here is handed to the cache lookup and then on to the SQL orchestrator,
        so a miss does not pay for a second embed."""
        sentinel = [0.25] * 384
        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        cache = MagicMock()
        cache.get.return_value = None

        async def run() -> None:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                    return_value=cache,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch("nlqueries.embeddings.embedder.embed_text", return_value=sentinel),
            ):
                async for _token in MultiAgentOrchestrator().handle_question(
                    "how many orders", "agent1", available_types=["sql"]
                ):
                    pass

        asyncio.run(run())

        assert cache.get.call_args.kwargs["vector"] == sentinel

    def test_a_failing_embedder_still_looks_the_cache_up(self) -> None:
        """Text-only lookup is the documented degradation, and it must survive
        being moved into a worker thread."""
        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        cache = MagicMock()
        cache.get.return_value = None

        async def run() -> None:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                    return_value=cache,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
                patch(
                    "nlqueries.embeddings.embedder.embed_text",
                    side_effect=RuntimeError("daemon down"),
                ),
            ):
                async for _token in MultiAgentOrchestrator().handle_question(
                    "how many orders", "agent1", available_types=["sql"]
                ):
                    pass

        asyncio.run(run())

        assert cache.get.called
        assert cache.get.call_args.kwargs["vector"] is None


# ---------------------------------------------------------------------------
# W-3: the auxiliary LLM calls must not block the event loop either
#
# resolve_followup and classify_intent are each a full LLM round trip, one to
# three seconds, and both were called bare inside this async generator. The only
# escape was CONVERSATION_ENGINE_ENABLED=true, which skips them — a workaround
# that is off by default, not a fix.
# ---------------------------------------------------------------------------


class TestAuxiliaryLlmCallsDoNotBlockTheEventLoop:
    def test_two_turns_overlap_through_followup_and_intent(self) -> None:
        """Two turns whose follow-up resolution and classification each take
        300 ms must overlap, not queue: about 600 ms in total rather than 1.2 s.

        History is supplied and no intent_override is given, so both auxiliary
        calls actually run — the configuration this is slowest under, and the
        default one.
        """
        import time

        from nlqueries.orchestrator.conversation import create_session

        session = create_session("agent1")
        session.add_turn("user", "Show orders by region")
        session.add_turn("assistant", "Here they are.")

        async def _slow_resolve(question: str, _history: object) -> object:
            from nlqueries.orchestrator.followup_resolver import ResolvedQuestion

            await asyncio.sleep(0.3)
            return ResolvedQuestion(
                original=question, resolved=question, is_followup=False, reasoning="stub"
            )

        async def _slow_classify(_question: str, _types: object) -> object:
            await asyncio.sleep(0.3)
            return _classify_result(IntentType.sql)

        sql_instance = MagicMock()
        sql_instance.handle_question = _async_gen_factory(_SQL_TOKENS)
        cache = MagicMock()
        cache.get.return_value = None

        async def _turn(question: str) -> None:
            async for _token in MultiAgentOrchestrator().handle_question(
                question,
                "agent1",
                available_types=["sql", "document"],
                history=session.turns,
            ):
                pass

        async def run() -> float:
            started = time.monotonic()
            await asyncio.gather(_turn("first question"), _turn("second question"))
            return time.monotonic() - started

        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aresolve_followup",
                _slow_resolve,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                _slow_classify,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        ):
            elapsed = asyncio.run(run())

        # Serialised this would be ~1.2s; overlapped it is ~0.6s.
        assert elapsed < 0.95, f"two concurrent turns took {elapsed:.3f}s — they serialised"
