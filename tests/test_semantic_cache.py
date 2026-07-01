"""Tests for nlqueries.cache.semantic_cache (Task 21.1).

All Qdrant client calls are mocked — no live Qdrant instance is required.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from nlqueries.cache.semantic_cache import (
    SIMILARITY_THRESHOLD,
    CacheEntry,
    SemanticCache,
    _point_id_for_question,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored_point(
    score: float,
    payload: dict[str, Any],
    point_id: int = 1,
) -> MagicMock:
    """Build a minimal ScoredPoint-like mock."""
    pt = MagicMock()
    pt.score = score
    pt.payload = payload
    pt.id = point_id
    return pt


def _make_query_response(points: list[Any]) -> MagicMock:
    resp = MagicMock()
    resp.points = points
    return resp


def _fresh_payload(
    question: str = "How many orders?",
    resolved: str = "How many orders?",
    agent_type: str = "sql",
    answer: str = "There were 42 orders.",
    sql: str | None = "SELECT COUNT(*) FROM orders",
    hours_old: int = 0,
) -> dict[str, Any]:
    created_at = datetime.now(UTC) - timedelta(hours=hours_old)
    return {
        "question": question,
        "resolved_question": resolved,
        "agent_type": agent_type,
        "answer": answer,
        "sql": sql,
        "created_at": created_at.isoformat(),
        "hit_count": 0,
    }


@dataclass
class _FakeResult:
    """Minimal AgentQueryResult-like carrier for SemanticCache.put()."""

    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None


# ---------------------------------------------------------------------------
# test_cache_hit_above_threshold_returns_entry
# ---------------------------------------------------------------------------


class TestCacheHitAboveThreshold:
    def test_returns_entry(self) -> None:
        """A Qdrant hit with score >= SIMILARITY_THRESHOLD returns a CacheEntry."""
        payload = _fresh_payload()
        scored_point = _make_scored_point(score=SIMILARITY_THRESHOLD, payload=payload, point_id=42)

        _coll = MagicMock()
        _coll.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [_coll]
        mock_client.query_points.return_value = _make_query_response([scored_point])
        mock_client.set_payload.return_value = None

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch(
                "nlqueries.cache.semantic_cache.embed_text",
                return_value=[0.1] * 384,
            ),
        ):
            cache = SemanticCache("agent1")
            entry = cache.get("How many orders?")

        assert entry is not None
        assert isinstance(entry, CacheEntry)
        assert entry.agent_type == "sql"
        assert entry.answer == "There were 42 orders."
        assert entry.sql == "SELECT COUNT(*) FROM orders"
        assert entry.hit_count == 1  # incremented from 0

    def test_hit_count_incremented_in_qdrant(self) -> None:
        """set_payload is called with the incremented hit_count on a cache hit."""
        payload = _fresh_payload()
        payload["hit_count"] = 4
        scored_point = _make_scored_point(score=0.99, payload=payload, point_id=7)

        _coll2 = MagicMock()
        _coll2.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [_coll2]
        mock_client.query_points.return_value = _make_query_response([scored_point])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            cache = SemanticCache("agent1")
            entry = cache.get("How many orders?")

        assert entry is not None
        assert entry.hit_count == 5
        mock_client.set_payload.assert_called_once_with(
            collection_name="cache_agent1",
            payload={"hit_count": 5},
            points=[7],
        )


# ---------------------------------------------------------------------------
# test_cache_miss_below_threshold_returns_none
# ---------------------------------------------------------------------------


class TestCacheMissBelowThreshold:
    def test_below_threshold_returns_none(self) -> None:
        """A Qdrant hit with score < SIMILARITY_THRESHOLD returns None."""
        payload = _fresh_payload()
        scored_point = _make_scored_point(score=SIMILARITY_THRESHOLD - 0.01, payload=payload)

        _coll_a = MagicMock()
        _coll_a.name = "cache_a"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [_coll_a]
        mock_client.query_points.return_value = _make_query_response([scored_point])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            cache = SemanticCache("a")
            entry = cache.get("Something different")

        assert entry is None
        mock_client.set_payload.assert_not_called()

    def test_no_points_returns_none(self) -> None:
        """Empty Qdrant response returns None."""
        _coll_a2 = MagicMock()
        _coll_a2.name = "cache_a"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [_coll_a2]
        mock_client.query_points.return_value = _make_query_response([])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            cache = SemanticCache("a")
            entry = cache.get("Anything")

        assert entry is None

    def test_missing_collection_returns_none(self) -> None:
        """If the cache collection does not exist yet, get() returns None silently."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []  # no collections

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            cache = SemanticCache("new_agent")
            entry = cache.get("Any question")

        assert entry is None
        mock_client.query_points.assert_not_called()

    def test_expired_entry_returns_none(self) -> None:
        """An entry older than ttl_hours returns None even if score is high."""
        payload = _fresh_payload(hours_old=25)  # 25 h > default 24 h TTL
        scored_point = _make_scored_point(score=0.99, payload=payload)

        _coll_a3 = MagicMock()
        _coll_a3.name = "cache_a"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [_coll_a3]
        mock_client.query_points.return_value = _make_query_response([scored_point])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            cache = SemanticCache("a", ttl_hours=24)
            entry = cache.get("How many orders?")

        assert entry is None


# ---------------------------------------------------------------------------
# test_put_uses_deterministic_point_id
# ---------------------------------------------------------------------------


class TestPutDeterministicPointId:
    def test_same_question_same_point_id(self) -> None:
        """Two put() calls with the same question produce the same Qdrant point ID."""
        question = "How many active users are there?"
        id1 = _point_id_for_question(question)
        id2 = _point_id_for_question(question)
        assert id1 == id2

    def test_different_questions_different_ids(self) -> None:
        """Different questions produce different point IDs (no collision for trivial cases)."""
        id1 = _point_id_for_question("How many orders?")
        id2 = _point_id_for_question("How many users?")
        assert id1 != id2

    def test_put_calls_upsert_with_correct_point_id(self) -> None:
        """put() derives the point ID from SHA-256(question) and passes it to upsert."""
        question = "Total revenue last month?"
        expected_id = _point_id_for_question(question)

        result = _FakeResult(
            resolved_question=question,
            agent_type="sql",
            answer="Revenue was $1M.",
            sql="SELECT SUM(amount) FROM sales",
        )

        mock_client = MagicMock()

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            cache = SemanticCache("agent1")
            cache.put(question, result)

        upsert_call = mock_client.upsert.call_args
        assert upsert_call is not None
        points_arg = upsert_call.kwargs.get("points") or upsert_call.args[1]
        assert len(points_arg) == 1
        assert points_arg[0].id == expected_id


# ---------------------------------------------------------------------------
# test_invalidate_deletes_collection
# ---------------------------------------------------------------------------


class TestListEntries:
    def _make_record(self, question: str, agent_type: str = "sql", hit_count: int = 0) -> Any:
        record = MagicMock()
        record.payload = {
            "question": question,
            "resolved_question": question,
            "agent_type": agent_type,
            "answer": f"Answer to: {question}",
            "sql": f"SELECT * FROM t WHERE q = '{question}'",
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": hit_count,
        }
        return record

    def test_returns_entries_sorted_newest_first(self) -> None:
        """list_entries() returns CacheEntry objects sorted by created_at descending."""
        from datetime import timedelta

        older = self._make_record("How many users?")
        older.payload["created_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        newer = self._make_record("How many orders?")
        newer.payload["created_at"] = datetime.now(UTC).isoformat()

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([older, newer], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1").list_entries()

        assert len(entries) == 2
        assert entries[0].question == "How many orders?"  # newer first
        assert entries[1].question == "How many users?"

    def test_empty_collection_returns_empty_list(self) -> None:
        """list_entries() returns [] when the collection has no points."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1").list_entries()

        assert entries == []

    def test_qdrant_error_returns_empty_list(self) -> None:
        """list_entries() returns [] silently if Qdrant is unreachable."""
        mock_client = MagicMock()
        mock_client.scroll.side_effect = RuntimeError("connection refused")

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1").list_entries()

        assert entries == []

    def test_respects_limit(self) -> None:
        """list_entries() passes the limit parameter to scroll()."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            SemanticCache("agent1").list_entries(limit=10)

        mock_client.scroll.assert_called_once()
        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs.get("limit") == 10


class TestInvalidate:
    def test_invalidate_calls_delete_collection(self) -> None:
        """invalidate() deletes the agent's cache collection from Qdrant."""
        mock_client = MagicMock()

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            cache = SemanticCache("sales-agent")
            cache.invalidate("sales-agent")

        mock_client.delete_collection.assert_called_once_with("cache_sales-agent")

    def test_invalidate_suppresses_errors(self) -> None:
        """invalidate() does not raise even if Qdrant throws."""
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = RuntimeError("Qdrant unavailable")

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            cache = SemanticCache("agent1")
            cache.invalidate("agent1")  # must not raise


# ---------------------------------------------------------------------------
# test_orchestrator_skips_llm_on_cache_hit
# ---------------------------------------------------------------------------


class TestOrchestratorSkipsLlmOnCacheHit:
    def test_llm_not_called_when_cache_returns_hit(self) -> None:
        """MultiAgentOrchestrator.handle_question() skips the LangGraph pipeline
        (and therefore the LLM) when SemanticCache.get() returns a hit."""

        cache_entry = CacheEntry(
            question="How many orders?",
            resolved_question="How many orders?",
            agent_type="sql",
            answer="There were 42 orders.",
            sql="SELECT COUNT(*) FROM orders",
            created_at=datetime.now(UTC),
            hit_count=1,
        )

        collected: list[str] = []

        async def _run() -> None:
            from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

            orch = MultiAgentOrchestrator()
            async for token in orch.handle_question("How many orders?", "agent1"):
                collected.append(token)

        mock_cache = MagicMock()
        mock_cache.get.return_value = cache_entry

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()  # must NOT be called

        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=mock_cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.resolve_followup",
                return_value=MagicMock(resolved="How many orders?"),
            ),
        ):
            # Patch the compiled graph on the instance
            orch_instance_holder: list[Any] = []

            original_init = __import__(
                "nlqueries.orchestrator.multi_agent_orchestrator",
                fromlist=["MultiAgentOrchestrator"],
            ).MultiAgentOrchestrator.__init__

            def _patched_init(self: Any) -> None:
                original_init(self)
                self._graph = mock_graph
                orch_instance_holder.append(self)

            with patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.MultiAgentOrchestrator.__init__",
                _patched_init,
            ):
                asyncio.run(_run())

        # The graph (LLM) must not have been called
        mock_graph.ainvoke.assert_not_called()

        # The answer should be served word-by-word from cache
        full_response = "".join(collected)
        # Final chunk contains agent_type and from_cache
        final_chunk = json.loads(collected[-1])
        assert final_chunk.get("from_cache") is True
        assert final_chunk.get("agent_type") == "sql"
        assert "There were" in full_response


# ---------------------------------------------------------------------------
# test_cache_write_called_from_async_context (regression for #30)
# ---------------------------------------------------------------------------


class TestCacheWriteFromOrchestratorAsyncContext:
    def test_put_called_after_successful_sql_response(self) -> None:
        """Cache.put() is invoked via run_in_executor when the orchestrator produces
        a SQL result inside asyncio.run() — regression test for the sync/async
        conflict where the sync QdrantClient conflicted with the running event loop
        and the write was silently swallowed by contextlib.suppress(Exception)."""
        sql_token = json.dumps(
            {
                "answer": "Five languages exist.",
                "sql": "SELECT COUNT(*) FROM language",
                "agent_type": "sql",
            }
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # force a cache miss

        async def _drive() -> None:
            from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

            orch = MultiAgentOrchestrator()
            async for _ in orch.handle_question("How many languages?", "agent1"):
                pass

        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=mock_cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.resolve_followup",
                return_value=MagicMock(resolved="How many languages?"),
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.classify_intent",
                return_value=MagicMock(intent="sql"),
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator._run_sql",
                new=AsyncMock(return_value=["Five languages ", "exist. ", sql_token]),
            ),
        ):
            asyncio.run(_drive())

        # put() must have been called exactly once with the resolved question
        mock_cache.put.assert_called_once()
        assert mock_cache.put.call_args[0][0] == "How many languages?"


# ---------------------------------------------------------------------------
# Agent-ID sanitization (fix for connector IDs that contain colons)
# ---------------------------------------------------------------------------


class TestAgentIdSanitization:
    def test_connector_id_with_colons_produces_valid_collection(self) -> None:
        """connector IDs like 'postgres:localhost:dvdrental' must not reach Qdrant as-is."""
        cache = SemanticCache("postgres:localhost:dvdrental")
        assert ":" not in cache._collection
        assert cache._collection == "cache_postgres_localhost_dvdrental"

    def test_plain_agent_id_unchanged(self) -> None:
        """Simple IDs like 'dvdrental' must stay unchanged after sanitization."""
        cache = SemanticCache("dvdrental")
        assert cache._collection == "cache_dvdrental"

    def test_slashes_also_sanitized(self) -> None:
        """Slashes (forward slash in some connector URLs) are also replaced."""
        cache = SemanticCache("myschema/mydb")
        assert "/" not in cache._collection
        assert cache._collection == "cache_myschema_mydb"
