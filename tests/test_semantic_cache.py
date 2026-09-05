"""Tests for nlqueries.cache.semantic_cache (Task 21.1).

All Qdrant client calls are mocked — no live Qdrant instance is required.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.cache.envelope import CacheBinding, sign
from nlqueries.cache.semantic_cache import (
    SIMILARITY_THRESHOLD,
    CacheEntry,
    SemanticCache,
    _bind_entities,
    _mask_entities,
    _normalize_question,
    _point_id_for_question,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


TEST_KEY = b"semantic-cache-test-key"

#: Entries are signed for the context they were produced in and verified before
#: use, so a test payload has to be signed and a cache has to carry the binding.
TEST_BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)


@pytest.fixture(autouse=True)
def _use_the_test_signing_key(monkeypatch):
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: TEST_KEY)


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
    kind: str = "answer",
) -> dict[str, Any]:
    created_at = datetime.now(UTC) - timedelta(hours=hours_old)
    return sign(
        {
            "question": question,
            "resolved_question": resolved,
            "agent_type": agent_type,
            "answer": answer,
            "sql": sql,
            "created_at": created_at.isoformat(),
            "hit_count": 0,
            "kind": kind,
        },
        TEST_BINDING,
        TEST_KEY,
    )


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
            cache = SemanticCache("agent1", binding=TEST_BINDING)
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
            cache = SemanticCache("agent1", binding=TEST_BINDING)
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
            cache = SemanticCache("a", binding=TEST_BINDING)
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
            cache = SemanticCache("a", binding=TEST_BINDING)
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
            cache = SemanticCache("new_agent", binding=TEST_BINDING)
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
            cache = SemanticCache("a", ttl_hours=24, binding=TEST_BINDING)
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
        """put() derives the point ID from SHA-256(normalized question) and passes it to upsert."""
        question = "Total revenue last month?"
        expected_id = _point_id_for_question(_normalize_question(question))

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
            cache = SemanticCache("agent1", binding=TEST_BINDING)
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
    def _make_record(
        self,
        question: str,
        agent_type: str = "sql",
        hit_count: int = 0,
        hours_old: int = 0,
    ) -> Any:
        record = MagicMock()
        # `created_at` is signed, so age is set before signing rather than
        # written over the payload afterwards.
        record.payload = sign(
            {
                "question": question,
                "resolved_question": question,
                "agent_type": agent_type,
                "answer": f"Answer to: {question}",
                "sql": f"SELECT * FROM t WHERE q = '{question}'",
                "created_at": (datetime.now(UTC) - timedelta(hours=hours_old)).isoformat(),
                "hit_count": hit_count,
                "kind": "answer",
            },
            TEST_BINDING,
            TEST_KEY,
        )
        return record

    def test_returns_entries_sorted_newest_first(self) -> None:
        """list_entries() returns CacheEntry objects sorted by created_at descending."""
        older = self._make_record("How many users?", hours_old=2)
        newer = self._make_record("How many orders?")

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([older, newer], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1", binding=TEST_BINDING).list_entries()

        assert len(entries) == 2
        assert entries[0].question == "How many orders?"  # newer first
        assert entries[1].question == "How many users?"

    def test_empty_collection_returns_empty_list(self) -> None:
        """list_entries() returns [] when the collection has no points."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1", binding=TEST_BINDING).list_entries()

        assert entries == []

    def test_qdrant_error_returns_empty_list(self) -> None:
        """list_entries() returns [] silently if Qdrant is unreachable."""
        mock_client = MagicMock()
        mock_client.scroll.side_effect = RuntimeError("connection refused")

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            entries = SemanticCache("agent1", binding=TEST_BINDING).list_entries()

        assert entries == []

    def test_respects_limit(self) -> None:
        """list_entries() passes the limit parameter to scroll()."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            SemanticCache("agent1", binding=TEST_BINDING).list_entries(limit=10)

        mock_client.scroll.assert_called_once()
        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs.get("limit") == 10


class TestInvalidate:
    def test_invalidate_calls_delete_collection(self) -> None:
        """invalidate() deletes the agent's cache collection from Qdrant."""
        mock_client = MagicMock()

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            cache = SemanticCache("sales-agent", binding=TEST_BINDING)
            cache.invalidate("sales-agent")

        mock_client.delete_collection.assert_called_once_with("cache_sales-agent")

    def test_invalidate_suppresses_errors(self) -> None:
        """invalidate() does not raise even if Qdrant throws."""
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = RuntimeError("Qdrant unavailable")

        with patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client):
            cache = SemanticCache("agent1", binding=TEST_BINDING)
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
                "nlqueries.orchestrator.multi_agent_orchestrator.aresolve_followup",
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
        # Final chunk must have "type" so _is_final_chunk() recognises it.
        sql_token = json.dumps(
            {
                "type": "sql",
                "sql": "SELECT COUNT(*) FROM language",
                "is_valid": True,
                "validation_error": None,
                "dialect": "postgres",
                "attempt_count": 1,
            }
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # force a cache miss

        # Mock the inner Orchestrator so it yields our predefined tokens.
        # The SQL branch in MultiAgentOrchestrator directly instantiates Orchestrator()
        # and calls handle_question() — _run_sql is only used in the hybrid path.
        async def _fake_handle(*args: object, **kwargs: object) -> object:
            for t in ["Five languages exist. ", sql_token]:
                yield t

        mock_sql_orch = MagicMock()
        mock_sql_orch.handle_question = _fake_handle

        async def _drive() -> None:
            from nlqueries.orchestrator.multi_agent_orchestrator import (
                MultiAgentOrchestrator,
                drain_background_tasks,
            )

            orch = MultiAgentOrchestrator()
            async for _ in orch.handle_question("How many languages?", "agent1"):
                pass
            # Wait for the fire-and-forget cache write to complete.
            await drain_background_tasks()

        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=mock_cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aresolve_followup",
                return_value=MagicMock(resolved="How many languages?"),
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=mock_sql_orch,
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
        cache = SemanticCache("postgres:localhost:dvdrental", binding=TEST_BINDING)
        assert ":" not in cache._collection
        assert cache._collection == "cache_postgres_localhost_dvdrental"

    def test_plain_agent_id_unchanged(self) -> None:
        """Simple IDs like 'dvdrental' must stay unchanged after sanitization."""
        cache = SemanticCache("dvdrental", binding=TEST_BINDING)
        assert cache._collection == "cache_dvdrental"

    def test_slashes_also_sanitized(self) -> None:
        """Slashes (forward slash in some connector URLs) are also replaced."""
        cache = SemanticCache("myschema/mydb", binding=TEST_BINDING)
        assert "/" not in cache._collection
        assert cache._collection == "cache_myschema_mydb"


# ---------------------------------------------------------------------------
# Phase 1C: pre-computed vector threading in SemanticCache.get()
# ---------------------------------------------------------------------------


class TestGetWithPrecomputedVector:
    def _make_hit_client(self, payload: dict[str, Any]) -> MagicMock:
        pt = _make_scored_point(score=1.0, payload=payload, point_id=1)
        coll = MagicMock()
        coll.name = "cache_agent1"
        client = MagicMock()
        client.get_collections.return_value.collections = [coll]
        client.query_points.return_value = _make_query_response([pt])
        client.set_payload.return_value = None
        return client

    def test_precomputed_vector_skips_embed_text(self) -> None:
        """When vector= is supplied, embed_text must NOT be called."""
        payload = _fresh_payload()
        mock_client = self._make_hit_client(payload)
        precomputed = [0.9] * 384

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch(
                "nlqueries.cache.semantic_cache.embed_text",
                side_effect=AssertionError("embed_text called unexpectedly"),
            ),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get(
                "How many orders?", vector=precomputed
            )

        assert entry is not None

    def test_precomputed_vector_passed_to_qdrant_query(self) -> None:
        """The pre-computed vector must be forwarded to query_points() as the query."""
        payload = _fresh_payload()
        mock_client = self._make_hit_client(payload)
        precomputed = [0.42] * 384

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).get(
                "How many orders?", vector=precomputed
            )

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query"] == precomputed

    def test_no_vector_falls_back_to_embed_text(self) -> None:
        """When vector=None (default), embed_text is called to compute the vector."""
        payload = _fresh_payload()
        mock_client = self._make_hit_client(payload)
        embed_vector = [0.77] * 384

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch(
                "nlqueries.cache.semantic_cache.embed_text",
                return_value=embed_vector,
            ) as mock_embed,
        ):
            SemanticCache("agent1", binding=TEST_BINDING).get("How many orders?")

        mock_embed.assert_called_once_with("How many orders?")
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query"] == embed_vector


# ---------------------------------------------------------------------------
# Phase 5A: helper function unit tests
# ---------------------------------------------------------------------------


class TestNormalizeQuestion:
    def test_lowercases(self) -> None:
        assert _normalize_question("HOW MANY Orders?") == "how many orders"

    def test_strips_punctuation(self) -> None:
        assert _normalize_question("revenue last month?") == "revenue last month"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_question("  how   many   orders  ") == "how many orders"

    def test_identical_questions_same_result(self) -> None:
        q1 = _normalize_question("How many orders?")
        q2 = _normalize_question("how many orders")
        assert q1 == q2


class TestMaskEntities:
    def test_masks_iso_date(self) -> None:
        assert _mask_entities("orders after 2024-01-01") == "orders after <DATE>"

    def test_masks_number(self) -> None:
        assert _mask_entities("revenue > 1000") == "revenue > <NUMBER>"

    def test_masks_month_name(self) -> None:
        result = _mask_entities("orders in January")
        assert "<MONTH>" in result

    def test_masks_currency(self) -> None:
        result = _mask_entities("sales above $500")
        assert "<CURRENCY>" in result

    def test_no_entities_unchanged(self) -> None:
        q = "how many active users are there"
        assert _mask_entities(q) == q

    def test_date_before_number(self) -> None:
        # ISO date should be masked as <DATE>, not as multiple <NUMBER>s
        result = _mask_entities("orders on 2024-03-15")
        assert "<DATE>" in result
        assert "<NUMBER>" not in result


class TestBindEntities:
    def test_binds_date_placeholder(self) -> None:
        """The exact shape `put()` stores, asserted on the whole statement.

        This used to assert only `"2024-06-01" in result`, which the old binder
        satisfied while producing `>= ''2024-06-01''` -- a doubled quote that
        failed to parse on every dialect, so every Tier 2 hit with a date served
        a stale answer beside "Cached SQL failed revalidation". A substring of a
        whole statement is not an assertion about that statement.
        """
        template = "SELECT COUNT(*) FROM orders WHERE order_date >= '[d:DATE]'"
        question = "how many orders after 2024-06-01"
        result = _bind_entities(question, template)
        assert result == "SELECT COUNT(*) FROM orders WHERE order_date >= '2024-06-01'"

    def test_binds_string_value_without_its_quote_characters(self) -> None:
        """`"East"` is the value East, not the five characters `"East"`.

        The entity regexes used to capture the delimiters along with the value,
        and the binder added quotes of its own around them.
        """
        template = "SELECT * FROM sales WHERE region = '[region:VARCHAR]'"
        assert (
            _bind_entities('sales for "East"', template)
            == "SELECT * FROM sales WHERE region = 'East'"
        )
        assert (
            _bind_entities("sales for 'East'", template)
            == "SELECT * FROM sales WHERE region = 'East'"
        )

    def test_escaping_belongs_to_the_dialect_not_to_us(self) -> None:
        """Why values are rendered by sqlglot rather than quoted by hand.

        The correct escape is not one rule. An apostrophe doubles on Postgres,
        MySQL and T-SQL but is backslash-escaped on BigQuery and Snowflake; a
        backslash is doubled on MySQL and left alone on Postgres. Any single
        hand-written quoting rule is wrong on some engine, and wrong here means a
        value stops being a value.
        """
        template = "SELECT * FROM users WHERE name = '[name:VARCHAR]'"
        apostrophe = 'users called "O\'Brien"'

        for dialect in ("postgres", "mysql", "tsql"):
            assert _bind_entities(apostrophe, template, dialect) == (
                "SELECT * FROM users WHERE name = 'O''Brien'"
            ), dialect
        backslash_escaped = "SELECT * FROM users WHERE name = 'O" + chr(92) + "'Brien'"
        for dialect in ("bigquery", "snowflake"):
            assert _bind_entities(apostrophe, template, dialect) == backslash_escaped, dialect

        backslash = 'users called "back' + chr(92) + 'slash"'
        assert _bind_entities(backslash, template, "mysql") == (
            "SELECT * FROM users WHERE name = 'back" + chr(92) * 2 + "slash'"
        )
        assert _bind_entities(backslash, template, "postgres") == (
            "SELECT * FROM users WHERE name = 'back" + chr(92) + "slash'"
        )

    def test_masking_is_unchanged_by_the_capture_groups(self) -> None:
        """Cache keys are built from the masked question, so every entry ever
        written depends on this being byte-identical to what it was."""
        assert _mask_entities('sales for "East" on 2024-06-01 over $1,000') == (
            "sales for <STRING> on <DATE> over <CURRENCY>"
        )
        assert _mask_entities("top 10 for 'West'") == "top <NUMBER> for <STRING>"

    def test_binds_number_int_placeholder(self) -> None:
        template = "SELECT * FROM orders WHERE amount > '[amount:INT]'"
        question = "orders where amount exceeds 500"
        result = _bind_entities(question, template)
        assert result is not None
        assert "500" in result

    def test_returns_none_when_insufficient_entities(self) -> None:
        template = "SELECT * FROM t WHERE d1 = '[d1:DATE]' AND d2 = '[d2:DATE]'"
        question = "only one date 2024-01-01"  # only one DATE entity
        result = _bind_entities(question, template)
        assert result is None

    def test_no_placeholders_returns_sql_unchanged(self) -> None:
        sql = "SELECT COUNT(*) FROM orders"
        result = _bind_entities("how many orders", sql)
        assert result == sql

    def test_binds_varchar_placeholder(self) -> None:
        template = "SELECT * FROM users WHERE status = '[status:VARCHAR]'"
        question = "users with status 'active'"
        result = _bind_entities(question, template)
        assert result is not None
        assert "active" in result

    # --- Semantic pre-assignment (fix for multi-number ordering bugs) ---

    def test_year_column_gets_four_digit_number(self) -> None:
        # "start_year" contains "year" → should receive 2025, not 10 or 10000.
        # Template uses unquoted INT placeholders (as the updated parameterizer generates).
        # A whole statement, not a fragment: binding parses the template now, so
        # `WHERE ...` alone is no longer bindable. Stored templates are always whole
        # statements -- `put()` gets them from `_parameterize_sql`, which returns the
        # SQL unchanged with no placeholders when it cannot parse it.
        template = (
            "SELECT tb.id FROM titles tb JOIN ratings tr ON tr.id = tb.id"
            " WHERE tb.start_year = [start_year:INT]"
            " AND tr.num_votes > [num_votes:INT]"
            " LIMIT [param_1:INT]"
        )
        question = "top 10 highest rated movies from 2025 with more than 10000 reviews"
        result = _bind_entities(question, template)
        assert result is not None
        assert "start_year = 2025" in result
        assert "num_votes > 10000" in result
        assert "LIMIT 10" in result

    def test_param_n_gets_top_n_value(self) -> None:
        template = "SELECT id FROM users LIMIT [param_1:INT]"
        question = "show the top 5 users"
        result = _bind_entities(question, template)
        assert result is not None
        assert "5" in result

    def test_multiple_numbers_no_year_column_uses_positional(self) -> None:
        # When placeholder name has no "year" and no "top N" pattern, positional binding applies.
        template = "SELECT * FROM orders WHERE amount > [amount:INT] AND qty < [qty:INT]"
        question = "orders where amount exceeds 100 and qty less than 50"
        result = _bind_entities(question, template)
        assert result is not None
        assert "100" in result
        assert "50" in result

    def test_year_preassign_does_not_consume_non_year_numbers(self) -> None:
        # After pre-assigning 2024 to start_year, 100 must still be available for num_votes.
        template = (
            "SELECT tb.id FROM titles tb JOIN ratings tr ON tr.id = tb.id"
            " WHERE tb.start_year = [start_year:INT] AND tr.num_votes > [num_votes:INT]"
        )
        question = "movies from 2024 with more than 100 votes"
        result = _bind_entities(question, template)
        assert result is not None
        assert "2024" in result
        assert "100" in result


# ---------------------------------------------------------------------------
# Phase 5A: Tier 0 exact-match lookup
# ---------------------------------------------------------------------------


class TestTier0ExactMatch:
    def _make_retrieve_client(self, payload: dict[str, Any]) -> MagicMock:
        """Build a mock client whose retrieve() returns one record."""
        record = MagicMock()
        record.id = _point_id_for_question(_normalize_question("How many orders?"))
        record.payload = payload
        coll = MagicMock()
        coll.name = "cache_agent1"
        client = MagicMock()
        client.get_collections.return_value.collections = [coll]
        client.retrieve.return_value = [record]
        return client

    def test_tier0_hit_skips_embed_text(self) -> None:
        """A Tier 0 hit must not call embed_text at all."""
        payload = _fresh_payload(kind="answer")
        mock_client = self._make_retrieve_client(payload)

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch(
                "nlqueries.cache.semantic_cache.embed_text",
                side_effect=AssertionError("embed_text must not be called on Tier 0 hit"),
            ),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get("How many orders?")

        assert entry is not None
        assert entry.agent_type == "sql"
        # query_points should NOT be called since Tier 0 returned a hit
        mock_client.query_points.assert_not_called()

    def test_tier0_uses_normalized_question_id(self) -> None:
        """Different capitalizations of the same question share the same Tier 0 ID."""
        id1 = _point_id_for_question(_normalize_question("How many orders?"))
        id2 = _point_id_for_question(_normalize_question("HOW MANY ORDERS"))
        id3 = _point_id_for_question(_normalize_question("how many orders"))
        assert id1 == id2 == id3

    def test_tier0_miss_falls_through_to_tier1(self) -> None:
        """When retrieve() returns nothing, Tier 1 cosine search is attempted."""
        coll = MagicMock()
        coll.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [coll]
        mock_client.retrieve.return_value = []  # Tier 0 miss
        mock_client.query_points.return_value = _make_query_response([])  # Tier 1 miss too

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get("How many orders?")

        assert entry is None
        # Tier 1 query_points must have been called after the Tier 0 miss
        mock_client.query_points.assert_called()


# ---------------------------------------------------------------------------
# Phase 5A: Tier 1 kind=answer filter
# ---------------------------------------------------------------------------


class TestTier1AnswerFilter:
    def test_tier1_passes_kind_answer_filter(self) -> None:
        """Tier 1 must include a kind=answer payload filter in its query_points call."""
        coll = MagicMock()
        coll.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [coll]
        mock_client.retrieve.return_value = []  # Tier 0 miss
        mock_client.query_points.return_value = _make_query_response([])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.5] * 384),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).get("How many orders?")

        call_kwargs = mock_client.query_points.call_args_list[0].kwargs
        q_filter = call_kwargs.get("query_filter")
        assert q_filter is not None
        # The filter must include a MatchAny for "answer"
        filter_str = str(q_filter)
        assert "answer" in filter_str

    def test_tier1_returns_entry_above_threshold(self) -> None:
        """Tier 1 returns a CacheEntry when cosine score meets CACHE_ANSWER_THRESHOLD."""
        payload = _fresh_payload(kind="answer")
        scored_point = _make_scored_point(score=0.98, payload=payload, point_id=55)

        coll = MagicMock()
        coll.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [coll]
        mock_client.retrieve.return_value = []  # Tier 0 miss
        mock_client.query_points.return_value = _make_query_response([scored_point])

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get("How many orders?")

        assert entry is not None
        assert entry.answer == "There were 42 orders."


# ---------------------------------------------------------------------------
# Phase 5A: Tier 2 template cache
# ---------------------------------------------------------------------------


class TestTier2TemplateCache:
    def _make_tier2_client(self, tmpl_payload: dict[str, Any], score: float) -> MagicMock:
        tmpl_point = _make_scored_point(score=score, payload=tmpl_payload, point_id=99)
        coll = MagicMock()
        coll.name = "cache_agent1"
        client = MagicMock()
        client.get_collections.return_value.collections = [coll]
        client.retrieve.return_value = []  # Tier 0 miss
        # First query_points call = Tier 1 (answer, miss); second = Tier 2 (template)
        client.query_points.side_effect = [
            _make_query_response([]),  # Tier 1 miss
            _make_query_response([tmpl_point]),  # Tier 2 hit
        ]
        return client

    def test_tier2_returns_bound_sql_on_hit(self) -> None:
        """A Tier 2 template hit returns a CacheEntry with entity-filled SQL."""
        # A binding whose dialect needs translating, so this exercises the wiring
        # rather than only the happy name: sqlglot rejects `mssql` outright, and
        # `get()` is the only place a dialect is taken from the binding at all.
        binding = replace(TEST_BINDING, dialect="mssql")
        tmpl_payload = sign(
            {
                "question": "orders after <DATE>",
                "resolved_question": "orders after <DATE>",
                "agent_type": "sql",
                "answer": "Found results.",
                # A quoted identifier, so the rendering is dialect-visible: T-SQL
                # writes it in brackets and every other dialect in double quotes.
                # Without it a bound DATE renders identically everywhere and the
                # dialect could be dropped on the way in without failing anything.
                "sql": "SELECT * FROM orders WHERE \"order date\" >= '[d:DATE]'",
                "created_at": datetime.now(UTC).isoformat(),
                "hit_count": 0,
                "kind": "template",
            },
            binding,
            TEST_KEY,
        )
        mock_client = self._make_tier2_client(tmpl_payload, score=0.95)

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=binding).get("orders after 2024-06-01")

        assert entry is not None
        # The whole statement, not a substring of it. `"2024-06-01" in entry.sql`
        # passed just as readily against the old `>= ''2024-06-01''`, which did
        # not parse on any dialect -- the same weak assertion that let the bug
        # this change fixes live in a covered code path.
        assert entry.sql == "SELECT * FROM orders WHERE [order date] >= '2024-06-01'"

    def test_tier2_miss_below_threshold_returns_none(self) -> None:
        """Tier 2 returns None when template score is below CACHE_TEMPLATE_THRESHOLD."""
        tmpl_payload = {
            "question": "orders after <DATE>",
            "resolved_question": "orders after <DATE>",
            "agent_type": "sql",
            "answer": "Found results.",
            "sql": "SELECT * FROM orders WHERE d >= '[d:DATE]'",
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "template",
        }
        mock_client = self._make_tier2_client(tmpl_payload, score=0.75)  # below 0.90

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get("orders after 2024-06-01")

        assert entry is None

    def test_tier2_binding_failure_returns_none(self) -> None:
        """Tier 2 returns None when _bind_entities cannot fill all placeholders."""
        # Template requires TWO dates but question has only ONE
        tmpl_payload = {
            "question": "orders between <DATE> and <DATE>",
            "resolved_question": "orders between <DATE> and <DATE>",
            "agent_type": "sql",
            "answer": "Found results.",
            "sql": "SELECT * FROM orders WHERE d BETWEEN '[d1:DATE]' AND '[d2:DATE]'",
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "template",
        }
        mock_client = self._make_tier2_client(tmpl_payload, score=0.95)

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get(
                "orders after 2024-06-01"
            )  # only 1 date

        assert entry is None

    def test_tier2_skipped_when_no_entities_in_question(self) -> None:
        """Tier 2 is never attempted when the question has no entity tokens."""
        coll = MagicMock()
        coll.name = "cache_agent1"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [coll]
        mock_client.retrieve.return_value = []
        mock_client.query_points.return_value = _make_query_response([])  # Tier 1 miss

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            entry = SemanticCache("agent1", binding=TEST_BINDING).get(
                "how many active users are there"
            )

        assert entry is None
        # Only ONE query_points call (Tier 1); Tier 2 must be skipped
        assert mock_client.query_points.call_count == 1


# ---------------------------------------------------------------------------
# Phase 5A: put() stores kind field and template
# ---------------------------------------------------------------------------


class TestPutStoresKindAndTemplate:
    def test_put_stores_kind_answer(self) -> None:
        """put() must include kind='answer' in the upserted payload."""
        result = _FakeResult(
            resolved_question="How many orders?",
            agent_type="sql",
            answer="42 orders.",
            sql="SELECT COUNT(*) FROM orders",
        )
        mock_client = MagicMock()

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).put("How many orders?", result)

        points = mock_client.upsert.call_args.kwargs.get("points") or []
        answer_point = next(p for p in points if p.payload.get("kind") == "answer")
        assert answer_point is not None

    def test_put_stores_template_for_sql_with_entities(self) -> None:
        """put() also stores a kind='template' point when the SQL has entity literals."""
        result = _FakeResult(
            resolved_question="orders after 2024-06-01",
            agent_type="sql",
            answer="Found 10 orders.",
            sql="SELECT * FROM orders WHERE order_date >= '2024-06-01'",
        )
        mock_client = MagicMock()

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).put("orders after 2024-06-01", result)

        points = mock_client.upsert.call_args.kwargs.get("points") or []
        kinds = [p.payload.get("kind") for p in points]
        assert "answer" in kinds
        assert "template" in kinds

    def test_put_skips_template_when_no_entities(self) -> None:
        """put() stores only the answer point when the question has no entity literals."""
        result = _FakeResult(
            resolved_question="how many active users",
            agent_type="sql",
            answer="There are 5 users.",
            sql="SELECT COUNT(*) FROM users WHERE active = TRUE",
        )
        mock_client = MagicMock()

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).put("how many active users", result)

        points = mock_client.upsert.call_args.kwargs.get("points") or []
        kinds = [p.payload.get("kind") for p in points]
        assert kinds == ["answer"]  # only one point

    def test_put_passes_payload_indexes_to_ensure_collection(self) -> None:
        """put() must request a 'kind' keyword index from ensure_collection."""
        result = _FakeResult(resolved_question="q", agent_type="sql", answer="a", sql=None)

        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=MagicMock()),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection") as mock_ensure,
        ):
            SemanticCache("agent1", binding=TEST_BINDING).put("q", result)

        _, kwargs = mock_ensure.call_args
        assert kwargs.get("payload_indexes") == ["kind"]


# ---------------------------------------------------------------------------
# Seam S2 — payload_extra on put() + payload_filter on get() scope entries
# (mechanism half of bug fix F5: a follow-up cached in one conversation context
# must not replay in another).
# ---------------------------------------------------------------------------


class TestPayloadScopedEntries:
    @staticmethod
    def _client_with_collection() -> MagicMock:
        coll = MagicMock()
        coll.name = "cache_agentX"
        client = MagicMock()
        client.get_collections.return_value.collections = [coll]
        return client

    def test_put_stores_payload_extra_in_every_point(self) -> None:
        """payload_extra keys land in the upserted point payload without clobbering
        the reserved keys."""
        client = self._client_with_collection()
        result = _FakeResult(
            resolved_question="only completed",
            agent_type="sql",
            answer="3",
            sql="SELECT count(*) FROM orders WHERE status = 'completed'",
        )
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            SemanticCache("agentX", binding=TEST_BINDING).put(
                "only completed", result, payload_extra={"context_fingerprint": "fp1"}
            )

        points = client.upsert.call_args.kwargs["points"]
        assert points, "put should upsert at least the answer point"
        for p in points:
            assert p.payload["context_fingerprint"] == "fp1"
            # Reserved keys survive the merge (payload_extra can't overwrite them).
            assert p.payload["kind"] in ("answer", "template")
            assert p.payload["sql"]  # not shadowed by payload_extra

    def test_tier0_hit_requires_matching_payload_filter(self) -> None:
        """An exact-id (Tier 0) hit whose payload matches the filter is returned."""
        client = self._client_with_collection()
        pt = _make_scored_point(
            score=1.0,
            payload={**_fresh_payload(), "context_fingerprint": "fp1"},
        )
        client.retrieve.return_value = [pt]
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            entry = SemanticCache("agentX", binding=TEST_BINDING).get(
                "How many orders?", payload_filter={"context_fingerprint": "fp1"}
            )
        assert entry is not None
        assert entry.agent_type == "sql"

    def test_tier0_foreign_context_is_not_returned(self) -> None:
        """F5: a Tier-0 exact hit from a *different* context is skipped, not replayed."""
        client = self._client_with_collection()
        pt = _make_scored_point(
            score=1.0,
            payload={**_fresh_payload(), "context_fingerprint": "OTHER"},
        )
        client.retrieve.return_value = [pt]
        client.query_points.return_value = _make_query_response([])  # Tier 1 finds nothing
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            entry = SemanticCache("agentX", binding=TEST_BINDING).get(
                "How many orders?", payload_filter={"context_fingerprint": "fp1"}
            )
        assert entry is None

    def test_tier1_query_filter_carries_payload_conditions(self) -> None:
        """The cosine (Tier 1) query filter includes the payload_filter keys."""
        client = self._client_with_collection()
        client.retrieve.return_value = []  # Tier 0 miss
        client.query_points.return_value = _make_query_response([])
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            SemanticCache("agentX", binding=TEST_BINDING).get(
                "q", payload_filter={"context_fingerprint": "fp1"}
            )

        query_filter = client.query_points.call_args.kwargs["query_filter"]
        keys = {cond.key for cond in query_filter.must}
        assert "kind" in keys
        assert "context_fingerprint" in keys

    def test_no_payload_filter_leaves_lookup_unchanged(self) -> None:
        """Without payload_filter, the Tier 1 query filter is just the kind condition."""
        client = self._client_with_collection()
        client.retrieve.return_value = []
        client.query_points.return_value = _make_query_response([])
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            SemanticCache("agentX", binding=TEST_BINDING).get("q")

        query_filter = client.query_points.call_args.kwargs["query_filter"]
        keys = [cond.key for cond in query_filter.must]
        assert keys == ["kind"]


# ---------------------------------------------------------------------------
# W-9: the negative lookup is cached too
#
# Only hits were remembered, so an agent whose collection did not exist yet paid
# a get_collections() round trip on every single query — forever, until its
# first cache write. New agents were slower than warm ones for a reason nobody
# would guess from reading the code.
# ---------------------------------------------------------------------------


class TestNegativeCollectionCache:
    def _client(self, existing: list[str]) -> MagicMock:
        client = MagicMock()
        collections = []
        for name in existing:
            entry = MagicMock()
            entry.name = name  # MagicMock(name=...) is reserved
            collections.append(entry)
        client.get_collections.return_value.collections = collections
        return client

    def _clear(self) -> None:
        from nlqueries.cache import semantic_cache as sc

        sc._known_collections.clear()
        sc._missing_collections.clear()

    def test_ten_lookups_for_a_missing_collection_ask_qdrant_once(self) -> None:
        from nlqueries.cache import semantic_cache as sc

        self._clear()
        client = self._client(existing=["cache_other"])

        results = [sc._collection_exists(client, "cache_new_agent") for _ in range(10)]

        assert results == [False] * 10
        assert client.get_collections.call_count == 1, (
            f"asked Qdrant {client.get_collections.call_count} times for the same missing "
            "collection"
        )

    def test_the_negative_answer_expires(self, monkeypatch) -> None:
        """A collection created by another process — the CLI, a worker, a second
        replica — must become visible without a restart."""
        from nlqueries.cache import semantic_cache as sc

        self._clear()
        client = self._client(existing=[])
        assert sc._collection_exists(client, "cache_agent") is False

        # Now it exists, and the remembered "no" has aged out.
        client = self._client(existing=["cache_agent"])
        monkeypatch.setattr(sc.time, "time", lambda: 10_000_000_000.0)

        assert sc._collection_exists(client, "cache_agent") is True

    def test_creating_the_collection_clears_the_negative_entry(self) -> None:
        """Otherwise the very process that just created it keeps reporting a
        miss until the TTL runs out."""
        from nlqueries.cache import semantic_cache as sc

        self._clear()
        client = self._client(existing=[])
        cache = sc.SemanticCache("agent-x", binding=TEST_BINDING)
        sc._collection_exists(client, cache._collection)
        assert cache._collection in sc._missing_collections

        with (
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
            patch("nlqueries.cache.semantic_cache._get_client", return_value=MagicMock()),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.0] * 384),
        ):
            result = SimpleNamespace(
                resolved_question="a question",
                agent_type="sql",
                answer="an answer",
                sql="SELECT 1",
            )
            cache.put("a question", result)

        assert cache._collection not in sc._missing_collections
        assert cache._collection in sc._known_collections

    def test_an_unreachable_qdrant_is_not_remembered_as_missing(self) -> None:
        """ "I could not ask" is not "it does not exist" — caching that would
        extend an outage past its end."""
        from nlqueries.cache import semantic_cache as sc

        self._clear()
        client = MagicMock()
        client.get_collections.side_effect = ConnectionError("qdrant down")

        assert sc._collection_exists(client, "cache_agent") is False
        assert sc._missing_collections == {}

        # Qdrant recovers; the very next lookup sees the truth.
        recovered = self._client(existing=["cache_agent"])
        assert sc._collection_exists(recovered, "cache_agent") is True

    def test_a_known_collection_still_short_circuits(self) -> None:
        from nlqueries.cache import semantic_cache as sc

        self._clear()
        client = self._client(existing=["cache_agent"])
        assert sc._collection_exists(client, "cache_agent") is True
        assert sc._collection_exists(client, "cache_agent") is True
        assert client.get_collections.call_count == 1
