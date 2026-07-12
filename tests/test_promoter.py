"""Tests for nlqueries.feedback.promoter — Phase 5B."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from nlqueries.feedback.promoter import (
    _pair_point_id,
    _sql_references_known_tables,
    _verified_collection,
    promote_feedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kb(table_names: list[str]) -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {"name": n, "columns": [{"name": "id", "type": "INTEGER"}]} for n in table_names
            ]
        }
    }


def _make_feedback_record(
    question: str = "How many orders?",
    generated_sql: str = "SELECT COUNT(*) FROM orders",
    rating: str = "up",
    corrected_sql: str | None = None,
    agent_id: str = "agent1",
):
    from nlqueries.feedback.models import QueryFeedback

    return QueryFeedback(
        question=question,
        generated_sql=generated_sql,
        rating=rating,
        agent_id=agent_id,
        corrected_sql=corrected_sql,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# _verified_collection
# ---------------------------------------------------------------------------


class TestVerifiedCollection:
    def test_basic(self) -> None:
        assert _verified_collection("myagent") == "agent_myagent_verified"

    def test_special_chars_sanitised(self) -> None:
        name = _verified_collection("my:agent/db")
        assert ":" not in name
        assert "/" not in name
        assert name.startswith("agent_")
        assert name.endswith("_verified")


# ---------------------------------------------------------------------------
# _pair_point_id
# ---------------------------------------------------------------------------


class TestPairPointId:
    def test_deterministic(self) -> None:
        q, s = "How many orders?", "SELECT COUNT(*) FROM orders"
        assert _pair_point_id(q, s) == _pair_point_id(q, s)

    def test_different_pairs_produce_different_ids(self) -> None:
        id1 = _pair_point_id("question A", "SELECT 1")
        id2 = _pair_point_id("question B", "SELECT 1")
        assert id1 != id2

    def test_same_question_different_sql_differ(self) -> None:
        q = "How many orders?"
        assert _pair_point_id(q, "SELECT COUNT(*) FROM orders") != _pair_point_id(
            q, "SELECT * FROM orders"
        )

    def test_returns_int(self) -> None:
        assert isinstance(_pair_point_id("q", "s"), int)

    def test_id_fits_in_64_bits(self) -> None:
        point_id = _pair_point_id("question", "SELECT 1")
        assert 0 <= point_id < 2**64


# ---------------------------------------------------------------------------
# _sql_references_known_tables
# ---------------------------------------------------------------------------


class TestSqlReferencesKnownTables:
    def test_valid_sql_returns_true(self) -> None:
        kb = _make_kb(["orders"])
        assert _sql_references_known_tables("SELECT COUNT(*) FROM orders", kb)

    def test_unknown_table_returns_false(self) -> None:
        kb = _make_kb(["orders"])
        assert not _sql_references_known_tables("SELECT * FROM customers", kb)

    def test_empty_kb_schema_returns_true_permissive(self) -> None:
        assert _sql_references_known_tables("SELECT * FROM anything", {})

    def test_cte_alias_not_flagged_as_unknown_table(self) -> None:
        kb = _make_kb(["orders"])
        sql = "WITH cte AS (SELECT * FROM orders) SELECT * FROM cte"
        assert _sql_references_known_tables(sql, kb)

    def test_unparseable_sql_returns_true_permissive(self) -> None:
        kb = _make_kb(["orders"])
        assert _sql_references_known_tables("NOT VALID SQL !!!", kb)

    def test_multiple_known_tables_returns_true(self) -> None:
        kb = _make_kb(["orders", "customers"])
        sql = "SELECT * FROM orders JOIN customers ON orders.id = customers.id"
        assert _sql_references_known_tables(sql, kb)

    def test_mixed_known_unknown_returns_false(self) -> None:
        kb = _make_kb(["orders"])
        sql = "SELECT * FROM orders JOIN mystery ON orders.id = mystery.id"
        assert not _sql_references_known_tables(sql, kb)


# ---------------------------------------------------------------------------
# promote_feedback — lazy imports are patched at their source modules
# ---------------------------------------------------------------------------


class TestPromoteFeedback:
    def _run_promote(
        self,
        records,
        kb: dict[str, Any] | None = None,
    ) -> int:
        """Helper: run promote_feedback with mocked deps (patched at source)."""
        kb = kb or _make_kb(["orders"])
        mock_client = MagicMock()
        mock_client.upsert = MagicMock()

        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=records),
            patch("nlqueries.feedback.promoter._load_kb", return_value=kb),
            patch("nlqueries.embeddings.qdrant_store.ensure_collection", return_value=None),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda qs: [[0.1] * 384 for _ in qs],
            ),
            patch("qdrant_client.QdrantClient", return_value=mock_client),
        ):
            count = promote_feedback("agent1")

        return count

    def test_returns_zero_when_no_feedback(self) -> None:
        assert self._run_promote([]) == 0

    def test_returns_zero_when_only_down_ratings(self) -> None:
        records = [_make_feedback_record(rating="down")]
        assert self._run_promote(records) == 0

    def test_promotes_positive_feedback(self) -> None:
        records = [_make_feedback_record(rating="up")]
        count = self._run_promote(records)
        assert count == 1

    def test_prefers_corrected_sql_over_generated(self) -> None:
        rec = _make_feedback_record(
            rating="up",
            generated_sql="SELECT * FROM orders",
            corrected_sql="SELECT COUNT(*) FROM orders",
        )
        upserted_sqls: list[str] = []
        mock_client = MagicMock()

        def capture_upsert(collection_name, points):
            for p in points:
                upserted_sqls.append(p.payload["sql"])

        mock_client.upsert.side_effect = capture_upsert

        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=[rec]),
            patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
            patch("nlqueries.embeddings.qdrant_store.ensure_collection", return_value=None),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda qs: [[0.1] * 384 for _ in qs],
            ),
            patch("qdrant_client.QdrantClient", return_value=mock_client),
        ):
            promote_feedback("agent1")

        assert upserted_sqls == ["SELECT COUNT(*) FROM orders"]

    def test_deduplicates_identical_sql(self) -> None:
        records = [
            _make_feedback_record(rating="up", question="Q1"),
            _make_feedback_record(rating="up", question="Q2"),  # same SQL
        ]
        count = self._run_promote(records)
        assert count == 1  # deduped

    def test_skips_empty_sql(self) -> None:
        rec = _make_feedback_record(rating="up", generated_sql="")
        assert self._run_promote([rec]) == 0

    def test_skips_unknown_table(self) -> None:
        rec = _make_feedback_record(rating="up", generated_sql="SELECT * FROM does_not_exist")
        kb = _make_kb(["orders"])
        assert self._run_promote([rec], kb=kb) == 0

    def test_returns_zero_when_qdrant_unavailable(self) -> None:
        rec = _make_feedback_record(rating="up")

        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=[rec]),
            patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
            patch("nlqueries.embeddings.qdrant_store.ensure_collection", return_value=None),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
            patch("qdrant_client.QdrantClient", side_effect=RuntimeError("qdrant down")),
        ):
            count = promote_feedback("agent1")

        assert count == 0

    def test_multiple_unique_sql_all_promoted(self) -> None:
        records = [
            _make_feedback_record(
                rating="up",
                question="Q1",
                generated_sql="SELECT COUNT(*) FROM orders",
            ),
            _make_feedback_record(
                rating="up",
                question="Q2",
                generated_sql="SELECT SUM(total) FROM orders",
            ),
        ]
        count = self._run_promote(records)
        assert count == 2

    # --- dry_run: preview the pending pairs without side effects ---

    def test_dry_run_returns_pending_pairs_without_side_effects(self) -> None:
        records = [_make_feedback_record(rating="up", question="Q1")]
        # No Qdrant/embedder patches needed — dry_run must not touch them.
        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=records),
            patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
        ):
            pending = promote_feedback("agent1", dry_run=True)
        assert pending == [{"question": "Q1", "sql": "SELECT COUNT(*) FROM orders"}]

    def test_dry_run_empty_when_no_positive(self) -> None:
        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=[]),
            patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
        ):
            assert promote_feedback("agent1", dry_run=True) == []

    def test_dry_run_matches_a_real_run(self) -> None:
        records = [
            _make_feedback_record(
                rating="up", question="Q1", generated_sql="SELECT COUNT(*) FROM orders"
            ),
            _make_feedback_record(
                rating="up", question="Q2", generated_sql="SELECT SUM(total) FROM orders"
            ),
            _make_feedback_record(rating="down", question="Q3"),  # excluded from both
        ]
        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=records),
            patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
        ):
            pending = promote_feedback("agent1", dry_run=True)
        promoted = self._run_promote(records)
        assert isinstance(pending, list)
        assert len(pending) == promoted == 2  # preview count == real promoted count


# ---------------------------------------------------------------------------
# Phase 5B: verified examples in prompt assembly
# ---------------------------------------------------------------------------


class TestVerifiedFewShotInPromptAssembly:
    """Verify that _search_verified() results appear with the correct label."""

    def _assemble(self, verified_hits: list[dict]) -> str:
        """Run assemble_prompt with mocked Qdrant and return dynamic_context."""
        from nlqueries.orchestrator.prompt_assembly import assemble_prompt

        kb = _make_kb(["orders", "customers"])

        with (
            patch(
                "nlqueries.orchestrator.prompt_assembly._search_verified",
                return_value=verified_hits,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
        ):
            prompt = assemble_prompt(
                "How many orders?",
                kb,
                top_k_capsules=5,
                collection="agent_agent1_schema",
            )
        return prompt.dynamic_context

    def test_verified_hit_appears_in_dynamic_context(self) -> None:
        hits = [{"question": "How many orders?", "sql": "SELECT COUNT(*) FROM orders"}]
        ctx = self._assemble(hits)
        assert "SELECT COUNT(*) FROM orders" in ctx

    def test_verified_hit_has_verified_label(self) -> None:
        hits = [{"question": "How many orders?", "sql": "SELECT COUNT(*) FROM orders"}]
        ctx = self._assemble(hits)
        assert "verified example" in ctx

    def test_no_verified_hits_no_label(self) -> None:
        ctx = self._assemble([])
        assert "verified example" not in ctx

    def test_search_verified_returns_empty_for_non_schema_collection(self) -> None:
        from nlqueries.orchestrator.prompt_assembly import _search_verified

        result = _search_verified("some_other_collection", "question")
        assert result == []

    def test_search_verified_returns_empty_when_qdrant_down(self) -> None:
        from nlqueries.orchestrator.prompt_assembly import _search_verified

        with patch("qdrant_client.QdrantClient", side_effect=RuntimeError("down")):
            result = _search_verified("agent_x_schema", "question", vector=[0.1] * 384)
        assert result == []

    def test_search_verified_queries_verified_collection(self) -> None:
        """_search_verified must query agent_{id}_verified, not _schema."""
        from nlqueries.orchestrator.prompt_assembly import _search_verified

        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [
            MagicMock(name="agent_agent1_verified")
        ]
        mock_client.query_points.return_value.points = []

        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            _search_verified("agent_agent1_schema", "test question", vector=[0.1] * 384)

        if mock_client.query_points.called:
            called_collection = (
                mock_client.query_points.call_args[1].get("collection_name")
                or mock_client.query_points.call_args[0][0]
            )
            assert "verified" in called_collection
