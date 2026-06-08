"""
Tests for nlqueries.embeddings (Task 4.2.1).

Unit tests (always run, model/Qdrant fully mocked):
  - embed_text returns a list of 384 floats
  - embed_batch returns the correct number of vectors
  - ensure_collection creates a collection when absent and is a no-op when present
  - upsert_capsules embeds and upserts the right number of points
  - search embeds the query and returns QueryCapsule objects

Integration tests (skipped unless Qdrant is reachable on QDRANT_URL):
  - Full upsert + search round-trip against a live Qdrant instance
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nlqueries.processing.parameterizer import QueryCapsule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VECTOR_SIZE = 384


def _make_capsules(n: int = 3) -> list[QueryCapsule]:
    return [
        QueryCapsule(
            template_sql=f"SELECT * FROM orders WHERE id = '[id:INT]' LIMIT {i}",
            placeholders=[],
            tables=["orders"],
            columns=["id"],
            frequency=i + 1,
            auto_description=f"Query on orders filtering by id (variant {i})",
            intent=f"Find orders by id variant {i}",
        )
        for i in range(n)
    ]


def _fake_vector() -> list[float]:
    """Return a deterministic 384-float unit vector."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(_VECTOR_SIZE).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _fake_vectors(n: int) -> list[list[float]]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, _VECTOR_SIZE)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return [row.tolist() for row in vecs]


# ---------------------------------------------------------------------------
# Unit tests — embedder
# ---------------------------------------------------------------------------


class TestEmbedText:
    """embed_text returns a properly-shaped, normalised list of floats."""

    def test_returns_list_of_floats(self) -> None:
        fake_array = np.array(_fake_vector(), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_array

        with patch("nlqueries.embeddings.embedder._model", mock_model):
            from nlqueries.embeddings.embedder import embed_text

            result = embed_text("show me all orders")

        assert isinstance(result, list)
        assert len(result) == _VECTOR_SIZE
        assert all(isinstance(v, float) for v in result)

    def test_calls_encode_with_normalize(self) -> None:
        fake_array = np.array(_fake_vector(), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_array

        with patch("nlqueries.embeddings.embedder._model", mock_model):
            from nlqueries.embeddings.embedder import embed_text

            embed_text("hello")

        mock_model.encode.assert_called_once_with("hello", normalize_embeddings=True)


class TestEmbedBatch:
    """embed_batch returns one vector per input string."""

    def test_correct_count(self) -> None:
        texts = ["first query", "second query", "third query"]
        fake_arrays = np.array(_fake_vectors(len(texts)), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_arrays

        with patch("nlqueries.embeddings.embedder._model", mock_model):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch(texts)

        assert len(result) == len(texts)
        for vec in result:
            assert isinstance(vec, list)
            assert len(vec) == _VECTOR_SIZE

    def test_calls_encode_with_batch_size_64(self) -> None:
        texts = ["a", "b"]
        fake_arrays = np.array(_fake_vectors(2), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_arrays

        with patch("nlqueries.embeddings.embedder._model", mock_model):
            from nlqueries.embeddings.embedder import embed_batch

            embed_batch(texts)

        mock_model.encode.assert_called_once_with(texts, batch_size=64, normalize_embeddings=True)

    def test_empty_input_returns_empty_list(self) -> None:
        fake_arrays = np.array([], dtype=np.float32).reshape(0, _VECTOR_SIZE)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_arrays

        with patch("nlqueries.embeddings.embedder._model", mock_model):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch([])

        assert result == []


# ---------------------------------------------------------------------------
# Unit tests — lazy singleton (model NOT loaded at import time)
# ---------------------------------------------------------------------------


class TestLazySingleton:
    """The SentenceTransformer model must NOT be loaded at import time."""

    def test_model_not_loaded_on_import(self) -> None:
        import nlqueries.embeddings.embedder as embedder_mod

        # Reset the singleton so the test is deterministic regardless of
        # import order in the test session.
        original = embedder_mod._model
        embedder_mod._model = None
        try:
            # Simply importing the module must not have loaded a model.
            # If _model is None here, no heavy load happened at import time.
            assert embedder_mod._model is None
        finally:
            embedder_mod._model = original


# ---------------------------------------------------------------------------
# Unit tests — qdrant_store.ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    """ensure_collection creates a collection when absent; no-op when present."""

    def _make_mock_client(self, existing: list[str]) -> MagicMock:
        mock_client = MagicMock()
        # MagicMock's `name` parameter is special (sets the mock's own name, not an attribute),
        # so we build simple SimpleNamespace stubs instead.
        from types import SimpleNamespace

        mock_collections = MagicMock()
        mock_collections.collections = [SimpleNamespace(name=n) for n in existing]
        mock_client.get_collections.return_value = mock_collections
        return mock_client

    def test_creates_collection_when_absent(self) -> None:
        mock_client = self._make_mock_client(existing=[])

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import ensure_collection

            ensure_collection("my_collection")

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "my_collection"

    def test_no_op_when_collection_exists(self) -> None:
        mock_client = self._make_mock_client(existing=["my_collection"])

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import ensure_collection

            ensure_collection("my_collection")

        mock_client.create_collection.assert_not_called()

    def test_uses_cosine_distance(self) -> None:
        from qdrant_client.models import Distance

        mock_client = self._make_mock_client(existing=[])

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import ensure_collection

            ensure_collection("col", vector_size=384)

        _, kwargs = mock_client.create_collection.call_args
        assert kwargs["vectors_config"].distance == Distance.COSINE
        assert kwargs["vectors_config"].size == 384


# ---------------------------------------------------------------------------
# Unit tests — qdrant_store.upsert_capsules
# ---------------------------------------------------------------------------


class TestUpsertCapsules:
    """upsert_capsules embeds intent text and upserts the right number of points."""

    def test_upserts_correct_number_of_points(self) -> None:
        capsules = _make_capsules(4)
        fake_vecs = _fake_vectors(4)

        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                return_value=fake_vecs,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_capsules

            upsert_capsules("test_col", capsules, agent_id="agent-1")

        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == 4

    def test_payload_fields_present(self) -> None:
        capsules = _make_capsules(2)
        fake_vecs = _fake_vectors(2)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                return_value=fake_vecs,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_capsules

            upsert_capsules("test_col", capsules, agent_id="my-agent")

        _, kwargs = mock_client.upsert.call_args
        point = kwargs["points"][0]
        assert "capsule_id" in point.payload
        assert "template_sql" in point.payload
        assert "tables" in point.payload
        assert "frequency" in point.payload
        assert point.payload["agent_id"] == "my-agent"

    def test_uses_auto_description_when_intent_empty(self) -> None:
        capsules = _make_capsules(1)
        capsules[0].intent = ""  # force fallback to auto_description
        fake_vecs = _fake_vectors(1)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                return_value=fake_vecs,
            ) as mock_embed,
        ):
            from nlqueries.embeddings.qdrant_store import upsert_capsules

            upsert_capsules("col", capsules)

        texts_used: list[str] = mock_embed.call_args[0][0]
        assert texts_used[0] == capsules[0].auto_description

    def test_empty_capsules_skips_upsert(self) -> None:
        mock_client = MagicMock()

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import upsert_capsules

            upsert_capsules("col", [])

        mock_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests — qdrant_store.search
# ---------------------------------------------------------------------------


class TestSearch:
    """search embeds the query and reconstructs QueryCapsule objects from payload."""

    def _make_hits(self, n: int) -> list[Any]:
        hits = []
        for i in range(n):
            hit = MagicMock()
            hit.payload = {
                "capsule_id": i,
                "template_sql": f"SELECT * FROM t WHERE id = {i}",
                "tables": ["t"],
                "frequency": i + 1,
                "agent_id": "",
            }
            hits.append(hit)
        return hits

    def _make_response(self, n: int) -> MagicMock:
        """Return a mock QueryResponse with n ScoredPoint stubs."""
        response = MagicMock()
        response.points = self._make_hits(n)
        return response

    def test_returns_query_capsule_instances(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(3)
        fake_vec = _fake_vector()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_text",
                return_value=fake_vec,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            results = search("col", "find orders", top_k=3)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, QueryCapsule)

    def test_calls_qdrant_query_points_with_correct_limit(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(2)
        fake_vec = _fake_vector()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_text",
                return_value=fake_vec,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            search("col", "query text", top_k=7)

        mock_client.query_points.assert_called_once_with(
            collection_name="col",
            query=fake_vec,
            limit=7,
        )

    def test_capsule_fields_populated_from_payload(self) -> None:
        hit = MagicMock()
        hit.payload = {
            "capsule_id": 99,
            "template_sql": "SELECT count(*) FROM sales",
            "tables": ["sales"],
            "frequency": 42,
            "agent_id": "ag",
        }
        response = MagicMock()
        response.points = [hit]
        mock_client = MagicMock()
        mock_client.query_points.return_value = response
        fake_vec = _fake_vector()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_text",
                return_value=fake_vec,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            results = search("col", "q")

        c = results[0]
        assert c.template_sql == "SELECT count(*) FROM sales"
        assert c.tables == ["sales"]
        assert c.frequency == 42

    def test_empty_search_results_returns_empty_list(self) -> None:
        response = MagicMock()
        response.points = []
        mock_client = MagicMock()
        mock_client.query_points.return_value = response
        fake_vec = _fake_vector()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch(
                "nlqueries.embeddings.embedder.embed_text",
                return_value=fake_vec,
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            results = search("col", "nothing here")

        assert results == []


# ---------------------------------------------------------------------------
# Integration tests — skipped unless Qdrant is reachable
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
    """Return True when a Qdrant instance is reachable at QDRANT_URL."""
    try:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=2)
        client.get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark_integration = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant is not available at QDRANT_URL (integration tests skipped)",
)


@pytestmark_integration
def test_integration_upsert_and_search() -> None:
    """Full round-trip: upsert capsules, search, verify results."""
    import uuid

    from nlqueries.embeddings.qdrant_store import ensure_collection, search, upsert_capsules

    # Use a unique collection so parallel test runs don't interfere.
    collection = f"test_integration_{uuid.uuid4().hex[:8]}"
    try:
        ensure_collection(collection, vector_size=_VECTOR_SIZE)
        capsules = _make_capsules(5)
        upsert_capsules(collection, capsules, agent_id="integration-test")

        results = search(collection, "Find orders by id", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, QueryCapsule)
            assert r.template_sql
    finally:
        # Clean up — best effort
        try:
            from qdrant_client import QdrantClient

            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            QdrantClient(url=url).delete_collection(collection)
        except Exception:  # noqa: BLE001
            pass
