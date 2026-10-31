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

        with (
            patch("nlqueries.embeddings.embedder._model", mock_model),
            # Suppress the daemon probe so the local-model path is always exercised,
            # even when `nlqueries embed-server` is running in the background.
            patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=None),
        ):
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

        with (
            patch("nlqueries.embeddings.embedder._model", mock_model),
            # Suppress the daemon probe so the local-model path is always exercised,
            # even when `nlqueries embed-server` is running in the background.
            patch("nlqueries.embeddings.embedder._try_daemon_batch", return_value=None),
        ):
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

    def test_a_600_text_batch_is_sent_to_the_daemon_as_two_requests(self) -> None:
        """The daemon refuses more than 512 texts in one request."""
        texts = [f"text {i}" for i in range(600)]
        sent: list[list[str]] = []

        def daemon_post(path: str, payload: dict[str, list[str]], key: str) -> list[list[float]]:
            assert (path, key) == ("/embed-batch", "vectors")
            sent.append(payload["texts"])
            return [[float(len(sent))] * _VECTOR_SIZE for _ in payload["texts"]]

        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            patch("nlqueries.embeddings.embedder._daemon_post", side_effect=daemon_post),
            patch("nlqueries.embeddings.embedder._get_model") as get_model,
        ):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch(texts)

        assert [len(part) for part in sent] == [512, 88]
        assert sent[0] + sent[1] == texts
        assert len(result) == 600
        assert result[511][0] == 1.0 and result[512][0] == 2.0
        get_model.assert_not_called()

    def test_a_batch_over_the_character_limit_is_split_into_parts_that_fit(self) -> None:
        """Fewer than 512 texts, more than MAX_BATCH_CHARS in all: the wide-schema
        case, which the daemon would refuse whole."""
        from nlqueries.embeddings.embed_server import MAX_BATCH_CHARS, MAX_BATCH_TEXTS

        texts = [chr(ord("a") + i) * 600_000 for i in range(5)]
        assert len(texts) < MAX_BATCH_TEXTS and sum(map(len, texts)) > MAX_BATCH_CHARS
        sent: list[list[str]] = []

        def daemon_post(path: str, payload: dict[str, list[str]], key: str) -> list[list[float]]:
            sent.append(payload["texts"])
            return [[0.0] * _VECTOR_SIZE for _ in payload["texts"]]

        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            patch("nlqueries.embeddings.embedder._daemon_post", side_effect=daemon_post),
            patch("nlqueries.embeddings.embedder._get_model") as get_model,
        ):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch(texts)

        assert len(sent) > 1
        assert all(sum(map(len, part)) <= MAX_BATCH_CHARS for part in sent)
        assert all(len(part) <= MAX_BATCH_TEXTS for part in sent)
        assert [t for part in sent for t in part] == texts
        assert len(result) == len(texts)
        get_model.assert_not_called()

    def test_a_single_text_over_the_character_limit_goes_alone(self) -> None:
        from nlqueries.embeddings.embed_server import MAX_BATCH_CHARS
        from nlqueries.embeddings.embedder import _daemon_parts

        oversized = "x" * (MAX_BATCH_CHARS + 1)

        assert list(_daemon_parts(["a", "b", oversized, "c"])) == [["a", "b"], [oversized], ["c"]]
        assert list(_daemon_parts([oversized])) == [[oversized]]

    def test_a_single_text_over_the_body_limit_goes_alone(self) -> None:
        """Under MAX_BATCH_CHARS, but its six-byte control characters make it a
        body over MAX_BODY_BYTES on its own."""
        from nlqueries.embeddings.embed_server import MAX_BATCH_CHARS, MAX_BODY_BYTES
        from nlqueries.embeddings.embedder import _daemon_parts

        oversized = "\x01" * 1_800_000
        assert len(oversized) < MAX_BATCH_CHARS and len(oversized) * 6 > MAX_BODY_BYTES

        assert list(_daemon_parts(["a", oversized, "b"])) == [["a"], [oversized], ["b"]]

    @pytest.mark.parametrize("off_by", [-1, 1])
    def test_a_part_answered_with_the_wrong_vector_count_falls_back(self, off_by: int) -> None:
        """Short (or long) by one on the second part: returning the list would put
        every later vector against the wrong text."""
        texts = [f"text {i}" for i in range(600)]
        replies = [
            [[0.0] * _VECTOR_SIZE] * 512,
            [[0.0] * _VECTOR_SIZE] * (88 + off_by),
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(_fake_vectors(600), dtype=np.float32)

        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            patch("nlqueries.embeddings.embedder._daemon_post", side_effect=replies),
            patch("nlqueries.embeddings.embedder._model", mock_model),
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        ):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch(texts)

        mock_model.encode.assert_called_once_with(texts, batch_size=64, normalize_embeddings=True)
        assert len(result) == 600

    @pytest.mark.parametrize(
        ("reply", "said"),
        [
            ([[0.0] * _VECTOR_SIZE] * 87, "87 vectors for 88 texts"),
            ({str(i): [0.0] for i in range(88)}, "dict vectors"),
        ],
    )
    def test_a_wrong_vector_count_raises_when_the_daemon_is_required(
        self, reply: object, said: str
    ) -> None:
        from nlqueries.embeddings.embedder import (
            FAILING,
            EmbeddingServiceUnavailable,
            embed_batch,
        )

        replies = [[[0.0] * _VECTOR_SIZE] * 512, reply]
        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            patch("nlqueries.embeddings.embedder._daemon_post", side_effect=replies),
            patch("nlqueries.embeddings.embedder._get_model") as get_model,
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", True),
            pytest.raises(EmbeddingServiceUnavailable) as raised,
        ):
            embed_batch([f"text {i}" for i in range(600)])

        assert raised.value.reason == FAILING
        assert said in str(raised.value)
        get_model.assert_not_called()

    def test_a_failed_part_falls_back_for_the_whole_batch(self) -> None:
        from nlqueries.embeddings.embedder import ABSENT, EmbeddingServiceUnavailable

        texts = [f"text {i}" for i in range(600)]
        replies = [
            [[0.0] * _VECTOR_SIZE] * 512,
            EmbeddingServiceUnavailable(ABSENT, "gone"),
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(_fake_vectors(600), dtype=np.float32)

        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            patch("nlqueries.embeddings.embedder._daemon_post", side_effect=replies),
            patch("nlqueries.embeddings.embedder._model", mock_model),
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        ):
            from nlqueries.embeddings.embedder import embed_batch

            result = embed_batch(texts)

        assert len(result) == 600
        mock_model.encode.assert_called_once_with(texts, batch_size=64, normalize_embeddings=True)


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
        from nlqueries.cache.envelope import sign_capsule  # noqa: PLC0415

        hits = []
        for i in range(n):
            hit = MagicMock()
            hit.payload = sign_capsule(
                {
                    "type": "capsule",
                    "capsule_id": i,
                    "template_sql": f"SELECT * FROM t WHERE id = {i}",
                    "intent": "",
                    "auto_description": "",
                    "tables": ["t"],
                    "frequency": i + 1,
                    "agent_id": "",
                }
            )
            hit.score = 1.0
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
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda texts: [fake_vec for _ in texts],
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
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda texts: [fake_vec for _ in texts],
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            search("col", "query text", top_k=7)

        # Capsule points only, and more candidates than wanted, so unsigned
        # ones nearer the question do not take the signed ones' places.
        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "col"
        assert kwargs["query"] == fake_vec
        assert kwargs["limit"] == 21
        (condition,) = kwargs["query_filter"].must
        assert condition.key == "type" and condition.match.value == "capsule"

    def test_capsule_fields_populated_from_payload(self) -> None:
        from nlqueries.cache.envelope import sign_capsule  # noqa: PLC0415

        hit = MagicMock()
        hit.payload = sign_capsule(
            {
                "type": "capsule",
                "capsule_id": 99,
                "template_sql": "SELECT count(*) FROM sales",
                "intent": "total sales",
                "auto_description": "count of sales rows",
                "tables": ["sales"],
                "frequency": 42,
                "agent_id": "ag",
            }
        )
        hit.score = 1.0
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
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda texts: [fake_vec for _ in texts],
            ),
        ):
            from nlqueries.embeddings.qdrant_store import search

            results = search("col", "q")

        c = results[0]
        assert c.template_sql == "SELECT count(*) FROM sales"
        assert c.tables == ["sales"]
        assert c.frequency == 42
        # Stored since capsules were embedded where questions read them; the
        # prompt shows the intent, so a capsule without it was a bare template.
        assert c.intent == "total sales"
        assert c.auto_description == "count of sales rows"

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
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                side_effect=lambda texts: [fake_vec for _ in texts],
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
        except Exception:  # noqa: BLE001, S110 - cleanup after the test, best effort
            pass


# ---------------------------------------------------------------------------
# A text UTF-8 cannot encode
# ---------------------------------------------------------------------------


class TestLoneSurrogate:
    """A lone surrogate, as ``json.loads`` or a non-UTF-8 ``sys.argv`` yields,
    cannot be sent to the daemon as UTF-8. It falls back like any other daemon
    failure rather than raising past `embed_text` and `embed_batch`."""

    _TEXT = "orders by region \ud800"

    @staticmethod
    def _no_daemon_contact() -> Any:
        return patch(
            "nlqueries.embeddings.embedder.urllib.request.urlopen",
            side_effect=AssertionError("the request was sent"),
        )

    def test_embed_text_falls_back_to_the_local_model(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(_fake_vectors(1)[0], dtype=np.float32)
        embed_text.cache_clear()
        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            self._no_daemon_contact(),
            patch("nlqueries.embeddings.embedder._model", mock_model),
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        ):
            result = embed_text(self._TEXT)

        mock_model.encode.assert_called_once_with(self._TEXT, normalize_embeddings=True)
        assert len(result) == _VECTOR_SIZE

    def test_embed_batch_falls_back_to_the_local_model(self) -> None:
        from nlqueries.embeddings.embedder import embed_batch

        texts = ["fine", self._TEXT, "also fine"]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(_fake_vectors(3), dtype=np.float32)
        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            self._no_daemon_contact(),
            patch("nlqueries.embeddings.embedder._model", mock_model),
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        ):
            result = embed_batch(texts)

        mock_model.encode.assert_called_once_with(texts, batch_size=64, normalize_embeddings=True)
        assert len(result) == 3

    @pytest.mark.parametrize("call", ["embed_text", "embed_batch"])
    def test_it_raises_when_the_daemon_is_required(self, call: str) -> None:
        from nlqueries.embeddings import embedder
        from nlqueries.embeddings.embedder import FAILING, EmbeddingServiceUnavailable

        embedder.embed_text.cache_clear()
        with (
            patch("nlqueries.embeddings.embedder._require_daemon_model_agreement"),
            self._no_daemon_contact(),
            patch("nlqueries.embeddings.embedder._get_model") as get_model,
            patch("nlqueries.config.EMBED_SERVER_REQUIRED", True),
            pytest.raises(EmbeddingServiceUnavailable) as raised,
        ):
            if call == "embed_text":
                embedder.embed_text(self._TEXT)
            else:
                embedder.embed_batch(["fine", self._TEXT])

        assert raised.value.reason == FAILING
        assert "text is not valid Unicode" in str(raised.value)
        assert isinstance(raised.value.__cause__, UnicodeEncodeError)
        get_model.assert_not_called()


def test_the_client_does_not_import_the_daemon() -> None:
    """The limits it sizes requests to live in `config`, so embedding a text
    does not load the daemon module and the HTTP server stack with it."""
    import subprocess
    import sys

    probe = (
        "import sys, nlqueries.embeddings.embedder; "
        "print('nlqueries.embeddings.embed_server' in sys.modules, 'http.server' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False"]


def test_the_daemon_and_the_client_read_one_declaration_of_the_limits() -> None:
    from nlqueries import config
    from nlqueries.embeddings import embed_server

    assert embed_server.MAX_BATCH_TEXTS == config.EMBED_MAX_BATCH_TEXTS
    assert embed_server.MAX_BATCH_CHARS == config.EMBED_MAX_BATCH_CHARS
    assert embed_server.MAX_BODY_BYTES == config.EMBED_MAX_BODY_BYTES
    assert embed_server._EmbedHandler.MAX_BODY_BYTES == config.EMBED_MAX_BODY_BYTES


def test_the_fallback_warning_does_not_say_the_daemon_is_required(caplog) -> None:
    """The exception's message is written for the required case ("will not load a
    model in-process"); the fallback, which does, says what happened instead."""
    from nlqueries.embeddings.embedder import (
        ABSENT,
        EmbeddingServiceUnavailable,
        _fall_back_or_raise,
    )

    with (
        patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        caplog.at_level("WARNING", logger="nlqueries.embeddings.embedder"),
    ):
        _fall_back_or_raise(EmbeddingServiceUnavailable(ABSENT, "connection refused"))

    (line,) = [r.getMessage() for r in caplog.records if "in-process instead" in r.getMessage()]
    assert "connection refused" in line
    assert "EMBED_SERVER_REQUIRED is set" not in line


@pytest.mark.parametrize(
    ("surface", "reason", "level"),
    [
        # A CLI command with no daemon: how the CLI works without one.
        ("cli", "absent", "INFO"),
        # A daemon that is there and failing, or on another model: worth a warning anywhere.
        ("cli", "failing", "WARNING"),
        ("cli", "running a different model", "WARNING"),
        # A server, or a library caller, loading a model into itself.
        ("mcp", "absent", "WARNING"),
        ("library", "absent", "WARNING"),
    ],
)
def test_the_fallback_is_a_warning_except_for_a_cli_command_with_no_daemon(
    caplog, surface: str, reason: str, level: str
) -> None:
    from nlqueries import audit
    from nlqueries.embeddings.embedder import EmbeddingServiceUnavailable, _fall_back_or_raise

    if surface != "library":
        audit.bind_process(audit.Surface(surface))

    with (
        patch("nlqueries.config.EMBED_SERVER_REQUIRED", False),
        caplog.at_level("DEBUG", logger="nlqueries.embeddings.embedder"),
    ):
        _fall_back_or_raise(EmbeddingServiceUnavailable(reason, "detail"))

    (record,) = [r for r in caplog.records if "in-process instead" in r.getMessage()]
    assert record.levelname == level


def test_a_cli_command_binds_the_surface_the_fallback_level_reads() -> None:
    """The quiet case keys on the CLI's own binding, made by the root group
    before any command runs."""
    from click.testing import CliRunner
    from nlqueries import audit
    from nlqueries.cli.main import cli

    result = CliRunner().invoke(cli, ["kb-stats", "--help"])

    assert result.exit_code == 0, result.output
    assert audit.current().surface == audit.Surface.CLI
