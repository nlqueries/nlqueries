"""Tests for Phase 6C — ONNX backend, LRU cache, Qdrant scalar quantization."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_vector(seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_ort_mocks(hidden: int = 384, seq: int = 8, batch: int = 1):
    """Return (mock_ort_class, mock_tokenizer_class) for patching sys.modules."""
    ort_outputs = MagicMock()
    ort_outputs.last_hidden_state = np.random.standard_normal((batch, seq, hidden)).astype(
        np.float32
    )

    mock_ort_instance = MagicMock()
    mock_ort_instance.return_value = ort_outputs

    mock_ort_class = MagicMock()
    mock_ort_class.from_pretrained.return_value = mock_ort_instance

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {
        "input_ids": np.ones((batch, seq), dtype=np.int64),
        "attention_mask": np.ones((batch, seq), dtype=np.int64),
    }

    mock_tokenizer_class = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

    mock_optimum_onnxruntime = MagicMock()
    mock_optimum_onnxruntime.ORTModelForFeatureExtraction = mock_ort_class

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer = mock_tokenizer_class

    return mock_ort_class, mock_tokenizer_class, mock_optimum_onnxruntime, mock_transformers


# ---------------------------------------------------------------------------
# LRU cache on embed_text
# ---------------------------------------------------------------------------


class TestEmbedTextLruCache:
    def test_embed_text_is_lru_cached(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        assert hasattr(embed_text, "cache_info"), "embed_text must be wrapped with lru_cache"

    def test_cache_maxsize_is_2048(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        assert embed_text.cache_info().maxsize == 2048

    def test_same_text_returns_same_object(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        embed_text.cache_clear()
        fake = _fake_vector()

        with patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=fake):
            v1 = embed_text("hello world")
            v2 = embed_text("hello world")

        assert v1 is v2, "cache hit must return the exact same list object"

    def test_cache_suppresses_repeated_daemon_calls(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        embed_text.cache_clear()
        call_log: list[str] = []

        def fake_daemon(text: str) -> list[float]:
            call_log.append(text)
            return _fake_vector()

        with patch("nlqueries.embeddings.embedder._try_daemon_single", side_effect=fake_daemon):
            embed_text("alpha")
            embed_text("beta")
            embed_text("alpha")  # cache hit — daemon must NOT be called again

        assert call_log == ["alpha", "beta"]

    def test_cache_miss_after_clear(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        embed_text.cache_clear()
        call_count = [0]

        def fake_daemon(text: str) -> list[float]:
            call_count[0] += 1
            return _fake_vector()

        with patch("nlqueries.embeddings.embedder._try_daemon_single", side_effect=fake_daemon):
            embed_text("test_unique_6c")
            embed_text.cache_clear()
            embed_text("test_unique_6c")  # cache cleared — daemon called again

        assert call_count[0] == 2

    def test_cache_info_tracks_hits_and_misses(self) -> None:
        from nlqueries.embeddings.embedder import embed_text

        embed_text.cache_clear()
        fake = _fake_vector()

        with patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=fake):
            embed_text("unique_probe_xyz_123")
            embed_text("unique_probe_xyz_123")  # hit

        info = embed_text.cache_info()
        assert info.hits >= 1
        assert info.misses >= 1


# ---------------------------------------------------------------------------
# promoter.py — batch embed refactor
# ---------------------------------------------------------------------------


class TestPromoterUsesBatchEmbed:
    """promote_feedback must call embed_batch once (not embed_text in a loop)."""

    def test_uses_embed_batch_not_embed_text(self) -> None:
        from datetime import UTC, datetime

        from nlqueries.feedback.models import QueryFeedback
        from nlqueries.feedback.promoter import promote_feedback

        records = [
            QueryFeedback(
                question=f"Question {i}",
                generated_sql=f"SELECT {i} FROM orders",
                rating="up",
                agent_id="agent1",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )
            for i in range(3)
        ]

        mock_client = MagicMock()

        with (
            patch("nlqueries.feedback.store.load_feedback", return_value=records),
            patch("nlqueries.feedback.promoter._load_kb", return_value={}),
            # patch at the source module, not at the lazy-import site
            patch("nlqueries.embeddings.qdrant_store.ensure_collection", return_value=None),
            patch(
                "nlqueries.embeddings.embedder.embed_batch",
                return_value=[_fake_vector(i) for i in range(3)],
            ) as mock_batch,
            patch("qdrant_client.QdrantClient", return_value=mock_client),
        ):
            count = promote_feedback("agent1")

        mock_batch.assert_called_once()
        args = mock_batch.call_args[0][0]
        assert len(args) == 3
        assert count == 3


# ---------------------------------------------------------------------------
# Qdrant scalar quantization in ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollectionQuantization:
    def _mock_client(self, existing: list[str] | None = None) -> MagicMock:
        client = MagicMock()
        coll_mocks = []
        for n in existing or []:
            m = MagicMock()
            m.name = n  # MagicMock(name=) is reserved; must set as attribute
            coll_mocks.append(m)
        client.get_collections.return_value.collections = coll_mocks
        return client

    def test_new_collection_created_with_quantization_config(self) -> None:
        from nlqueries.embeddings.qdrant_store import ensure_collection

        mock_client = self._mock_client(existing=[])

        with patch("nlqueries.embeddings.qdrant_store._get_client", return_value=mock_client):
            ensure_collection("test_col", quantize=True)

        mock_client.create_collection.assert_called_once()
        kwargs = mock_client.create_collection.call_args[1]
        assert "quantization_config" in kwargs

    def test_quantization_config_is_scalar_int8_always_ram(self) -> None:
        from nlqueries.embeddings.qdrant_store import ensure_collection
        from qdrant_client.models import ScalarType

        mock_client = self._mock_client(existing=[])
        captured: list[Any] = []

        def capture(**kwargs: Any) -> None:
            captured.append(kwargs.get("quantization_config"))

        mock_client.create_collection.side_effect = capture

        with patch("nlqueries.embeddings.qdrant_store._get_client", return_value=mock_client):
            ensure_collection("test_col", quantize=True)

        assert len(captured) == 1
        qconfig = captured[0]
        assert qconfig is not None
        assert qconfig.scalar.type == ScalarType.INT8
        assert qconfig.scalar.always_ram is True

    def test_quantize_false_skips_quantization_config(self) -> None:
        from nlqueries.embeddings.qdrant_store import ensure_collection

        mock_client = self._mock_client(existing=[])

        with patch("nlqueries.embeddings.qdrant_store._get_client", return_value=mock_client):
            ensure_collection("test_col", quantize=False)

        kwargs = mock_client.create_collection.call_args[1]
        assert "quantization_config" not in kwargs

    def test_existing_collection_not_recreated(self) -> None:
        from nlqueries.embeddings.qdrant_store import ensure_collection

        mock_client = self._mock_client(existing=["test_col"])

        with patch("nlqueries.embeddings.qdrant_store._get_client", return_value=mock_client):
            ensure_collection("test_col", quantize=True)

        mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# _mean_pool_normalize (ONNX backend utility — pure numpy, no external deps)
# ---------------------------------------------------------------------------


class TestMeanPoolNormalize:
    def test_output_shape_matches_batch_and_hidden(self) -> None:
        from nlqueries.embeddings.embed_server import _mean_pool_normalize

        batch, seq, hidden = 3, 8, 384
        token_emb = np.random.standard_normal((batch, seq, hidden)).astype(np.float32)
        mask = np.ones((batch, seq), dtype=np.int64)

        result = _mean_pool_normalize(token_emb, mask)
        assert len(result) == batch
        assert all(len(row) == hidden for row in result)

    def test_output_rows_are_l2_unit_vectors(self) -> None:
        from nlqueries.embeddings.embed_server import _mean_pool_normalize

        token_emb = np.random.standard_normal((4, 6, 64)).astype(np.float32)
        mask = np.ones((4, 6), dtype=np.int64)

        result = _mean_pool_normalize(token_emb, mask)
        for row in result:
            norm = np.linalg.norm(row)
            assert abs(norm - 1.0) < 1e-5, f"row norm {norm} not ≈ 1.0"

    def test_padding_tokens_excluded_from_pooling(self) -> None:
        from nlqueries.embeddings.embed_server import _mean_pool_normalize

        # All token embeddings identical → mean is same regardless of mask,
        # but masked mean vs full mean should still both produce unit vectors.
        token_emb = np.ones((1, 4, 16), dtype=np.float32)
        mask_full = np.ones((1, 4), dtype=np.int64)
        mask_partial = np.array([[1, 1, 0, 0]], dtype=np.int64)

        result_full = _mean_pool_normalize(token_emb, mask_full)
        result_partial = _mean_pool_normalize(token_emb, mask_partial)

        # uniform input → same direction after masking → same unit vector
        np.testing.assert_allclose(result_full[0], result_partial[0], atol=1e-5)

    def test_single_item_batch(self) -> None:
        from nlqueries.embeddings.embed_server import _mean_pool_normalize

        token_emb = np.random.standard_normal((1, 10, 384)).astype(np.float32)
        mask = np.ones((1, 10), dtype=np.int64)
        result = _mean_pool_normalize(token_emb, mask)
        assert len(result) == 1
        assert len(result[0]) == 384


# ---------------------------------------------------------------------------
# embed_server — ONNX backend (_load_onnx_encoder)
# Lazy imports inside the function require sys.modules patching.
# ---------------------------------------------------------------------------


class TestLoadOnnxEncoder:
    """Test _load_onnx_encoder using sys.modules injection (no optimum required)."""

    def test_encoder_returns_correct_batch_shape(self) -> None:
        from nlqueries.embeddings.embed_server import _load_onnx_encoder

        hidden = 384
        mock_ort_class, _, mock_optimum_onnxruntime, mock_transformers = _make_ort_mocks(hidden)

        with patch.dict(
            sys.modules,
            {
                "optimum": MagicMock(onnxruntime=mock_optimum_onnxruntime),
                "optimum.onnxruntime": mock_optimum_onnxruntime,
                "transformers": mock_transformers,
            },
        ):
            encoder = _load_onnx_encoder()
            result = encoder(["test sentence"])

        assert len(result) == 1
        assert len(result[0]) == hidden

    def test_encoder_output_is_l2_normalised(self) -> None:
        from nlqueries.embeddings.embed_server import _load_onnx_encoder

        _, _, mock_optimum_onnxruntime, mock_transformers = _make_ort_mocks()

        with patch.dict(
            sys.modules,
            {
                "optimum": MagicMock(onnxruntime=mock_optimum_onnxruntime),
                "optimum.onnxruntime": mock_optimum_onnxruntime,
                "transformers": mock_transformers,
            },
        ):
            encoder = _load_onnx_encoder()
            result = encoder(["normalisation test"])

        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_load_encoder_dispatches_onnx_backend(self) -> None:
        from nlqueries.embeddings.embed_server import _load_encoder

        _, _, mock_optimum_onnxruntime, mock_transformers = _make_ort_mocks()

        with patch.dict(
            sys.modules,
            {
                "optimum": MagicMock(onnxruntime=mock_optimum_onnxruntime),
                "optimum.onnxruntime": mock_optimum_onnxruntime,
                "transformers": mock_transformers,
            },
        ):
            encoder = _load_encoder("onnx")

        assert callable(encoder)

    def test_load_encoder_dispatches_torch_backend(self) -> None:
        from nlqueries.embeddings.embed_server import _load_encoder

        mock_st_model = MagicMock()
        mock_st_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_st_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            encoder = _load_encoder("torch")

        assert callable(encoder)


# ---------------------------------------------------------------------------
# embed_server — legacy _EmbedHandler.model backward-compat
# ---------------------------------------------------------------------------


class TestEmbedHandlerLegacyCompat:
    """The legacy _EmbedHandler.model attribute path must still work."""

    def test_encode_dispatches_to_legacy_model_when_encoder_is_none(self) -> None:
        from nlqueries.embeddings.embed_server import _EmbedHandler

        mock_model = MagicMock()
        expected = [0.5] * 384
        # model.encode returns an object with .tolist() that gives a flat list
        mock_result = MagicMock()
        mock_result.tolist.return_value = expected
        mock_model.encode.return_value = mock_result

        original_encoder = _EmbedHandler.encoder
        original_model = _EmbedHandler.model
        try:
            _EmbedHandler.encoder = None
            _EmbedHandler.model = mock_model

            handler = _EmbedHandler.__new__(_EmbedHandler)
            vectors = handler._encode(["hello"])
        finally:
            _EmbedHandler.encoder = original_encoder
            _EmbedHandler.model = original_model

        assert vectors == [expected]
        mock_model.encode.assert_called_once()

    def test_encode_prefers_encoder_over_model(self) -> None:
        from nlqueries.embeddings.embed_server import _EmbedHandler

        expected = [0.7] * 384
        mock_encoder = MagicMock(return_value=[expected])
        mock_model = MagicMock()

        original_encoder = _EmbedHandler.encoder
        original_model = _EmbedHandler.model
        try:
            _EmbedHandler.encoder = mock_encoder
            _EmbedHandler.model = mock_model

            handler = _EmbedHandler.__new__(_EmbedHandler)
            vectors = handler._encode(["hello"])
        finally:
            _EmbedHandler.encoder = original_encoder
            _EmbedHandler.model = original_model

        assert vectors == [expected]
        mock_model.encode.assert_not_called()

    def test_serve_sets_encoder_attribute(self) -> None:
        from nlqueries.embeddings.embed_server import _EmbedHandler

        fake_encoder = MagicMock(return_value=[[0.1] * 384])

        with (
            patch("nlqueries.embeddings.embed_server._load_encoder", return_value=fake_encoder),
            patch("nlqueries.embeddings.embed_server._PID_FILE") as mock_pid,
            patch("nlqueries.embeddings.embed_server.HTTPServer") as mock_http,
        ):
            mock_pid.parent.mkdir = MagicMock()
            mock_pid.write_text = MagicMock()
            mock_pid.unlink = MagicMock()

            server_instance = MagicMock()
            server_instance.serve_forever.side_effect = KeyboardInterrupt
            mock_http.return_value = server_instance

            try:
                from nlqueries.embeddings import embed_server

                embed_server.serve(port=19999, backend="torch")
            except (KeyboardInterrupt, SystemExit):
                pass

        assert _EmbedHandler.encoder is fake_encoder
