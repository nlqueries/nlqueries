"""Tests for nlqueries.embeddings.qdrant_store — schema, capsule, and document chunk functions."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec

_VECTOR_SIZE = 384


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


def _make_col(
    name: str,
    col_type: str = "VARCHAR",
    description: str | None = None,
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type=col_type,
        nullable=True,
        is_primary_key=False,
        is_foreign_key=False,
        references=None,
        description=description,
    )


def _make_table(
    name: str,
    columns: list[ColumnSpec] | None = None,
    description: str | None = None,
) -> TableSpec:
    return TableSpec(
        name=name,
        schema="public",
        row_count=None,
        columns=columns if columns is not None else [_make_col("id", "INT")],
        description=description,
    )


def _make_schema(tables: list[TableSpec]) -> SchemaSpec:
    return SchemaSpec(database="testdb", tables=tables, extracted_at="2026-06-08T00:00:00+00:00")


def _fake_vectors(n: int) -> list[list[float]]:
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((n, _VECTOR_SIZE)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return [row.tolist() for row in vecs]


# ---------------------------------------------------------------------------
# Unit tests — upsert_schema
# ---------------------------------------------------------------------------


class TestUpsertSchema:
    def test_empty_schema_skips_upsert(self) -> None:
        mock_client = MagicMock()
        schema = _make_schema([])

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema, agent_id="a1")

        mock_client.upsert.assert_not_called()

    def test_upserts_one_point_per_table_no_col_descriptions(self) -> None:
        tables = [
            _make_table("orders", columns=[_make_col("id"), _make_col("status")]),
            _make_table("customers", columns=[_make_col("id"), _make_col("name")]),
            _make_table("products"),
        ]
        schema = _make_schema(tables)
        n_expected = 3  # one point per table, no column descriptions
        fake_vecs = _fake_vectors(n_expected)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema, agent_id="a1")

        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == n_expected

    def test_upserts_extra_points_for_columns_with_descriptions(self) -> None:
        tables = [
            _make_table(
                "orders",
                columns=[
                    _make_col("id", description="Primary key"),
                    _make_col("status", description="Order lifecycle state"),
                    _make_col("total"),  # no description — not embedded
                ],
            ),
            _make_table("customers", columns=[_make_col("id")]),
        ]
        schema = _make_schema(tables)
        # 2 tables + 2 columns with descriptions = 4 points
        n_expected = 4
        fake_vecs = _fake_vectors(n_expected)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema)

        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == n_expected

    def test_table_payload_fields(self) -> None:
        schema = _make_schema([_make_table("orders", description="Order records")])
        fake_vecs = _fake_vectors(1)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema, agent_id="myagent")

        _, kwargs = mock_client.upsert.call_args
        payload = kwargs["points"][0].payload
        assert payload["type"] == "table"
        assert payload["table_name"] == "orders"
        assert payload["agent_id"] == "myagent"
        assert "column_name" not in payload

    def test_column_payload_fields(self) -> None:
        schema = _make_schema(
            [_make_table("orders", columns=[_make_col("status", description="State")])]
        )
        fake_vecs = _fake_vectors(2)  # 1 table + 1 column
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema, agent_id="myagent")

        _, kwargs = mock_client.upsert.call_args
        col_point = kwargs["points"][1]
        payload = col_point.payload
        assert payload["type"] == "column"
        assert payload["table_name"] == "orders"
        assert payload["column_name"] == "status"
        assert payload["agent_id"] == "myagent"

    def test_table_text_includes_description_and_columns(self) -> None:
        schema = _make_schema(
            [
                _make_table(
                    "orders",
                    columns=[_make_col("id", "INT"), _make_col("status", "VARCHAR")],
                    description="Customer orders",
                )
            ]
        )
        fake_vecs = _fake_vectors(1)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs) as mock_emb,
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema)

        texts: list[str] = mock_emb.call_args[0][0]
        assert texts[0] == "orders: Customer orders. Columns: id INT, status VARCHAR"

    def test_table_text_without_description(self) -> None:
        schema = _make_schema(
            [_make_table("orders", columns=[_make_col("id", "INT")], description=None)]
        )
        fake_vecs = _fake_vectors(1)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs) as mock_emb,
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema)

        texts: list[str] = mock_emb.call_args[0][0]
        assert texts[0] == "orders. Columns: id INT"

    def test_column_text_format(self) -> None:
        schema = _make_schema(
            [
                _make_table(
                    "orders",
                    columns=[_make_col("status", "VARCHAR", description="Order state")],
                )
            ]
        )
        fake_vecs = _fake_vectors(2)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs) as mock_emb,
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema)

        texts: list[str] = mock_emb.call_args[0][0]
        assert texts[1] == "orders.status (VARCHAR): Order state"

    def test_columns_without_description_not_embedded(self) -> None:
        schema = _make_schema(
            [
                _make_table(
                    "orders",
                    columns=[
                        _make_col("id", description=None),
                        _make_col("status", description=""),
                        _make_col("total", description="Line total"),
                    ],
                )
            ]
        )
        # Only 1 table + 1 column with description → 2 points
        fake_vecs = _fake_vectors(2)
        mock_client = MagicMock()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=fake_vecs),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_schema

            upsert_schema("col", schema)

        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == 2


# ---------------------------------------------------------------------------
# Unit tests — search_schema
# ---------------------------------------------------------------------------


class TestSearchSchema:
    def _make_hit(self, type_: str, table: str, score: float = 0.9) -> MagicMock:
        hit = MagicMock()
        hit.payload = {"type": type_, "table_name": table, "agent_id": "a1"}
        hit.score = score
        return hit

    def _make_response(self, hits: list[MagicMock]) -> MagicMock:
        r = MagicMock()
        r.points = hits
        return r

    def test_returns_list_of_dicts(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(
            [self._make_hit("table", "orders")]
        )
        fake_vec = [0.1] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            results = search_schema("col", "customer orders")

        assert isinstance(results, list)
        assert isinstance(results[0], dict)

    def test_score_included_in_results(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(
            [self._make_hit("table", "orders", score=0.87)]
        )
        fake_vec = [0.1] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            results = search_schema("col", "q")

        assert "score" in results[0]
        assert results[0]["score"] == pytest.approx(0.87)

    def test_payload_fields_in_results(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(
            [self._make_hit("table", "customers", score=0.9)]
        )
        fake_vec = [0.1] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            results = search_schema("col", "q")

        assert results[0]["type"] == "table"
        assert results[0]["table_name"] == "customers"
        assert results[0]["agent_id"] == "a1"

    def test_calls_query_points_with_filter(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response([])
        fake_vec = [0.0] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            search_schema("col", "find tables", top_k=5)

        _, kwargs = mock_client.query_points.call_args
        assert kwargs["collection_name"] == "col"
        assert kwargs["limit"] == 5
        assert kwargs["query_filter"] is not None

    def test_empty_results_returns_empty_list(self) -> None:
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response([])
        fake_vec = [0.0] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            results = search_schema("col", "nothing")

        assert results == []

    def test_hit_with_none_payload_handled(self) -> None:
        hit = MagicMock()
        hit.payload = None
        hit.score = 0.5
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response([hit])
        fake_vec = [0.0] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_schema

            results = search_schema("col", "q")

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration test — skipped unless Qdrant is reachable
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
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
def test_integration_schema_upsert_and_search() -> None:
    """Upsert a 5-table schema and verify that searching 'customer orders'
    returns the orders table in the top 3 results."""
    import uuid

    from nlqueries.embeddings.qdrant_store import ensure_collection, search_schema, upsert_schema

    collection = f"test_schema_{uuid.uuid4().hex[:8]}"
    try:
        ensure_collection(collection, vector_size=_VECTOR_SIZE)

        tables = [
            _make_table("orders", description="Customer orders and purchase history"),
            _make_table("customers", description="Customer profiles and account details"),
            _make_table("products", description="Product catalog and inventory"),
            _make_table("invoices", description="Invoice documents for billing"),
            _make_table("payments", description="Payment transaction records"),
        ]
        schema = _make_schema(tables)
        upsert_schema(collection, schema, agent_id="integration-test")

        results = search_schema(collection, "customer orders", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3

        top_table_names = [r.get("table_name") for r in results]
        assert "orders" in top_table_names, (
            f"Expected 'orders' in top 3 results for 'customer orders', got: {top_table_names}"
        )
    finally:
        try:
            from qdrant_client import QdrantClient

            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            QdrantClient(url=url).delete_collection(collection)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Unit tests — document chunk functions (Task 9.2)
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "a1b2c3d4e5f6a7b8",
    source_id: str = "src-001",
    source_name: str = "report.pdf",
    page_number: int | None = 1,
    chunk_index: int = 0,
    text: str = "Sample chunk text.",
    metadata: dict | None = None,
):
    from nlqueries.document_connectors.base import DocumentChunk

    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_name=source_name,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        metadata=metadata if metadata is not None else {"connector": "pdf"},
    )


class TestUpsertChunks:
    def test_upsert_chunks_calls_upsert_with_correct_payload(self) -> None:
        """upsert_chunks stores all DocumentChunk fields in the Qdrant payload."""
        chunk = _make_chunk(
            chunk_id="aabbccddeeff0011",
            source_id="src-001",
            source_name="doc.pdf",
            page_number=3,
            chunk_index=2,
            text="Hello world content.",
            metadata={"connector": "pdf", "total_pages": 5},
        )
        fake_vec = [0.1] * _VECTOR_SIZE
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=[fake_vec]),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_chunks

            upsert_chunks("doc_src-001_chunks", [chunk])

        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "doc_src-001_chunks"
        assert len(kwargs["points"]) == 1
        payload = kwargs["points"][0].payload
        assert payload["chunk_id"] == "aabbccddeeff0011"
        assert payload["source_id"] == "src-001"
        assert payload["source_name"] == "doc.pdf"
        assert payload["page_number"] == 3
        assert payload["chunk_index"] == 2
        assert payload["text"] == "Hello world content."
        assert payload["metadata"]["connector"] == "pdf"

    def test_upsert_chunks_skips_empty_list(self) -> None:
        """upsert_chunks with an empty list must not call upsert."""
        mock_client = MagicMock()
        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import upsert_chunks

            upsert_chunks("doc_src-001_chunks", [])
        mock_client.upsert.assert_not_called()

    def test_upsert_chunks_calls_ensure_collection(self) -> None:
        """upsert_chunks must call ensure_collection before upserting."""
        chunk = _make_chunk()
        fake_vec = [0.0] * _VECTOR_SIZE
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=[fake_vec]),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_chunks

            upsert_chunks("doc_src-001_chunks", [chunk])

        mock_client.get_collections.assert_called()


class TestSearchChunks:
    def _make_hit(
        self,
        chunk_id: str = "aabbccddeeff0011",
        source_id: str = "src-001",
        page_number: int | None = 1,
        chunk_index: int = 0,
        text: str = "chunk text",
        score: float = 0.9,
    ) -> MagicMock:
        hit = MagicMock()
        hit.payload = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_name": "doc.pdf",
            "page_number": page_number,
            "chunk_index": chunk_index,
            "text": text,
            "metadata": {"connector": "pdf"},
        }
        hit.score = score
        return hit

    def _make_response(self, hits: list[MagicMock]) -> MagicMock:
        r = MagicMock()
        r.points = hits
        return r

    def test_search_chunks_returns_reconstructed_chunks(self) -> None:
        """search_chunks must return a list[DocumentChunk] reconstructed from the payload."""
        from nlqueries.document_connectors.base import DocumentChunk

        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response(
            [
                self._make_hit(chunk_id="aabb000000000001", text="first chunk"),
                self._make_hit(chunk_id="aabb000000000002", text="second chunk"),
            ]
        )
        fake_vec = [0.1] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_chunks

            results = search_chunks("doc_src-001_chunks", "some query")

        assert len(results) == 2
        assert all(isinstance(r, DocumentChunk) for r in results)
        assert results[0].text == "first chunk"
        assert results[1].text == "second chunk"

    def test_source_id_filter_applied_in_search(self) -> None:
        """search_chunks must pass a filter to Qdrant when source_id_filter is given."""
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response([])
        fake_vec = [0.0] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_chunks

            search_chunks("doc_src-001_chunks", "query", source_id_filter="src-001")

        _, kwargs = mock_client.query_points.call_args
        assert kwargs.get("query_filter") is not None

    def test_no_source_id_filter_passes_none_to_qdrant(self) -> None:
        """search_chunks must not add a filter when source_id_filter is None."""
        mock_client = MagicMock()
        mock_client.query_points.return_value = self._make_response([])
        fake_vec = [0.0] * _VECTOR_SIZE

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=fake_vec),
        ):
            from nlqueries.embeddings.qdrant_store import search_chunks

            search_chunks("doc_src-001_chunks", "query")

        _, kwargs = mock_client.query_points.call_args
        assert kwargs.get("query_filter") is None


class TestDeleteChunks:
    def test_delete_chunks_filters_by_source_id(self) -> None:
        """delete_chunks must call client.delete with a filter on source_id."""
        mock_client = MagicMock()

        with patch("nlqueries.embeddings.qdrant_store._client", mock_client):
            from nlqueries.embeddings.qdrant_store import delete_chunks

            delete_chunks("doc_src-001_chunks", "src-001")

        mock_client.delete.assert_called_once()
        _, kwargs = mock_client.delete.call_args
        assert kwargs["collection_name"] == "doc_src-001_chunks"
        selector = kwargs["points_selector"]
        # FilterSelector wraps a Filter — verify the filter has a condition on source_id
        filter_ = selector.filter
        assert filter_ is not None
        must_conditions = filter_.must
        assert len(must_conditions) == 1
        condition = must_conditions[0]
        assert condition.key == "source_id"
        assert condition.match.value == "src-001"
