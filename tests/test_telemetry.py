"""Tests for nlqueries.telemetry — tracer/meter setup and instrumentation smoke-tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tracer() -> tuple[MagicMock, MagicMock]:
    """Return (mock_tracer, mock_span) pre-wired as a context manager."""
    mock_span = MagicMock()
    mock_tracer = MagicMock()
    # make start_as_current_span() return a context manager that yields mock_span
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    return mock_tracer, mock_span


def _write_kb(path: Path, agent_id: str, kb: dict[str, Any]) -> None:
    import re

    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    (path / f"{safe_id}.yaml").write_text(yaml.dump(kb), encoding="utf-8")


def _make_kb() -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "Orders table",
                    "row_count": 10,
                    "columns": [{"name": "id", "type": "INT", "description": ""}],
                }
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


async def _collect(gen: Any) -> list[str]:
    out: list[str] = []
    async for tok in gen:
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# telemetry module — get_tracer / get_meter
# ---------------------------------------------------------------------------


class TestGetTracerAndMeter:
    def test_get_tracer_returns_tracer(self) -> None:
        from nlqueries.telemetry import get_tracer

        tracer = get_tracer()
        assert tracer is not None

    def test_get_meter_returns_meter(self) -> None:
        from nlqueries.telemetry import get_meter

        meter = get_meter()
        assert meter is not None

    def test_get_tracer_is_callable(self) -> None:
        from nlqueries.telemetry import get_tracer

        tracer = get_tracer()
        assert callable(getattr(tracer, "start_as_current_span", None))


# ---------------------------------------------------------------------------
# telemetry module — pre-defined instruments
# ---------------------------------------------------------------------------


class TestPredefinedInstruments:
    def test_query_counter_is_counter(self) -> None:
        from nlqueries.telemetry import query_counter
        from opentelemetry.metrics import Counter

        assert isinstance(query_counter, Counter)

    def test_query_latency_is_histogram(self) -> None:
        from nlqueries.telemetry import query_latency
        from opentelemetry.metrics import Histogram

        assert isinstance(query_latency, Histogram)

    def test_chunk_search_latency_is_histogram(self) -> None:
        from nlqueries.telemetry import chunk_search_latency
        from opentelemetry.metrics import Histogram

        assert isinstance(chunk_search_latency, Histogram)

    def test_query_counter_add_does_not_raise(self) -> None:
        from nlqueries.telemetry import query_counter

        # No exporter configured — must be a no-op, not an error.
        query_counter.add(1, {"agent_type": "sql"})

    def test_query_latency_record_does_not_raise(self) -> None:
        from nlqueries.telemetry import query_latency

        query_latency.record(123.4, {"dialect": "postgres"})

    def test_chunk_search_latency_record_does_not_raise(self) -> None:
        from nlqueries.telemetry import chunk_search_latency

        chunk_search_latency.record(42.0, {"collection": "doc_abc_chunks"})


# ---------------------------------------------------------------------------
# Orchestrator span — unit tests via mock tracer
# ---------------------------------------------------------------------------


def _make_async_llm(tokens: list[str] | None = None) -> MagicMock:
    """Return an LLM mock with an async-generator ``astream`` method."""

    async def _astream(*a: object, **kw: object) -> object:
        for t in tokens or []:
            yield t

    mock_llm = MagicMock()
    mock_llm.supports_prompt_caching = False
    mock_llm.astream = _astream
    return mock_llm


class TestOrchestratorSpan:
    def test_orchestrator_emits_span(self) -> None:
        """handle_question must create an 'orchestrator.handle_question' span."""
        from nlqueries.orchestrator.sql_generation import SQLGenerationResult

        mock_tracer, mock_span = _make_mock_tracer()
        sql_result = SQLGenerationResult(
            sql="SELECT 1",
            is_valid=True,
            validation_error=None,
            dialect="postgres",
            attempt_count=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            _write_kb(kb_path, "agent1", _make_kb())

            with (
                patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
                patch(
                    "nlqueries.orchestrator.orchestrator.get_llm_client",
                    return_value=_make_async_llm(),
                ),
                patch(
                    "nlqueries.orchestrator.orchestrator.validate_and_repair",
                    new=AsyncMock(return_value=sql_result),
                ),
                patch("nlqueries.orchestrator.orchestrator.get_tracer", return_value=mock_tracer),
                patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
                patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            ):
                mock_cfg.KB_PATH = kb_path
                from nlqueries.orchestrator.orchestrator import Orchestrator

                asyncio.run(_collect(Orchestrator().handle_question("q", "agent1")))

        mock_tracer.start_as_current_span.assert_called_once_with("orchestrator.handle_question")

    def test_orchestrator_span_sets_agent_id_attribute(self) -> None:
        from nlqueries.orchestrator.sql_generation import SQLGenerationResult

        mock_tracer, mock_span = _make_mock_tracer()
        sql_result = SQLGenerationResult(
            sql="SELECT 1",
            is_valid=True,
            validation_error=None,
            dialect="postgres",
            attempt_count=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            _write_kb(kb_path, "my_agent", _make_kb())

            with (
                patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
                patch(
                    "nlqueries.orchestrator.orchestrator.get_llm_client",
                    return_value=_make_async_llm(),
                ),
                patch(
                    "nlqueries.orchestrator.orchestrator.validate_and_repair",
                    new=AsyncMock(return_value=sql_result),
                ),
                patch("nlqueries.orchestrator.orchestrator.get_tracer", return_value=mock_tracer),
                patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
                patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            ):
                mock_cfg.KB_PATH = kb_path
                from nlqueries.orchestrator.orchestrator import Orchestrator

                asyncio.run(_collect(Orchestrator().handle_question("q", "my_agent")))

        mock_span.set_attribute.assert_any_call("agent_id", "my_agent")

    def test_orchestrator_span_sets_sql_valid_and_attempt_count(self) -> None:
        from nlqueries.orchestrator.sql_generation import SQLGenerationResult

        mock_tracer, mock_span = _make_mock_tracer()
        sql_result = SQLGenerationResult(
            sql="SELECT 1",
            is_valid=True,
            validation_error=None,
            dialect="postgres",
            attempt_count=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            _write_kb(kb_path, "agent1", _make_kb())

            with (
                patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
                patch(
                    "nlqueries.orchestrator.orchestrator.get_llm_client",
                    return_value=_make_async_llm(),
                ),
                patch(
                    "nlqueries.orchestrator.orchestrator.validate_and_repair",
                    new=AsyncMock(return_value=sql_result),
                ),
                patch("nlqueries.orchestrator.orchestrator.get_tracer", return_value=mock_tracer),
                patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
                patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            ):
                mock_cfg.KB_PATH = kb_path
                from nlqueries.orchestrator.orchestrator import Orchestrator

                asyncio.run(_collect(Orchestrator().handle_question("q", "agent1")))

        mock_span.set_attribute.assert_any_call("sql_valid", True)
        mock_span.set_attribute.assert_any_call("attempt_count", 2)


# ---------------------------------------------------------------------------
# DocumentOrchestrator span — unit tests via mock tracer
# ---------------------------------------------------------------------------


class TestDocumentOrchestratorSpan:
    def test_document_orchestrator_emits_span(self) -> None:
        from nlqueries.orchestrator.document_retrieval import DocumentRetrievalResult

        mock_tracer, mock_span = _make_mock_tracer()
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([])
        mock_retrieval = DocumentRetrievalResult(
            chunks=[],
            citations=[],
            collection="doc_src_chunks",
        )

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                return_value=mock_retrieval,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.assemble_document_prompt",
                return_value=("sys", "usr"),
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_tracer",
                return_value=mock_tracer,
            ),
        ):
            from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

            asyncio.run(_collect(DocumentOrchestrator().handle_question("q", "doc_src_chunks")))

        mock_tracer.start_as_current_span.assert_called_once_with(
            "document_orchestrator.handle_question"
        )

    def test_document_orchestrator_span_sets_intent_type(self) -> None:
        from nlqueries.orchestrator.document_retrieval import DocumentRetrievalResult

        mock_tracer, mock_span = _make_mock_tracer()
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([])
        mock_retrieval = DocumentRetrievalResult(chunks=[], citations=[], collection="col")

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                return_value=mock_retrieval,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.assemble_document_prompt",
                return_value=("sys", "usr"),
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_tracer",
                return_value=mock_tracer,
            ),
        ):
            from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

            asyncio.run(_collect(DocumentOrchestrator().handle_question("q", "col")))

        mock_span.set_attribute.assert_any_call("intent_type", "document")


# ---------------------------------------------------------------------------
# qdrant_store spans — unit tests via mock tracer
# ---------------------------------------------------------------------------


class TestQdrantStoreSpans:
    def _make_chunk(self) -> Any:
        from nlqueries.document_connectors.base import DocumentChunk

        return DocumentChunk(
            chunk_id="a" * 16,
            source_id="src1",
            source_name="doc.pdf",
            page_number=1,
            chunk_index=0,
            text="hello world",
            metadata={},
        )

    def test_upsert_chunks_emits_span(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        chunk = self._make_chunk()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=[[0.0] * 384]),
            patch("nlqueries.embeddings.qdrant_store.get_tracer", return_value=mock_tracer),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_chunks

            upsert_chunks("doc_src1_chunks", [chunk])

        mock_tracer.start_as_current_span.assert_called_once_with("qdrant_store.upsert_chunks")

    def test_upsert_chunks_span_sets_collection_and_count(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        chunk = self._make_chunk()

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_batch", return_value=[[0.0] * 384]),
            patch("nlqueries.embeddings.qdrant_store.get_tracer", return_value=mock_tracer),
        ):
            from nlqueries.embeddings.qdrant_store import upsert_chunks

            upsert_chunks("doc_src1_chunks", [chunk])

        mock_span.set_attribute.assert_any_call("collection", "doc_src1_chunks")
        mock_span.set_attribute.assert_any_call("chunk_count", 1)

    def test_search_chunks_emits_span(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        mock_hit = MagicMock()
        mock_hit.payload = {
            "chunk_id": "a" * 16,
            "source_id": "src1",
            "source_name": "doc.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "hello",
            "metadata": {},
        }
        mock_hit.score = 0.9
        mock_client = MagicMock()
        mock_client.query_points.return_value.points = [mock_hit]

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.embeddings.qdrant_store.get_tracer", return_value=mock_tracer),
        ):
            from nlqueries.embeddings.qdrant_store import search_chunks

            search_chunks("doc_src1_chunks", "hello")

        mock_tracer.start_as_current_span.assert_called_once_with("qdrant_store.search_chunks")

    def test_search_chunks_span_sets_collection_attribute(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        mock_client = MagicMock()
        mock_client.query_points.return_value.points = []

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.embeddings.qdrant_store.get_tracer", return_value=mock_tracer),
        ):
            from nlqueries.embeddings.qdrant_store import search_chunks

            search_chunks("doc_src1_chunks", "test")

        mock_span.set_attribute.assert_any_call("collection", "doc_src1_chunks")


# ---------------------------------------------------------------------------
# PostgresConnector span — unit tests via mock tracer
# ---------------------------------------------------------------------------


class TestPostgresConnectorSpan:
    def test_execute_query_emits_span(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        from nlqueries.connectors.postgres import PostgresConnector

        connector = PostgresConnector()
        mock_cursor = MagicMock()
        mock_cursor.returns_rows = True
        mock_cursor.keys.return_value = ["id"]
        mock_cursor.fetchall.return_value = [(1,)]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_cursor
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn
        connector._engine = mock_engine

        with patch("nlqueries.connectors.postgres.get_tracer", return_value=mock_tracer):
            connector.execute_query("SELECT 1")

        mock_tracer.start_as_current_span.assert_called_once_with(
            "postgres_connector.execute_query"
        )

    def test_execute_query_span_sets_db_system_attribute(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        from nlqueries.connectors.postgres import PostgresConnector

        connector = PostgresConnector()
        mock_cursor = MagicMock()
        mock_cursor.returns_rows = False
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_cursor
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn
        connector._engine = mock_engine

        with patch("nlqueries.connectors.postgres.get_tracer", return_value=mock_tracer):
            connector.execute_query("SELECT 1")

        mock_span.set_attribute.assert_any_call("db.system", "postgresql")

    def test_execute_query_span_sets_error_attribute_on_failure(self) -> None:
        mock_tracer, mock_span = _make_mock_tracer()
        from nlqueries.connectors.postgres import PostgresConnector

        connector = PostgresConnector()
        mock_engine = MagicMock()
        mock_engine.begin.side_effect = RuntimeError("connection refused")
        connector._engine = mock_engine

        with patch("nlqueries.connectors.postgres.get_tracer", return_value=mock_tracer):
            result = connector.execute_query("SELECT 1")

        assert result.error is not None
        mock_span.set_attribute.assert_any_call("error", True)
