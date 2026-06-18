"""Tests for nlqueries.orchestrator.document_retrieval and assemble_document_prompt."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_VECTOR_SIZE = 384
_FAKE_VEC = [0.0] * _VECTOR_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(
    chunk_id: str,
    source_id: str,
    source_name: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
    score: float,
    metadata: dict | None = None,
) -> MagicMock:
    """Build a mock Qdrant ScoredPoint-like object."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_name": source_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": text,
        "metadata": metadata or {},
    }
    return hit


def _make_response(hits: list[MagicMock]) -> MagicMock:
    r = MagicMock()
    r.points = hits
    return r


# ---------------------------------------------------------------------------
# Tests — retrieve_for_question
# ---------------------------------------------------------------------------


class TestRetrieveForQuestion:
    def test_retrieve_returns_citations(self) -> None:
        """retrieve_for_question returns one Citation per chunk with correct excerpt."""
        from nlqueries.orchestrator.document_retrieval import Citation, retrieve_for_question

        long_text = "A" * 300
        hits = [
            _make_hit("id1", "src1", "doc.pdf", 1, 0, long_text, 0.8),
            _make_hit("id2", "src1", "doc.pdf", 2, 0, "Short text.", 0.6),
            _make_hit("id3", "src1", "doc.pdf", 3, 0, "Another chunk.", 0.4),
        ]
        mock_client = MagicMock()
        mock_client.query_points.return_value = _make_response(hits)

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=_FAKE_VEC),
        ):
            result = retrieve_for_question("What is this?", "doc_src1_chunks", top_k=3)

        assert len(result.citations) == 3
        assert all(isinstance(c, Citation) for c in result.citations)
        # excerpt is the first 200 chars of the chunk text
        assert result.citations[0].excerpt == long_text[:200]
        assert result.citations[1].excerpt == "Short text."
        assert result.citations[2].excerpt == "Another chunk."
        assert result.collection == "doc_src1_chunks"

    def test_retrieve_sorted_by_relevance(self) -> None:
        """retrieve_for_question returns chunks sorted by relevance score descending."""
        from nlqueries.orchestrator.document_retrieval import retrieve_for_question

        hits = [
            _make_hit("id1", "src1", "doc.pdf", 1, 0, "chunk A", 0.7),
            _make_hit("id2", "src1", "doc.pdf", 2, 0, "chunk B", 0.9),
            _make_hit("id3", "src1", "doc.pdf", 3, 0, "chunk C", 0.5),
        ]
        mock_client = MagicMock()
        mock_client.query_points.return_value = _make_response(hits)

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=_FAKE_VEC),
        ):
            result = retrieve_for_question("query", "doc_src1_chunks")

        scores = [c.relevance_score for c in result.citations]
        assert scores == sorted(scores, reverse=True)
        assert scores == [0.9, 0.7, 0.5]

    def test_retrieve_empty_collection_returns_empty_result(self) -> None:
        """retrieve_for_question handles an empty Qdrant response gracefully."""
        from nlqueries.orchestrator.document_retrieval import retrieve_for_question

        mock_client = MagicMock()
        mock_client.query_points.return_value = _make_response([])

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=_FAKE_VEC),
        ):
            result = retrieve_for_question("query", "doc_src1_chunks")

        assert result.chunks == []
        assert result.citations == []

    def test_retrieve_passes_source_id_filter_to_qdrant(self) -> None:
        """retrieve_for_question forwards source_id_filter as a Qdrant query filter."""
        from nlqueries.orchestrator.document_retrieval import retrieve_for_question

        mock_client = MagicMock()
        mock_client.query_points.return_value = _make_response([])

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=_FAKE_VEC),
        ):
            retrieve_for_question("query", "doc_src1_chunks", source_id_filter="src1")

        _, kwargs = mock_client.query_points.call_args
        assert kwargs.get("query_filter") is not None

    def test_retrieve_no_filter_when_source_id_filter_is_none(self) -> None:
        """retrieve_for_question passes no filter when source_id_filter is None."""
        from nlqueries.orchestrator.document_retrieval import retrieve_for_question

        mock_client = MagicMock()
        mock_client.query_points.return_value = _make_response([])

        with (
            patch("nlqueries.embeddings.qdrant_store._client", mock_client),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=_FAKE_VEC),
        ):
            retrieve_for_question("query", "doc_src1_chunks")

        _, kwargs = mock_client.query_points.call_args
        assert kwargs.get("query_filter") is None


# ---------------------------------------------------------------------------
# Tests — assemble_document_prompt
# ---------------------------------------------------------------------------


class TestAssembleDocumentPrompt:
    def _make_retrieval_result(self) -> object:
        from nlqueries.document_connectors.base import DocumentChunk
        from nlqueries.orchestrator.document_retrieval import (
            Citation,
            DocumentRetrievalResult,
        )

        chunks = [
            DocumentChunk(
                chunk_id="aabb",
                source_id="src1",
                source_name="policy.pdf",
                page_number=2,
                chunk_index=0,
                text="The refund policy allows returns within 30 days.",
                metadata={},
            ),
            DocumentChunk(
                chunk_id="ccdd",
                source_id="src1",
                source_name="policy.pdf",
                page_number=None,
                chunk_index=1,
                text="No refunds after 30 days.",
                metadata={},
            ),
        ]
        citations = [
            Citation(
                chunk_id="aabb",
                source_name="policy.pdf",
                page_number=2,
                chunk_index=0,
                excerpt="The refund policy allows returns within 30 days.",
                relevance_score=0.9,
            ),
            Citation(
                chunk_id="ccdd",
                source_name="policy.pdf",
                page_number=None,
                chunk_index=1,
                excerpt="No refunds after 30 days.",
                relevance_score=0.7,
            ),
        ]
        return DocumentRetrievalResult(chunks=chunks, citations=citations, collection="col")

    def test_assemble_document_prompt_includes_citations(self) -> None:
        """assemble_document_prompt system prompt instructs the LLM to cite sources."""
        from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt

        result = self._make_retrieval_result()
        system_prompt, user_prompt = assemble_document_prompt("What is the refund policy?", result)  # type: ignore[arg-type]

        assert "cite sources" in system_prompt.lower() or "cite" in system_prompt.lower()
        assert "source" in system_prompt.lower()

    def test_assemble_document_prompt_user_prompt_contains_question(self) -> None:
        """assemble_document_prompt embeds the original question in the user prompt."""
        from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt

        result = self._make_retrieval_result()
        _, user_prompt = assemble_document_prompt("What is the refund policy?", result)  # type: ignore[arg-type]

        assert "What is the refund policy?" in user_prompt

    def test_assemble_document_prompt_includes_chunk_text(self) -> None:
        """assemble_document_prompt embeds chunk text in the user prompt."""
        from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt

        result = self._make_retrieval_result()
        _, user_prompt = assemble_document_prompt("What is the refund policy?", result)  # type: ignore[arg-type]

        assert "refund policy allows returns within 30 days" in user_prompt
        assert "No refunds after 30 days" in user_prompt

    def test_assemble_document_prompt_labels_with_page_number(self) -> None:
        """assemble_document_prompt includes page number in chunk labels when available."""
        from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt

        result = self._make_retrieval_result()
        _, user_prompt = assemble_document_prompt("query", result)  # type: ignore[arg-type]

        assert "page 2" in user_prompt

    def test_assemble_document_prompt_omits_page_when_none(self) -> None:
        """assemble_document_prompt omits page number label when page_number is None."""
        from nlqueries.orchestrator.prompt_assembly import assemble_document_prompt

        result = self._make_retrieval_result()
        _, user_prompt = assemble_document_prompt("query", result)  # type: ignore[arg-type]

        assert "policy.pdf]" in user_prompt  # no page number for chunk 2
