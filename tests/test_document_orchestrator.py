"""Tests for nlqueries.orchestrator.document_orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieval_result(num_chunks: int = 2) -> object:
    """Build a minimal DocumentRetrievalResult with mock chunks and citations."""
    from nlqueries.document_connectors.base import DocumentChunk
    from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult

    chunks = [
        DocumentChunk(
            chunk_id=f"chunk_{i}",
            source_id="src1",
            source_name="policy.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=f"Chunk text number {i}.",
            metadata={},
        )
        for i in range(num_chunks)
    ]
    citations = [
        Citation(
            chunk_id=f"chunk_{i}",
            source_name="policy.pdf",
            page_number=i + 1,
            chunk_index=i,
            excerpt=f"Chunk text number {i}.",
            relevance_score=0.9 - i * 0.1,
        )
        for i in range(num_chunks)
    ]
    return DocumentRetrievalResult(chunks=chunks, citations=citations, collection="col")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDocumentOrchestrator:
    def test_handle_question_streams_tokens_then_citations(self) -> None:
        """handle_question yields LLM tokens before the final citations JSON chunk."""
        import asyncio

        from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

        retrieval_result = _make_retrieval_result(2)
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["Hello", " world", "."])

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                return_value=retrieval_result,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
        ):
            orch = DocumentOrchestrator()

            async def _collect() -> list[str]:
                tokens: list[str] = []
                async for token in orch.handle_question("What is the policy?", "col"):
                    tokens.append(token)
                return tokens

            tokens = asyncio.run(_collect())

        # All LLM tokens come before the final citations chunk
        assert len(tokens) >= 2
        non_citation_tokens = [t for t in tokens if not _is_citations_chunk(t)]
        assert non_citation_tokens == ["Hello", " world", "."]
        # Citations chunk is the last item
        assert _is_citations_chunk(tokens[-1])

    def test_final_chunk_type_is_citations(self) -> None:
        """The last yielded item is a JSON object with type='citations'."""
        import asyncio

        from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

        retrieval_result = _make_retrieval_result(3)
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["Answer text."])

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                return_value=retrieval_result,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
        ):
            orch = DocumentOrchestrator()

            async def _collect() -> list[str]:
                tokens: list[str] = []
                async for token in orch.handle_question("query", "col"):
                    tokens.append(token)
                return tokens

            tokens = asyncio.run(_collect())

        final = json.loads(tokens[-1])
        assert final["type"] == "citations"
        assert isinstance(final["citations"], list)
        assert len(final["citations"]) == 3
        # Each citation entry has the expected fields
        for entry in final["citations"]:
            assert "source_name" in entry
            assert "page_number" in entry
            assert "excerpt" in entry

    def test_citations_payload_matches_retrieval_result(self) -> None:
        """Citations in the final chunk match the retrieval result's citation data."""
        import asyncio

        from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

        retrieval_result = _make_retrieval_result(2)
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["token"])

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                return_value=retrieval_result,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
        ):
            orch = DocumentOrchestrator()

            async def _collect() -> list[str]:
                tokens: list[str] = []
                async for token in orch.handle_question("query", "col"):
                    tokens.append(token)
                return tokens

            tokens = asyncio.run(_collect())

        final = json.loads(tokens[-1])
        assert final["citations"][0]["source_name"] == "policy.pdf"
        assert final["citations"][0]["page_number"] == 1
        assert "Chunk text number 0" in final["citations"][0]["excerpt"]

    def test_handle_question_passes_source_id_to_retrieval(self) -> None:
        """handle_question forwards source_id to retrieve_for_question."""
        import asyncio

        from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

        retrieval_result = _make_retrieval_result(1)
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([])
        mock_retrieve = MagicMock(return_value=retrieval_result)

        with (
            patch(
                "nlqueries.orchestrator.document_orchestrator.retrieve_for_question",
                mock_retrieve,
            ),
            patch(
                "nlqueries.orchestrator.document_orchestrator.get_llm_client",
                return_value=mock_llm,
            ),
        ):
            orch = DocumentOrchestrator()

            async def _run() -> None:
                async for _ in orch.handle_question(
                    "query", "col", source_id="my-source-uuid", top_k=3
                ):
                    pass

            asyncio.run(_run())

        mock_retrieve.assert_called_once_with(
            "query", "col", top_k=3, source_id_filter="my-source-uuid"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _is_citations_chunk(token: str) -> bool:
    """Return True if *token* is a valid JSON citations chunk."""
    try:
        parsed = json.loads(token)
        return isinstance(parsed, dict) and parsed.get("type") == "citations"
    except (ValueError, TypeError):
        return False
