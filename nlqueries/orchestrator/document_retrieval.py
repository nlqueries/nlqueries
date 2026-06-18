"""
nlqueries.orchestrator.document_retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Document retrieval with source attribution for the document Q&A pipeline.

Given a question, finds the most relevant chunks from a Qdrant collection and
returns them paired with citation metadata suitable for LLM grounding.

Public API
----------
``retrieve_for_question(question, collection, top_k, source_id_filter)``
    Search the Qdrant collection and return a :class:`DocumentRetrievalResult`
    with chunks and citations sorted by relevance score descending.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlqueries.document_connectors.base import DocumentChunk


@dataclass
class Citation:
    """Source attribution for a single retrieved document chunk."""

    chunk_id: str
    source_name: str
    page_number: int | None
    chunk_index: int
    excerpt: str  # first 200 chars of the chunk text
    relevance_score: float


@dataclass
class DocumentRetrievalResult:
    """Result of a document retrieval operation."""

    chunks: list[DocumentChunk]
    citations: list[Citation]
    collection: str


def retrieve_for_question(
    question: str,
    collection: str,
    top_k: int = 5,
    source_id_filter: str | None = None,
) -> DocumentRetrievalResult:
    """Search *collection* for chunks relevant to *question*.

    Performs a semantic search against the Qdrant collection and constructs a
    :class:`Citation` for each returned chunk.  Results are sorted by
    relevance score descending.

    Args:
        question:         Natural-language question to search against.
        collection:       Qdrant collection name (convention: ``doc_{source_id}_chunks``).
        top_k:            Maximum number of chunks to retrieve (default 5).
        source_id_filter: Restrict results to this source ID when set.

    Returns:
        :class:`DocumentRetrievalResult` with chunks and citations sorted by
        relevance score descending.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from nlqueries.embeddings.embedder import embed_text
    from nlqueries.embeddings.qdrant_store import _get_client  # noqa: PLC2701

    vector = embed_text(question)
    query_filter: Filter | None = None
    if source_id_filter is not None:
        query_filter = Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id_filter))]
        )

    client = _get_client()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )

    chunks: list[DocumentChunk] = []
    citations: list[Citation] = []

    for hit in response.points:
        p = hit.payload or {}
        chunk = DocumentChunk(
            chunk_id=str(p.get("chunk_id", "")),
            source_id=str(p.get("source_id", "")),
            source_name=str(p.get("source_name", "")),
            page_number=p.get("page_number"),
            chunk_index=int(p.get("chunk_index", 0)),
            text=str(p.get("text", "")),
            metadata=dict(p.get("metadata") or {}),
        )
        chunks.append(chunk)
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                source_name=chunk.source_name,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                excerpt=chunk.text[:200],
                relevance_score=float(hit.score),
            )
        )

    # Sort by relevance score descending
    paired = sorted(
        zip(chunks, citations, strict=True),
        key=lambda x: x[1].relevance_score,
        reverse=True,
    )
    if paired:
        sorted_chunks, sorted_citations = zip(*paired, strict=True)
        return DocumentRetrievalResult(
            chunks=list(sorted_chunks),
            citations=list(sorted_citations),
            collection=collection,
        )
    return DocumentRetrievalResult(chunks=[], citations=[], collection=collection)
