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

``retrieve_across(question, collections, top_k)``
    The same over several collections -- an agent's listed document sources --
    keeping the *top_k* nearest overall.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nlqueries.document_connectors.base import DocumentChunk

logger = logging.getLogger(__name__)

#: Collections that could not be searched, reported once each per process.
_UNSEARCHABLE_WARNED: set[str] = set()


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
    from nlqueries.embeddings.qdrant_store import _get_client, chunk_from_payload  # noqa: PLC2701

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
    return _result(_cited(response.points, chunk_from_payload), collection, top_k)


def retrieve_across(
    question: str, collections: Sequence[str], top_k: int = 5
) -> DocumentRetrievalResult:
    """Search every collection in *collections* and keep the *top_k* nearest overall.

    For an agent that answers from several document sources. The question is
    embedded once. A collection that cannot be searched -- a listed source not
    ingested yet, or Qdrant refusing -- is skipped with a warning, once per
    collection, so the others still answer. The result's ``collection`` names
    them all.
    """
    from nlqueries.embeddings.embedder import embed_text
    from nlqueries.embeddings.qdrant_store import _get_client, chunk_from_payload  # noqa: PLC2701

    vector = embed_text(question)
    client = _get_client()
    pairs: list[tuple[DocumentChunk, Citation]] = []
    for collection in collections:
        try:
            response = client.query_points(collection_name=collection, query=vector, limit=top_k)
        except Exception as exc:  # noqa: BLE001 - one source's failure leaves the others
            if collection not in _UNSEARCHABLE_WARNED:
                _UNSEARCHABLE_WARNED.add(collection)
                logger.warning(
                    "Could not search %r (%s); answering from the other document sources. "
                    "A listed source that was never ingested has no collection.",
                    collection,
                    type(exc).__name__,
                )
            continue
        pairs.extend(_cited(response.points, chunk_from_payload))
    return _result(pairs, ", ".join(collections), top_k)


def _cited(points: Any, chunk_from_payload: Any) -> list[tuple[DocumentChunk, Citation]]:
    """Each readable hit as its chunk and the citation that names it."""
    pairs: list[tuple[DocumentChunk, Citation]] = []
    for hit in points:
        chunk = chunk_from_payload(hit.payload or {})
        if chunk is None:
            continue
        pairs.append(
            (
                chunk,
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_name=chunk.source_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    excerpt=chunk.text[:200],
                    relevance_score=float(hit.score),
                ),
            )
        )
    return pairs


def _result(
    pairs: list[tuple[DocumentChunk, Citation]], collection: str, top_k: int
) -> DocumentRetrievalResult:
    """*pairs* nearest first, at most *top_k* of them."""
    ranked = sorted(pairs, key=lambda pair: pair[1].relevance_score, reverse=True)[:top_k]
    return DocumentRetrievalResult(
        chunks=[chunk for chunk, _ in ranked],
        citations=[citation for _, citation in ranked],
        collection=collection,
    )
