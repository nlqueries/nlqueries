"""
nlqueries.embeddings.qdrant_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Qdrant vector-store helpers for QueryCapsule, schema, and document chunk
upsert and nearest-neighbour search.

Public API
----------
``ensure_collection(name, vector_size)``
    Create a Qdrant collection with cosine distance if it does not exist yet.

``upsert_capsules(collection, capsules, agent_id)``
    Embed each capsule's intent (falling back to ``auto_description``) and
    upsert into Qdrant with a structured payload.

``search(collection, query, top_k)``
    Embed *query*, run a nearest-neighbour search, and return the top-k
    ``QueryCapsule`` objects reconstructed from the stored payload.

``upsert_schema(collection, schema, agent_id)``
    Embed table and column descriptions from a ``SchemaSpec`` and upsert
    into Qdrant with ``type`` payloads of ``"table"`` or ``"column"``.

``search_schema(collection, query, top_k)``
    Embed *query*, filter by ``type IN ["table", "column"]``, and return
    matching payloads with scores.

``upsert_chunks(collection, chunks)``
    Embed each ``DocumentChunk``'s text and upsert into Qdrant.
    Collection naming convention: ``doc_{source_id}_chunks``.

``search_chunks(collection, query, top_k, source_id_filter)``
    Embed *query* and return the top-k most similar ``DocumentChunk`` objects.
    Optionally filter by ``source_id``.

``delete_chunks(collection, source_id)``
    Delete all chunks belonging to *source_id* from the collection.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from typing import TYPE_CHECKING, Any

from nlqueries.connectors.base import SchemaSpec
from nlqueries.document_connectors.base import DocumentChunk
from nlqueries.processing.parameterizer import QueryCapsule
from nlqueries.telemetry import chunk_search_latency, get_tracer

if TYPE_CHECKING:
    from qdrant_client import QdrantClient as _QdrantClient

# ---------------------------------------------------------------------------
# Lazy client singleton
# ---------------------------------------------------------------------------

_client: _QdrantClient | None = None

# Tracks "{collection}:{field}" keys that already have a payload index created,
# so repeated put() calls skip the create_payload_index round-trip.
_indexed_fields: set[str] = set()


def _get_client() -> _QdrantClient:
    """Return the shared ``QdrantClient``, creating it on first call."""
    global _client  # noqa: PLW0603
    if _client is None:
        from nlqueries.embeddings.qdrant_client import build_qdrant_client  # deferred

        _client = build_qdrant_client()
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema_point_id(key: str) -> int:
    """Derive a stable integer point-ID from a schema key string."""
    digest = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()  # noqa: S324
    return int(digest[:16], 16)


def _capsule_id(capsule: QueryCapsule, index: int) -> int:
    """Derive a stable integer point-ID from the capsule's template SQL.

    Falls back to *index* so that callers never produce duplicate IDs within
    a single batch (even for identical SQL templates).
    """
    raw = f"{index}:{capsule.template_sql}"
    digest = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()  # noqa: S324
    # Qdrant point IDs must be unsigned 64-bit integers.
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_collection(
    name: str,
    vector_size: int = 384,
    *,
    payload_indexes: list[str] | None = None,
    quantize: bool = True,
) -> None:
    """Create a Qdrant collection with cosine distance if it does not exist.

    Args:
        name:            Collection name.
        vector_size:     Dimensionality of the stored vectors (default 384 for
                         ``all-MiniLM-L6-v2``).
        payload_indexes: Optional list of payload field names to index as
                         keyword fields.  Existing indexes are skipped via a
                         module-level cache, so repeated calls are cheap.
        quantize:        When ``True`` (default), enable INT8 scalar quantization
                         on new collections.  Reduces on-disk and RAM footprint
                         ~4× with <1% recall loss.  Pass ``False`` to disable
                         (e.g. for small test collections).
    """
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        create_kwargs: dict[str, Any] = {
            "collection_name": name,
            "vectors_config": VectorParams(size=vector_size, distance=Distance.COSINE),
        }
        if quantize:
            from qdrant_client.models import (  # noqa: PLC0415
                ScalarQuantization,
                ScalarQuantizationConfig,
                ScalarType,
            )

            create_kwargs["quantization_config"] = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                )
            )
        client.create_collection(**create_kwargs)

    if payload_indexes:
        from qdrant_client.models import PayloadSchemaType

        for field in payload_indexes:
            key = f"{name}:{field}"
            if key not in _indexed_fields:
                with contextlib.suppress(Exception):
                    client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                _indexed_fields.add(key)


def upsert_capsules(
    collection: str,
    capsules: list[QueryCapsule],
    agent_id: str = "",
) -> None:
    """Embed and upsert *capsules* into *collection*.

    Each capsule's ``intent`` is embedded when non-empty; otherwise
    ``auto_description`` is used as the fallback text.  The full payload
    stored alongside each vector is::

        {
            "capsule_id": <int>,
            "template_sql": <str>,
            "tables": <list[str]>,
            "frequency": <int>,
            "agent_id": <str>,
        }

    Args:
        collection: Target Qdrant collection name.
        capsules:   Capsules to upsert.
        agent_id:   Optional agent / connector identifier stored in payload.
    """
    if not capsules:
        return

    from qdrant_client.models import PointStruct

    from nlqueries.embeddings.embedder import embed_batch

    texts = [c.intent if c.intent else c.auto_description for c in capsules]
    vectors = embed_batch(texts)

    client = _get_client()
    points = [
        PointStruct(
            id=_capsule_id(capsule, idx),
            vector=vectors[idx],
            payload={
                "capsule_id": _capsule_id(capsule, idx),
                "template_sql": capsule.template_sql,
                "tables": capsule.tables,
                "frequency": capsule.frequency,
                "agent_id": agent_id,
            },
        )
        for idx, capsule in enumerate(capsules)
    ]
    client.upsert(collection_name=collection, points=points)


def search(
    collection: str,
    query: str,
    top_k: int = 5,
    *,
    vector: list[float] | None = None,
) -> list[QueryCapsule]:
    """Embed *query*, search *collection*, and return ``QueryCapsule`` objects.

    Capsules are reconstructed from the payload stored alongside each vector.
    Fields not persisted in the payload (``placeholders``, ``columns``,
    ``auto_description``, ``intent``) are given sensible defaults.

    Args:
        collection: Qdrant collection to search.
        query:      Natural-language query string to embed (ignored when
                    *vector* is supplied).
        top_k:      Maximum number of results to return.
        vector:     Pre-computed embedding vector.  When provided, *query* is
                    not embedded again, saving one embed_text() call per request.

    Returns:
        ``list[QueryCapsule]`` of length ≤ *top_k*, ordered by relevance.
    """
    if vector is None:
        from nlqueries.embeddings.embedder import embed_text

        vector = embed_text(query)
    client = _get_client()
    response = client.query_points(collection_name=collection, query=vector, limit=top_k)

    capsules: list[QueryCapsule] = []
    for hit in response.points:
        payload = hit.payload or {}
        capsules.append(
            QueryCapsule(
                template_sql=str(payload.get("template_sql", "")),
                placeholders=[],
                tables=list(payload.get("tables", [])),
                columns=[],
                frequency=int(payload.get("frequency", 0)),
                auto_description="",
                intent="",
            )
        )
    return capsules


def upsert_schema(
    collection: str,
    schema: SchemaSpec,
    agent_id: str = "",
) -> None:
    """Embed and upsert schema descriptions into *collection*.

    One point is created per table; an additional point is created for each
    column that has a non-empty description.

    Table embedding text: ``"{table}: {description}. Columns: {name type, ...}"``
    Column embedding text: ``"{table}.{col} ({type}): {description}"``

    Payload for tables:  ``{agent_id, table_name, type: "table"}``
    Payload for columns: ``{agent_id, table_name, column_name, type: "column"}``

    Args:
        collection: Target Qdrant collection name.
        schema:     Schema whose tables and columns will be embedded.
        agent_id:   Optional agent / connector identifier stored in payload.
    """
    if not schema.tables:
        return

    from qdrant_client.models import PointStruct

    from nlqueries.embeddings.embedder import embed_batch

    texts: list[str] = []
    payloads: list[dict[str, Any]] = []

    for table in schema.tables:
        col_summary = ", ".join(f"{col.name} {col.type}" for col in table.columns)
        if table.description:
            table_text = f"{table.name}: {table.description}. Columns: {col_summary}"
        else:
            table_text = f"{table.name}. Columns: {col_summary}"
        texts.append(table_text)
        payloads.append({"agent_id": agent_id, "table_name": table.name, "type": "table"})

        for col in table.columns:
            if col.description:
                col_text = f"{table.name}.{col.name} ({col.type}): {col.description}"
                texts.append(col_text)
                payloads.append(
                    {
                        "agent_id": agent_id,
                        "table_name": table.name,
                        "column_name": col.name,
                        "type": "column",
                    }
                )

    vectors = embed_batch(texts)
    client = _get_client()
    points = [
        PointStruct(
            id=_schema_point_id(
                f"{payloads[i]['type']}:{agent_id}:{payloads[i]['table_name']}"
                f":{payloads[i].get('column_name', '')}"
            ),
            vector=vectors[i],
            payload=payloads[i],
        )
        for i in range(len(texts))
    ]
    client.upsert(collection_name=collection, points=points)


def search_schema(
    collection: str,
    query: str,
    top_k: int = 10,
    *,
    vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Embed *query* and search for matching schema objects in *collection*.

    Only points whose ``type`` payload field is ``"table"`` or ``"column"``
    are returned.

    Args:
        collection: Qdrant collection to search.
        query:      Natural-language query string to embed (ignored when
                    *vector* is supplied).
        top_k:      Maximum number of results to return.
        vector:     Pre-computed embedding vector.  When provided, *query* is
                    not embedded again, saving one embed_text() call per request.

    Returns:
        ``list[dict]`` each containing the stored payload plus a ``"score"`` key,
        ordered by relevance descending.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    if vector is None:
        from nlqueries.embeddings.embedder import embed_text

        vector = embed_text(query)
    query_filter = Filter(
        must=[FieldCondition(key="type", match=MatchAny(any=["table", "column"]))]
    )
    client = _get_client()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [{**(hit.payload or {}), "score": hit.score} for hit in response.points]


# ---------------------------------------------------------------------------
# Document chunk API (Task 9.2)
# Collection naming convention: doc_{source_id}_chunks
# ---------------------------------------------------------------------------

DOCUMENT_VECTOR_SIZE = 384  # same as existing sentence-transformer model


def upsert_chunks(
    collection: str,
    chunks: list[DocumentChunk],
) -> None:
    """Embed each chunk's text and upsert into Qdrant.

    Calls ``ensure_collection`` before upserting so callers do not need to
    create the collection manually.  Each point payload includes all
    ``DocumentChunk`` fields: ``chunk_id``, ``source_id``, ``source_name``,
    ``page_number``, ``chunk_index``, ``text``, ``metadata``.

    Args:
        collection: Target collection name (convention: ``doc_{source_id}_chunks``).
        chunks:     Document chunks to embed and store.
    """
    if not chunks:
        return

    from qdrant_client.models import PointStruct

    from nlqueries.embeddings.embedder import embed_batch

    tracer = get_tracer()
    with tracer.start_as_current_span("qdrant_store.upsert_chunks") as span:
        span.set_attribute("collection", collection)
        span.set_attribute("chunk_count", len(chunks))

        ensure_collection(collection, DOCUMENT_VECTOR_SIZE)

        texts = [c.text for c in chunks]
        vectors = embed_batch(texts)

        client = _get_client()
        points = [
            PointStruct(
                id=int(chunk.chunk_id, 16),
                vector=vectors[idx],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_name": chunk.source_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
        client.upsert(collection_name=collection, points=points)


def search_chunks(
    collection: str,
    query: str,
    top_k: int = 5,
    source_id_filter: str | None = None,
) -> list[DocumentChunk]:
    """Embed *query* and return the top-k most similar ``DocumentChunk`` objects.

    When *source_id_filter* is provided, only chunks whose ``source_id``
    matches are considered.

    Args:
        collection:       Qdrant collection to search.
        query:            Natural-language query string to embed.
        top_k:            Maximum number of results to return.
        source_id_filter: Restrict results to this source ID when set.

    Returns:
        ``list[DocumentChunk]`` of length ≤ *top_k*, ordered by relevance.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from nlqueries.embeddings.embedder import embed_text

    tracer = get_tracer()
    start_ms = time.perf_counter() * 1000
    with tracer.start_as_current_span("qdrant_store.search_chunks") as span:
        span.set_attribute("collection", collection)
        span.set_attribute("top_k", top_k)
        if source_id_filter is not None:
            span.set_attribute("source_id_filter", source_id_filter)

        vector = embed_text(query)
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

        result: list[DocumentChunk] = []
        for hit in response.points:
            p = hit.payload or {}
            result.append(
                DocumentChunk(
                    chunk_id=str(p.get("chunk_id", "")),
                    source_id=str(p.get("source_id", "")),
                    source_name=str(p.get("source_name", "")),
                    page_number=p.get("page_number"),
                    chunk_index=int(p.get("chunk_index", 0)),
                    text=str(p.get("text", "")),
                    metadata=dict(p.get("metadata") or {}),
                )
            )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        chunk_search_latency.record(elapsed_ms, {"collection": collection})
        span.set_attribute("result_count", len(result))
        return result


def delete_chunks(collection: str, source_id: str) -> None:
    """Delete all chunks belonging to *source_id* from *collection*.

    Uses a Qdrant filter on the ``source_id`` payload field so only chunks
    for the given document are removed; other documents in the collection
    are untouched.

    Args:
        collection: Qdrant collection name.
        source_id:  The source identifier whose chunks should be deleted.
    """
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    filter_ = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
    client = _get_client()
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(filter=filter_),
    )
