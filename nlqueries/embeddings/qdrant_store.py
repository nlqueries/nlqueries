"""
nlqueries.embeddings.qdrant_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Qdrant vector-store helpers for QueryCapsule and schema upsert and
nearest-neighbour search.

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
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from nlqueries import config
from nlqueries.connectors.base import SchemaSpec
from nlqueries.processing.parameterizer import QueryCapsule

if TYPE_CHECKING:
    from qdrant_client import QdrantClient as _QdrantClient

# ---------------------------------------------------------------------------
# Lazy client singleton
# ---------------------------------------------------------------------------

_client: _QdrantClient | None = None


def _get_client() -> _QdrantClient:
    """Return the shared ``QdrantClient``, creating it on first call."""
    global _client  # noqa: PLW0603
    if _client is None:
        from qdrant_client import QdrantClient  # deferred — heavy import

        _client = QdrantClient(url=config.QDRANT_URL)
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


def ensure_collection(name: str, vector_size: int = 384) -> None:
    """Create a Qdrant collection with cosine distance if it does not exist.

    Args:
        name:        Collection name.
        vector_size: Dimensionality of the stored vectors (default 384 for
                     ``all-MiniLM-L6-v2``).
    """
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


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
) -> list[QueryCapsule]:
    """Embed *query*, search *collection*, and return ``QueryCapsule`` objects.

    Capsules are reconstructed from the payload stored alongside each vector.
    Fields not persisted in the payload (``placeholders``, ``columns``,
    ``auto_description``, ``intent``) are given sensible defaults.

    Args:
        collection: Qdrant collection to search.
        query:      Natural-language query string to embed.
        top_k:      Maximum number of results to return.

    Returns:
        ``list[QueryCapsule]`` of length ≤ *top_k*, ordered by relevance.
    """
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
) -> list[dict[str, Any]]:
    """Embed *query* and search for matching schema objects in *collection*.

    Only points whose ``type`` payload field is ``"table"`` or ``"column"``
    are returned.

    Args:
        collection: Qdrant collection to search.
        query:      Natural-language query string to embed.
        top_k:      Maximum number of results to return.

    Returns:
        ``list[dict]`` each containing the stored payload plus a ``"score"`` key,
        ordered by relevance descending.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

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
