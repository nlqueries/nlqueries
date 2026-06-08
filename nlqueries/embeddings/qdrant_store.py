"""
nlqueries.embeddings.qdrant_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Qdrant vector-store helpers for QueryCapsule upsert and nearest-neighbour
search.

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
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from nlqueries import config
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
