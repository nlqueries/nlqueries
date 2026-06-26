"""
nlqueries.cache.semantic_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic query cache backed by Qdrant.

Instead of exact-match caching, this module embeds each question and performs
nearest-neighbour search in a per-agent Qdrant collection.  If a sufficiently
similar question was answered recently (cosine similarity >= 0.97) and has not
yet expired, the cached answer is returned without hitting the LLM.

Collection naming convention: ``cache_{agent_id}`` (one per agent, separate
from the ``agent_{id}_schema``, ``agent_{id}_capsules`` and
``doc_{source_id}_chunks`` collections used elsewhere).

Public API
----------
``CacheEntry``
    Dataclass representing a stored cache entry.
``SemanticCache``
    Per-agent cache manager with ``get``, ``put``, ``invalidate``, and
    ``stats`` methods.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from nlqueries.embeddings.embedder import embed_text
from nlqueries.embeddings.qdrant_store import ensure_collection

CACHE_COLLECTION_PREFIX = "cache_"
CACHE_VECTOR_SIZE = 384
SIMILARITY_THRESHOLD = 0.97


# ---------------------------------------------------------------------------
# Protocol — accepted by SemanticCache.put(); satisfied by AgentQueryResult
# and by the internal _CacheData carrier in multi_agent_orchestrator.
# ---------------------------------------------------------------------------


class _PutResult(Protocol):
    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Stored cache entry returned on a semantic cache hit."""

    question: str
    resolved_question: str
    agent_type: str
    answer: str
    sql: str | None
    created_at: datetime
    hit_count: int


# ---------------------------------------------------------------------------
# Lazy QdrantClient singleton (separate from qdrant_store's singleton so that
# the cache module can be tested independently without touching capsule/chunk
# collections).
# ---------------------------------------------------------------------------

_cache_client: Any = None


def _get_client() -> Any:
    """Return a shared QdrantClient instance, creating it lazily on first call."""
    global _cache_client  # noqa: PLW0603
    if _cache_client is None:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        from nlqueries import config  # noqa: PLC0415

        _cache_client = QdrantClient(url=config.QDRANT_URL)
    return _cache_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point_id_for_question(question: str) -> int:
    """Derive a deterministic unsigned-64-bit Qdrant point ID from a question.

    Uses the first 16 hex characters of SHA-256(question), which gives a
    uniform distribution over 2^64 values — collision probability is
    negligible for any realistic cache size.
    """
    digest = hashlib.sha256(question.encode()).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Semantic query cache for a single agent, backed by a Qdrant collection.

    Usage::

        cache = SemanticCache(agent_id="sales-agent", ttl_hours=24)
        entry = cache.get("How many orders last month?")
        if entry is not None:
            return cached_response(entry)
        result = run_orchestrator(question)
        cache.put("How many orders last month?", result)
    """

    def __init__(self, agent_id: str, ttl_hours: int = 24) -> None:
        self._collection = f"{CACHE_COLLECTION_PREFIX}{agent_id}"
        self._ttl_hours = ttl_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question: str) -> CacheEntry | None:
        """Embed *question* and search the cache collection.

        Returns the top hit if its cosine similarity is >= SIMILARITY_THRESHOLD
        and the entry has not yet expired (``created_at + ttl_hours``).
        Increments ``hit_count`` in the Qdrant payload on a successful hit.

        Returns ``None`` when the collection does not exist, no similar entry
        is found, the top score is below threshold, or the entry has expired.
        """
        client = _get_client()

        # Guard: collection might not exist yet.
        try:
            existing = {c.name for c in client.get_collections().collections}
            if self._collection not in existing:
                return None
        except Exception:  # noqa: BLE001
            return None

        vector = embed_text(question)

        try:
            response = client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=1,
            )
        except Exception:  # noqa: BLE001
            return None

        if not response.points:
            return None

        hit = response.points[0]
        if hit.score < SIMILARITY_THRESHOLD:
            return None

        payload = hit.payload or {}
        created_at_raw = payload.get("created_at")
        if not created_at_raw:
            return None

        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except ValueError:
            return None

        if datetime.now(UTC) - created_at > timedelta(hours=self._ttl_hours):
            return None  # expired

        # Increment hit_count (best-effort — do not fail the whole get on error).
        new_count = int(payload.get("hit_count", 0)) + 1
        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=self._collection,
                payload={"hit_count": new_count},
                points=[hit.id],
            )

        return CacheEntry(
            question=str(payload.get("question", "")),
            resolved_question=str(payload.get("resolved_question", "")),
            agent_type=str(payload.get("agent_type", "")),
            answer=str(payload.get("answer", "")),
            sql=payload.get("sql") or None,
            created_at=created_at,
            hit_count=new_count,
        )

    def put(self, question: str, result: _PutResult) -> None:
        """Embed *question* and upsert the answer into the cache collection.

        Uses ``SHA-256(question)[:16]`` as the point ID so that repeated puts
        for the same question are idempotent (last write wins).
        Sets ``created_at`` to now and initialises ``hit_count`` to 0.

        Args:
            question: The (resolved) question string used as the cache key.
            result:   Any object exposing ``resolved_question``, ``agent_type``,
                      ``answer``, and ``sql`` attributes (e.g. AgentQueryResult
                      or the internal ``_CacheData`` carrier used by the
                      orchestrator).
        """
        import logging as _logging

        _logging.getLogger(__name__).debug("cache.put collection=%r", self._collection)

        from qdrant_client.models import PointStruct  # noqa: PLC0415

        ensure_collection(self._collection, CACHE_VECTOR_SIZE)

        vector = embed_text(question)
        point_id = _point_id_for_question(question)

        payload: dict[str, Any] = {
            "question": question,
            "resolved_question": result.resolved_question,
            "agent_type": result.agent_type,
            "answer": result.answer,
            "sql": result.sql,
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
        }

        client = _get_client()
        client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def invalidate(self, agent_id: str) -> None:  # noqa: ARG002
        """Delete all points in the cache collection (full invalidation).

        The *agent_id* parameter is accepted for API symmetry with CLI callers
        but is not used — the collection to clear is always ``self._collection``.
        """
        with contextlib.suppress(Exception):
            _get_client().delete_collection(self._collection)

    def list_entries(self, limit: int = 100) -> list[CacheEntry]:
        """Return up to *limit* cached entries, ordered by creation time descending.

        Uses Qdrant's ``scroll`` API to page through all points without
        requiring a query vector.  Returns an empty list if the collection
        does not exist or Qdrant is unreachable.
        """
        client = _get_client()
        try:
            records, _ = client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001
            return []

        entries: list[CacheEntry] = []
        for record in records:
            payload = record.payload or {}
            raw_ts = payload.get("created_at")
            if not raw_ts:
                continue
            try:
                created_at = datetime.fromisoformat(str(raw_ts))
            except ValueError:
                continue
            entries.append(
                CacheEntry(
                    question=str(payload.get("question", "")),
                    resolved_question=str(payload.get("resolved_question", "")),
                    agent_type=str(payload.get("agent_type", "")),
                    answer=str(payload.get("answer", "")),
                    sql=payload.get("sql") or None,
                    created_at=created_at,
                    hit_count=int(payload.get("hit_count", 0)),
                )
            )

        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def stats(self) -> dict[str, Any]:
        """Return basic statistics for this agent's cache collection.

        Returns:
            ``{"total_entries": N, "collection": str}``
        """
        client = _get_client()
        try:
            info = client.get_collection(self._collection)
            total: int = info.points_count or 0
        except Exception:  # noqa: BLE001
            total = 0
        return {"total_entries": total, "collection": self._collection}
