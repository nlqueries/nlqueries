"""
nlqueries.cache.semantic_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic query cache backed by Qdrant with three lookup tiers.

**Tier 0 — exact-match** (zero embed calls on hit):
    Normalize the question (lowercase, strip punctuation, collapse whitespace),
    derive a deterministic point ID via SHA-256, and call ``client.retrieve()``
    directly.  No embedding is needed.

**Tier 1 — answer cache** (cosine similarity ≥ CACHE_ANSWER_THRESHOLD):
    Embed the question once (or reuse the pre-computed *vector* kwarg) and
    run a nearest-neighbour search filtered to ``kind=answer`` points.
    This is the original Sprint 21 cache, unchanged except for the kind filter.

**Tier 2 — template cache** (cosine similarity ≥ CACHE_TEMPLATE_THRESHOLD):
    Mask entity tokens in the question (dates → ``<DATE>``, numbers →
    ``<NUMBER>`` etc.) and search against stored ``kind=template`` points.
    On a hit, re-extract the concrete entity values from the original question
    and bind them into the parameterized SQL template
    (placeholders look like ``[column_name:DATE]``).  The bound SQL is
    validated syntactically before being returned.

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
import logging
import re
import string
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from nlqueries import config
from nlqueries.cache.envelope import CacheBinding, sign, verify
from nlqueries.embeddings.embedder import embed_text
from nlqueries.embeddings.qdrant_store import ensure_collection

logger = logging.getLogger(__name__)

CACHE_COLLECTION_PREFIX = "cache_"
CACHE_VECTOR_SIZE = 384
SIMILARITY_THRESHOLD = 0.97  # kept for backward-compat; Tier 1 uses config value

# Qdrant collection names must not contain ":" or other special chars.
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")

# Module-level set of known-existing Qdrant collection names.
_known_collections: set[str] = set()

# Collections we have looked for and not found, with the time each entry stops
# being trusted.
#
# Without this, only *hits* were cached: an agent whose cache collection does not
# exist yet paid a full get_collections() round trip on every single query,
# forever, until its first cache write created the collection. New agents were
# therefore slower than warm ones for a reason nobody would guess from the code.
#
# The entries expire because a collection created by another process — the CLI,
# a worker, a second API replica — would otherwise stay invisible here. Sixty
# seconds of staleness on a cache lookup is harmless: the consequence is a cache
# miss, which is what was happening anyway.
_missing_collections: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Entity masking helpers (Tier 2 template cache)
# ---------------------------------------------------------------------------

# Regex patterns applied in order (most specific first).
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_DQUOTE_RE = re.compile(r'"[^"]*"')
_SQUOTE_RE = re.compile(r"'[^']*'")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Each entry: (pattern, token_type_string)
_ENTITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_ISO_DATE_RE, "<DATE>"),
    (_MONTH_RE, "<MONTH>"),
    (_CURRENCY_RE, "<CURRENCY>"),
    (_DQUOTE_RE, "<STRING>"),
    (_SQUOTE_RE, "<STRING>"),
    (_NUMBER_RE, "<NUMBER>"),
]

# Regex to find parameterized SQL placeholders like [column_name:DATE]
_PLACEHOLDER_RE = re.compile(r"\[([^:\]]+):([^\]]+)\]")

# Maps parameterizer placeholder types → entity type keys used by _extract_entities_by_type
_PARAM_TYPE_TO_ENTITY: dict[str, str] = {
    "INT": "NUMBER",
    "DECIMAL": "NUMBER",
    "DATE": "DATE",
    "TIMESTAMP": "DATE",
    "VARCHAR": "STRING",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_question(question: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for exact-match lookup."""
    q = question.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", q).strip()


def _mask_entities(question: str) -> str:
    """Replace entity tokens in *question* with typed placeholders.

    Patterns are applied in order (most specific first) so that ISO dates are
    captured before the generic number pattern would match the year/month/day
    digits.
    """
    masked = question
    for pattern, token in _ENTITY_PATTERNS:
        masked = pattern.sub(token, masked)
    return masked


def _extract_entities_by_type(question: str) -> dict[str, list[str]]:
    """Return entity values from *question* grouped by type token (no angle brackets).

    Each pattern is applied to *question* and the matched strings are collected.
    A running copy of *question* is masked after each pattern so later patterns
    don't double-count characters that already matched.
    """
    result: dict[str, list[str]] = {}
    remaining = question
    for pattern, token in _ENTITY_PATTERNS:
        type_key = token[1:-1]  # strip < > → "DATE", "NUMBER", etc.
        values = pattern.findall(remaining)
        if values:
            result.setdefault(type_key, []).extend(values)
        remaining = pattern.sub(token, remaining)
    return result


def _quote_sql_value(value: str, param_type: str) -> str:
    """Format *value* for inline SQL according to its *param_type*."""
    if param_type.upper() in ("INT", "DECIMAL"):
        return value
    return f"'{value}'"


def _bind_entities(question: str, template_sql: str) -> str | None:
    """Substitute question entities into *template_sql* placeholders.

    Placeholders look like ``[column_name:DATE]``.  Entities are matched using
    two strategies applied in order:

    1. **Semantic pre-assignment** (before positional fallback):

       - Placeholders whose name contains ``year`` receive the first 4-digit
         number found in the question (years are nearly always 4 digits).
       - ``param_N`` placeholders (LIMIT / OFFSET generated by the parameterizer
         when no column context is available) receive the number following
         ``top`` or ``first`` in the question (e.g. "top 10 movies" → 10).

    2. **Positional fallback**: remaining placeholders are filled left-to-right
       with remaining entities of the matching type.

    Returns ``None`` when there are fewer available entities than required
    placeholders of a given type (binding would produce an incomplete or
    incorrect query).
    """
    placeholders = _PLACEHOLDER_RE.findall(template_sql)  # [(name, type), ...]
    if not placeholders:
        return template_sql  # no placeholders — use as-is

    entities = _extract_entities_by_type(question)
    # Mutable copy of the number pool so pre-assignments can claim values.
    number_pool: list[str] = list(entities.get("NUMBER", []))

    # --- Semantic pre-assignment 1: year columns → first 4-digit number ---
    pre_assigned: dict[tuple[str, str], str] = {}
    four_digit = [n for n in number_pool if re.match(r"^\d{4}$", n)]
    for name, ptype in placeholders:
        if ptype.upper() in ("INT", "DECIMAL") and "year" in name.lower() and four_digit:
            val = four_digit.pop(0)
            pre_assigned[(name, ptype)] = val
            number_pool.remove(val)

    # --- Semantic pre-assignment 2: param_N (LIMIT) → "top N" / "first N" ---
    top_match = re.search(r"\b(?:top|first)\s+(\d+(?:\.\d+)?)\b", question, re.IGNORECASE)
    if top_match:
        top_val = top_match.group(1)
        for name, ptype in placeholders:
            if (
                re.match(r"^param_\d+$", name.lower())
                and ptype.upper() in ("INT", "DECIMAL")
                and (name, ptype) not in pre_assigned
                and top_val in number_pool
            ):
                pre_assigned[(name, ptype)] = top_val
                number_pool.remove(top_val)
                break  # one LIMIT per "top N" match

    # --- Positional fallback for remaining placeholders ---
    remaining: dict[str, list[str]] = {k: list(v) for k, v in entities.items()}
    remaining["NUMBER"] = number_pool
    type_used: dict[str, int] = {}

    bound = template_sql
    for name, ptype in placeholders:
        if (name, ptype) in pre_assigned:
            sql_value = _quote_sql_value(pre_assigned[(name, ptype)], ptype)
        else:
            entity_type = _PARAM_TYPE_TO_ENTITY.get(ptype.upper(), "STRING")
            idx = type_used.get(entity_type, 0)
            available = remaining.get(entity_type, [])
            if idx >= len(available):
                return None  # not enough entities — unsafe to bind
            value = available[idx]
            type_used[entity_type] = idx + 1
            sql_value = _quote_sql_value(value, ptype)

        bound = bound.replace(f"[{name}:{ptype}]", sql_value, 1)

    return bound


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
    kind: str = field(default="answer")


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
        # One factory, one authentication rule — see qdrant_client's docstring.
        from nlqueries.embeddings.qdrant_client import build_qdrant_client  # noqa: PLC0415

        _cache_client = build_qdrant_client()
    return _cache_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collection_exists(client: Any, collection: str) -> bool:
    """Return True if *collection* exists in Qdrant.

    Both answers are cached. The positive cache is unbounded, which is fine —
    collection names are bounded by agent count. The negative cache expires
    after ``CACHE_COLLECTION_NEGATIVE_TTL_SECONDS`` so a collection created
    elsewhere becomes visible without a restart.
    """
    if collection in _known_collections:
        return True

    expires_at = _missing_collections.get(collection)
    if expires_at is not None:
        if time.time() < expires_at:
            return False
        del _missing_collections[collection]

    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception:  # noqa: BLE001
        # Qdrant is unreachable, which is not the same as "the collection is
        # missing" — so this is not remembered. Caching it would extend an
        # outage past its end.
        return False

    _known_collections.update(existing)
    if collection in existing:
        return True

    _missing_collections[collection] = time.time() + config.CACHE_COLLECTION_NEGATIVE_TTL_SECONDS
    return False


def _point_id_for_question(question: str) -> int:
    """Derive a deterministic unsigned-64-bit Qdrant point ID from a question.

    Uses the first 16 hex characters of SHA-256(question), which gives a
    uniform distribution over 2^64 values — collision probability is
    negligible for any realistic cache size.
    """
    digest = hashlib.sha256(question.encode()).hexdigest()
    return int(digest[:16], 16)


def _payload_matches(payload: dict[str, Any], payload_filter: dict[str, str] | None) -> bool:
    """True when *payload* contains every key of *payload_filter* with an equal value.

    Used to scope a Tier 0 (exact-id) hit by the caller's ``payload_filter`` —
    Tier 1/2 push the same constraint down into the Qdrant query filter instead.
    An empty/``None`` filter matches everything (default behaviour unchanged).
    """
    if not payload_filter:
        return True
    return all(str(payload.get(k)) == str(v) for k, v in payload_filter.items())


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

    def __init__(
        self,
        agent_id: str,
        ttl_hours: int = 24,
        *,
        binding: CacheBinding | None = None,
    ) -> None:
        """*binding* is required to read or write entries, not to manage them.

        An entry carries SQL that is executed on a hit with no model in front of
        it, so it is signed for the context it was produced in and verified
        before use (SEC-09). Without a binding there is nothing to verify
        against: :meth:`get` reports a miss and :meth:`put` stores nothing.

        :meth:`stats`, :meth:`invalidate` and :meth:`list_entries` do not read
        entry contents and work without one.
        """
        safe_id = _SAFE_ID_RE.sub("_", agent_id)
        self._collection = f"{CACHE_COLLECTION_PREFIX}{safe_id}"
        self._ttl_hours = ttl_hours
        self._binding = binding

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _payload_to_entry(self, payload: dict[str, Any]) -> CacheEntry | None:
        """Deserialize a ``CacheEntry`` from *payload*, checking signature and TTL.

        Returns ``None`` when the entry does not verify, has expired, or the
        timestamp is missing or unparseable.

        Does not verify the signature. Verification belongs to the read path
        that feeds execution -- see :meth:`_verified_entry` -- so that
        :meth:`list_entries` can show an operator what is actually stored,
        including entries that do not verify.
        """
        created_at_raw = payload.get("created_at")
        if not created_at_raw:
            return None
        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except ValueError:
            return None
        if datetime.now(UTC) - created_at > timedelta(hours=self._ttl_hours):
            return None
        return CacheEntry(
            question=str(payload.get("question", "")),
            resolved_question=str(payload.get("resolved_question", "")),
            agent_type=str(payload.get("agent_type", "")),
            answer=str(payload.get("answer", "")),
            sql=payload.get("sql") or None,
            created_at=created_at,
            hit_count=int(payload.get("hit_count", 0)),
            kind=str(payload.get("kind", "answer")),
        )

    def _verified_entry(self, payload: dict[str, Any]) -> CacheEntry | None:
        """Deserialize *payload* only if it was signed for this cache's binding.

        An entry that does not verify is a miss rather than an error: entries
        written before signing existed are unsigned, and a question that misses
        is answered by generating SQL again.
        """
        if self._binding is None or not verify(payload, self._binding):
            return None
        return self._payload_to_entry(payload)

    def _increment_hit_count(self, client: Any, point_id: Any, payload: dict[str, Any]) -> int:
        """Increment ``hit_count`` in the Qdrant payload and return the new value."""
        new_count = int(payload.get("hit_count", 0)) + 1
        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=self._collection,
                payload={"hit_count": new_count},
                points=[point_id],
            )
        return new_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        question: str,
        *,
        vector: list[float] | None = None,
        payload_filter: dict[str, str] | None = None,
    ) -> CacheEntry | None:
        """Look up *question* in the cache, trying three tiers in order.

        **Tier 0** — exact-match via SHA-256 hash: zero embed calls on hit.
        **Tier 1** — cosine similarity ≥ ``config.CACHE_ANSWER_THRESHOLD`` against
        ``kind=answer`` points; uses *vector* when supplied to avoid a second
        ``embed_text()`` call.
        **Tier 2** — entity-masked cosine similarity ≥ ``config.CACHE_TEMPLATE_THRESHOLD``
        against ``kind=template`` points; entities from *question* are bound
        into the stored parameterized SQL template.

        Returns ``None`` when the collection does not exist, no tier produces a
        valid non-expired hit, or Tier 2 entity binding fails.

        Args:
            question: The user question.
            vector:   Pre-computed embedding vector.  When provided, the
                      ``embed_text()`` call for Tier 1 is skipped.
            payload_filter: Optional exact-match constraints on stored payload
                      keys (paired with :meth:`put`'s ``payload_extra``). A hit is
                      only returned when the stored point carries every key/value
                      here — every tier applies it. Enterprise uses this to scope
                      a follow-up's cache entry to one conversation context
                      (``{"context_fingerprint": ...}``); ``None`` (default)
                      leaves lookups exactly as before.
        """
        client = _get_client()

        # Lazy import to avoid a cycle: the orchestrator package eagerly imports
        # this module, so importing provenance at module top would be circular.
        from nlqueries.orchestrator.provenance import record_cache  # noqa: PLC0415

        if not _collection_exists(client, self._collection):
            return None

        from qdrant_client.models import (  # noqa: PLC0415
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        def _kind_filter(kind: str) -> Filter:
            """kind== plus every payload_filter key as an exact-match must-condition."""
            must: list[Any] = [FieldCondition(key="kind", match=MatchAny(any=[kind]))]
            if payload_filter:
                must += [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in payload_filter.items()
                ]
            return Filter(must=must)

        # --- Tier 0: exact-match hash lookup (zero embed calls on hit) ---
        normalized = _normalize_question(question)
        tier0_id = _point_id_for_question(normalized)
        with contextlib.suppress(Exception):
            tier0_hits = client.retrieve(
                collection_name=self._collection,
                ids=[tier0_id],
                with_payload=True,
            )
            if tier0_hits:
                payload = tier0_hits[0].payload or {}
                # retrieve() is id-only, so enforce payload_filter here (Tier 1/2
                # push it into the Qdrant query filter). A mismatch falls through
                # to the cosine tiers rather than returning a foreign-context hit.
                if _payload_matches(payload, payload_filter):
                    entry = self._verified_entry(payload)
                    if entry is not None:
                        entry.hit_count = self._increment_hit_count(client, tier0_id, payload)
                        record_cache(hit=True, tier="exact")  # provenance (SYL-1.1)
                        return entry

        # --- Tier 1: answer cache (cosine similarity, kind=answer filter) ---
        v = vector if vector is not None else embed_text(question)
        try:
            response = client.query_points(
                collection_name=self._collection,
                query=v,
                query_filter=_kind_filter("answer"),
                limit=1,
            )
        except Exception:  # noqa: BLE001
            return None

        if response.points:
            hit = response.points[0]
            if hit.score >= config.CACHE_ANSWER_THRESHOLD:
                payload = hit.payload or {}
                entry = self._verified_entry(payload)
                if entry is not None:
                    entry.hit_count = self._increment_hit_count(client, hit.id, payload)
                    record_cache(hit=True, similarity=float(hit.score), tier="answer")  # SYL-1.1
                    return entry

        # --- Tier 2: template cache (masked cosine similarity, kind=template) ---
        masked = _mask_entities(question)
        if masked == question:
            # No entities found — template would be identical to answer; skip.
            return None

        try:
            masked_vector = embed_text(masked)
            tmpl_response = client.query_points(
                collection_name=self._collection,
                query=masked_vector,
                query_filter=_kind_filter("template"),
                limit=1,
            )
        except Exception:  # noqa: BLE001
            return None

        if not tmpl_response.points:
            return None

        tmpl_hit = tmpl_response.points[0]
        if tmpl_hit.score < config.CACHE_TEMPLATE_THRESHOLD:
            return None

        tmpl_payload = tmpl_hit.payload or {}
        template_sql = str(tmpl_payload.get("sql") or "")
        if not template_sql:
            return None

        bound_sql = _bind_entities(question, template_sql)
        if bound_sql is None:
            return None

        # Basic syntactic validation before returning a template-filled SQL.
        with contextlib.suppress(Exception):
            import sqlglot  # noqa: PLC0415

            sqlglot.parse_one(bound_sql)

        entry = self._verified_entry(tmpl_payload)
        if entry is None:
            return None

        entry.question = question
        entry.sql = bound_sql
        entry.hit_count = 0
        entry.kind = "template"
        record_cache(hit=True, similarity=float(tmpl_hit.score), tier="template")  # SYL-1.1
        return entry

    def put(
        self,
        question: str,
        result: _PutResult,
        *,
        payload_extra: dict[str, str] | None = None,
    ) -> None:
        """Embed *question* and upsert the answer into the cache collection.

        Stores a ``kind="answer"`` point keyed on the **normalized** question so
        that minor variations (capitalisation, punctuation) map to the same entry.

        For SQL results that contain entity literals in the question, also stores
        a ``kind="template"`` point keyed on the masked question embedding, with
        the SQL parameterized via placeholder substitution.  Both points are
        upserted in a single Qdrant call.

        Args:
            question: The (resolved) question string used as the cache key.
            result:   Any object exposing ``resolved_question``, ``agent_type``,
                      ``answer``, and ``sql`` attributes.
            payload_extra: Optional extra string keys merged into every stored
                      point's payload, to be matched later by :meth:`get`'s
                      ``payload_filter``. Reserved payload keys (``question``,
                      ``sql``, ``kind``, ``created_at``, …) take precedence and
                      cannot be overwritten. ``None`` (default) stores exactly as
                      before.
        """
        from qdrant_client.models import PointStruct  # noqa: PLC0415

        ensure_collection(self._collection, CACHE_VECTOR_SIZE, payload_indexes=["kind"])
        # It exists as of now, so stop remembering that it did not — otherwise
        # the very process that just created it would keep reporting a miss for
        # up to the negative TTL.
        _missing_collections.pop(self._collection, None)
        _known_collections.add(self._collection)

        # Answer entry (Tier 0 exact-match + Tier 1 cosine)
        normalized = _normalize_question(question)
        answer_id = _point_id_for_question(normalized)
        answer_vector = embed_text(question)

        if self._binding is None:
            # Nothing to sign for, so nothing is stored. An unsigned entry would
            # not verify on read and would occupy the collection for its TTL.
            logger.debug("Cache write skipped: no binding was supplied.")
            return

        answer_payload: dict[str, Any] = {
            **(payload_extra or {}),
            "question": question,
            "resolved_question": result.resolved_question,
            "agent_type": result.agent_type,
            "answer": result.answer,
            "sql": result.sql,
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "answer",
        }

        answer_payload = sign(answer_payload, self._binding)

        points: list[PointStruct] = [
            PointStruct(id=answer_id, vector=answer_vector, payload=answer_payload)
        ]

        # Template entry (Tier 2) — only for SQL results with entity literals
        if result.agent_type == "sql" and result.sql:
            masked = _mask_entities(question)
            if masked != question:  # entities were found
                try:
                    from nlqueries.processing.parameterizer import (  # noqa: PLC0415
                        _parameterize_sql,
                    )

                    template_sql, placeholders = _parameterize_sql(
                        result.sql, {}, skip_string_literals=True
                    )
                    if placeholders:  # only store if SQL has literal parameters
                        masked_vector = embed_text(masked)
                        tmpl_id = _point_id_for_question(f"tmpl:{masked}")
                        tmpl_payload: dict[str, Any] = {
                            **(payload_extra or {}),
                            "question": masked,
                            "resolved_question": result.resolved_question,
                            "agent_type": result.agent_type,
                            "answer": result.answer,
                            "sql": template_sql,
                            "created_at": datetime.now(UTC).isoformat(),
                            "hit_count": 0,
                            "kind": "template",
                        }
                        points.append(
                            PointStruct(
                                id=tmpl_id,
                                vector=masked_vector,
                                payload=sign(tmpl_payload, self._binding),
                            )
                        )
                except Exception:  # noqa: BLE001
                    pass  # template storage is best-effort

        client = _get_client()
        client.upsert(collection_name=self._collection, points=points)

    def invalidate(self, agent_id: str) -> None:  # noqa: ARG002
        """Delete all points in the cache collection (full invalidation).

        The *agent_id* parameter is accepted for API symmetry with CLI callers
        but is not used — the collection to clear is always ``self._collection``.
        """
        _known_collections.discard(self._collection)
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
            entry = self._payload_to_entry(payload)
            if entry is not None:
                entries.append(entry)

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
