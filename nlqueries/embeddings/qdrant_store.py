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
    upsert into Qdrant with a signed payload of ``type`` ``"capsule"``.

``replace_capsules(collection, capsules, agent_id)``
    Make *capsules* the whole of an agent's capsules in the collection.

``search(collection, query, top_k, agent_id)``
    Embed *query*, search the collection's signed capsules, and return the
    top-k ``QueryCapsule`` objects reconstructed from the stored payload.

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

``replace_chunks(collection, source_id, chunks)``
    Make *chunks* the whole of *source_id*'s content in the collection.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from nlqueries import config
from nlqueries.at_rest import chunk_text_purpose, seal, unseal
from nlqueries.connectors.base import SchemaSpec
from nlqueries.document_connectors.base import DocumentChunk
from nlqueries.processing.parameterizer import QueryCapsule
from nlqueries.state_files import agent_storage_id
from nlqueries.telemetry import chunk_search_latency, get_tracer

logger = logging.getLogger(__name__)

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


def schema_collection(agent_id: str) -> str:
    """The collection holding *agent_id*'s table, column and capsule points.

    The one place the name is made, for the knowledge-base generator and
    ``process-history --embed``, which write it, and the question path, which
    reads it. It still uses the raw ID, unlike the document and cache
    collections (see :func:`agent_collection`).
    """
    return f"agent_{agent_id}_schema"


def agent_collection(prefix: str, agent_id: str, suffix: str) -> str:
    """The name of *agent_id*'s collection, ``{prefix}{storage name}{suffix}``.

    For writers, always that name (see
    :func:`~nlqueries.state_files.agent_storage_id`).
    """
    return f"{prefix}{agent_storage_id(agent_id)}{suffix}"


#: How long a reader's choice between a collection's name and its older name
#: is reused before Qdrant is asked again.
_RESOLVED_TTL_SECONDS = 60.0
_resolved: dict[str, tuple[float, str]] = {}


def agent_collection_to_read(prefix: str, agent_id: str, suffix: str) -> str:
    """The collection to read for *agent_id*: its name, or the one it had before.

    Document collections were named with the raw agent ID until every store
    took the agent's storage name, so an ID with a character that name
    replaces -- the ``:`` of a connector ID such as ``postgres:localhost:mydb``
    -- was written under the raw ID. When the collection under the new name
    does not exist, or holds nothing (an ingest that failed before its first
    batch landed still creates it), and the one under the raw ID does exist,
    that one is read. So an agent ingested before the upgrade keeps answering
    until it is ingested again; a successful ingest writes the new name and
    deletes the old one (:func:`retire_legacy_collection`).

    Only an ID the storage name changes needs Qdrant asked at all, and the
    answer is reused for a minute rather than costing a round trip per
    question. Qdrant out of reach gives the new name, which fails downstream
    as it would have anyway.
    """
    name = agent_collection(prefix, agent_id, suffix)
    legacy = f"{prefix}{agent_id}{suffix}"
    if legacy == name:
        return name
    now = time.monotonic()
    cached = _resolved.get(name)
    if cached is not None and now - cached[0] < _RESOLVED_TTL_SECONDS:
        return cached[1]
    try:
        client = _get_client()
        existing = {c.name for c in client.get_collections().collections}
        chosen = (
            legacy
            if legacy in existing and (name not in existing or not _holds_points(client, name))
            else name
        )
    except Exception:  # noqa: BLE001 - the caller reads `name` and reports the failure
        return name
    _resolved[name] = (now, chosen)
    return chosen


def _holds_points(client: Any, collection: str) -> bool:
    return bool(client.count(collection_name=collection, exact=False).count)


def retire_legacy_collection(prefix: str, agent_id: str, suffix: str) -> str | None:
    """Delete the collection an earlier release wrote for *agent_id* under its raw ID.

    Call it once the collection under the storage name holds the agent's data
    (or, for a deliberate removal, once that one has been emptied): from then on
    the raw-ID copy is never read, and leaving it would keep the old text in
    Qdrant. Returns the name deleted, or None when there was nothing to delete.
    Raises when Qdrant cannot be reached or refuses the deletion.
    """
    name = agent_collection(prefix, agent_id, suffix)
    legacy = f"{prefix}{agent_id}{suffix}"
    if legacy == name:
        return None
    client = _get_client()
    if legacy not in {c.name for c in client.get_collections().collections}:
        return None
    client.delete_collection(legacy)
    forget_collection_indexes(legacy)
    _resolved.pop(name, None)
    return legacy


def forget_collection_indexes(name: str) -> None:
    """Drop *name*'s entries from the payload-index cache.

    Call this whenever a collection is deleted. `_indexed_fields` records which
    indexes have been created so repeated `ensure_collection` calls are cheap,
    and it is keyed by collection -- but nothing invalidated it when the
    collection went away, so recreating one in the same process skipped every
    `create_payload_index` call as already done. The collection came back
    without its indexes and stayed that way for the life of the process.

    The key format is this module's business, which is why the clearing lives
    here rather than at the call site.
    """
    # A snapshot, not a live iteration. `_indexed_fields` is shared across
    # threads -- cache writes reach `ensure_collection` through
    # `asyncio.to_thread` -- so another agent's first write can add a key while
    # this is scanning, and iterating the set directly raises "Set changed size
    # during iteration". `set.copy()` is a single C-level operation, so the
    # snapshot itself cannot tear.
    #
    # This runs on the housekeeping path -- callers invoke it to tidy up after
    # a collection is gone, not to accomplish anything a user is waiting on --
    # so it must not raise at all.
    prefix = f"{name}:"
    for key in _indexed_fields.copy():
        if key.startswith(prefix):
            _indexed_fields.discard(key)


def ensure_collection(
    name: str,
    vector_size: int = config.EMBED_DIMENSIONS,
    *,
    payload_indexes: list[str] | None = None,
    datetime_indexes: list[str] | None = None,
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
        datetime_indexes: Optional list of payload field names to index as
                         *datetime* fields.  A keyword index does not accelerate
                         a range query over a timestamp, so a field filtered by
                         ``DatetimeRange`` belongs here rather than above.
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

    if payload_indexes or datetime_indexes:
        # Imported for both loops. It used to be imported inside the keyword
        # branch, which a caller passing only `datetime_indexes` never entered --
        # and the `suppress(Exception)` below swallowed the resulting NameError,
        # so the index was silently never created and the field was still marked
        # done, so it was never retried either.
        from qdrant_client.models import PayloadSchemaType

        wanted: list[tuple[str, str, Any]] = [
            (field, f"{name}:{field}", PayloadSchemaType.KEYWORD) for field in payload_indexes or ()
        ]
        wanted += [
            (field, f"{name}:{field}:datetime", PayloadSchemaType.DATETIME)
            for field in datetime_indexes or ()
        ]

        for field, key, schema in wanted:
            if key in _indexed_fields:
                continue
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
                # Recorded only on success, so a transient failure is retried.
                # Marking it regardless meant Qdrant being briefly unreachable
                # during a process's first write disabled that index for the life
                # of the process, leaving the sweep to full-scan exactly the
                # collections it exists for -- with a debug line to show for it.
                #
                # Safe to retry because creating an index that already exists
                # succeeds rather than raising. Measured on v1.9.3 and v1.18.2,
                # since the value of retrying rests entirely on that. (v1.9.3 is
                # no longer shipped -- it predates the query API -- but it is
                # what the measurement was taken on, so it is what is recorded.)
                _indexed_fields.add(key)
            except Exception:  # noqa: BLE001
                # Logged rather than suppressed outright: a genuine failure and a
                # programming mistake both land here, and the latter is what went
                # unseen when the import was in the wrong scope.
                logger.debug(
                    "Payload index %s on %s was not created; it will be retried.",
                    field,
                    name,
                    exc_info=True,
                )


#: The ``type`` payload of a capsule point. A schema collection also holds the
#: ``"table"`` and ``"column"`` points :func:`upsert_schema` writes.
CAPSULE_TYPE = "capsule"

#: Candidates fetched per capsule wanted, so a few unsigned points nearer the
#: question do not take the signed capsules' places.
_CAPSULE_CANDIDATE_FACTOR = 3

#: Collections whose unsigned capsules have been reported, once each.
_UNSIGNED_CAPSULES_WARNED: set[str] = set()

#: Collections whose re-pointed capsules, or whose failure to re-embed them,
#: have been reported, once each.
_REPOINTED_CAPSULES_WARNED: set[str] = set()

#: How far a capsule's re-embedded text may score below the score Qdrant gave
#: its stored vector; as ``semantic_cache._RECHECK_TOLERANCE``.
_CAPSULE_RECHECK_TOLERANCE = 0.01


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def _found_by_their_own_text(
    hits: list[tuple[dict[str, Any], float]], vector: list[float], collection: str
) -> list[dict[str, Any]] | None:
    """The capsules in *hits* whose own text is as near the query as their vector.

    The signature covers a capsule's text but not the vector it is stored
    under, so a writer to Qdrant could copy a genuine capsule onto another
    question's vector and have it ranked first for that question. Each
    capsule's text -- its intent, else its description, which is what was
    embedded -- is embedded again in one batch, and must score against the
    query no lower than its stored vector did, within a tolerance for
    quantization. Returns None when the batch cannot be embedded: none is kept.
    """
    if not hits:
        return []
    from nlqueries.embeddings.embedder import embed_batch  # noqa: PLC0415

    texts = [str(p.get("intent") or p.get("auto_description") or "") for p, _ in hits]
    try:
        own = embed_batch(texts)
    except Exception as exc:  # noqa: BLE001 - unconfirmable is left out
        if collection not in _REPOINTED_CAPSULES_WARNED:
            _REPOINTED_CAPSULES_WARNED.add(collection)
            logger.warning(
                "Could not re-embed %d query capsule(s) in %r to confirm them (%s); "
                "leaving them out of the prompt. The embedding service may be "
                "unavailable.",
                len(hits),
                collection,
                type(exc).__name__,
            )
        return None
    kept = [
        payload
        for (payload, score), mine in zip(hits, own, strict=False)
        if _cosine(vector, mine) >= score - _CAPSULE_RECHECK_TOLERANCE
    ]
    if len(kept) < len(hits) and collection not in _REPOINTED_CAPSULES_WARNED:
        _REPOINTED_CAPSULES_WARNED.add(collection)
        logger.warning(
            "Ignoring %d query capsule(s) in %r found by a vector their own text "
            "does not match; this should not happen unless the collection has been "
            "written to directly, or by a process with a different embedding model.",
            len(hits) - len(kept),
            collection,
        )
    return kept


def _capsule_payload(capsule: QueryCapsule, point_id: int, agent_id: str) -> dict[str, Any]:
    from nlqueries.cache.envelope import sign_capsule  # noqa: PLC0415

    return sign_capsule(
        {
            "type": CAPSULE_TYPE,
            "capsule_id": point_id,
            "template_sql": capsule.template_sql,
            "intent": capsule.intent,
            "auto_description": capsule.auto_description,
            "tables": capsule.tables,
            "frequency": capsule.frequency,
            "agent_id": agent_id,
        }
    )


def upsert_capsules(
    collection: str,
    capsules: list[QueryCapsule],
    agent_id: str = "",
) -> None:
    """Embed and upsert *capsules* into *collection*, signed.

    Each capsule's ``intent`` is embedded when non-empty; otherwise
    ``auto_description`` is used as the fallback text.  The payload stored
    alongside each vector is::

        {
            "type": "capsule",
            "capsule_id": <int>,
            "template_sql": <str>,
            "intent": <str>,
            "auto_description": <str>,
            "tables": <list[str]>,
            "frequency": <int>,
            "agent_id": <str>,
            "signature": <str>,
            "capsule_version": <str>,
        }

    The signature (:func:`~nlqueries.cache.envelope.sign_capsule`) is made
    with the cache's signing key, and :func:`search` returns only capsules it
    verifies. Raises :class:`~nlqueries.cache.envelope.SigningKeyError` when
    the key is unusable, before anything is written.

    Args:
        collection: Target Qdrant collection name.
        capsules:   Capsules to upsert.
        agent_id:   The agent the capsules belong to, stored in the payload.
    """
    if not capsules:
        return

    from qdrant_client.models import PointStruct

    from nlqueries.embeddings.embedder import embed_batch

    payloads = [
        _capsule_payload(capsule, _capsule_id(capsule, idx), agent_id)
        for idx, capsule in enumerate(capsules)
    ]
    texts = [c.intent if c.intent else c.auto_description for c in capsules]
    vectors = embed_batch(texts)

    client = _get_client()
    points = [
        PointStruct(id=payload["capsule_id"], vector=vectors[idx], payload=payload)
        for idx, payload in enumerate(payloads)
    ]
    client.upsert(collection_name=collection, points=points)


def replace_capsules(collection: str, capsules: list[QueryCapsule], agent_id: str) -> None:
    """Make *capsules* the whole of *agent_id*'s capsules in *collection*.

    The new capsules are stored first and the agent's other capsule points
    deleted after, as :func:`replace_chunks` does, so a failure before the
    store leaves the previous set in place. The collection's table and column
    points are untouched.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        FilterSelector,
        HasIdCondition,
        MatchValue,
    )

    ensure_collection(collection)
    upsert_capsules(collection, capsules, agent_id=agent_id)
    keep: list[Any] = [_capsule_id(c, idx) for idx, c in enumerate(capsules)]
    filter_ = Filter(
        must=[
            FieldCondition(key="type", match=MatchValue(value=CAPSULE_TYPE)),
            FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
        ],
        must_not=[HasIdCondition(has_id=keep)] if keep else None,
    )
    _get_client().delete(collection_name=collection, points_selector=FilterSelector(filter=filter_))


def search(
    collection: str,
    query: str,
    top_k: int = 5,
    *,
    vector: list[float] | None = None,
    agent_id: str | None = None,
) -> list[QueryCapsule]:
    """Embed *query*, search *collection*'s capsules, and return ``QueryCapsule`` objects.

    Only capsule points are searched (``type == "capsule"``), and with
    *agent_id* only that agent's. Only capsules whose signature verifies are
    returned: what this returns is shown to the model as example queries, and
    Qdrant can be written by more than this process. Unsigned ones -- written
    before capsules were signed, or by something else -- are skipped, and
    reported once per collection. So is a signed capsule whose own text does
    not score as its stored vector did (:func:`_found_by_their_own_text`),
    since the signature does not cover the vector a capsule is ranked by.

    Args:
        collection: Qdrant collection to search.
        query:      Natural-language query string to embed (ignored when
                    *vector* is supplied).
        top_k:      Maximum number of results to return.
        vector:     Pre-computed embedding vector.  When provided, *query* is
                    not embedded again, saving one embed_text() call per request.
        agent_id:   Return only capsules stored for this agent.

    Returns:
        ``list[QueryCapsule]`` of length <= *top_k*, ordered by relevance.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from nlqueries.cache.envelope import verify_capsule  # noqa: PLC0415

    if vector is None:
        from nlqueries.embeddings.embedder import embed_text

        vector = embed_text(query)
    must: list[Any] = [FieldCondition(key="type", match=MatchValue(value=CAPSULE_TYPE))]
    if agent_id is not None:
        must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
    client = _get_client()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=Filter(must=must),
        limit=top_k * _CAPSULE_CANDIDATE_FACTOR,
    )

    hits = [(hit.payload, float(hit.score)) for hit in response.points if hit.payload]
    payloads = [p for p, _ in hits]
    # The type and agent are checked again after the signature: the filter
    # above ran on fields not yet verified.
    signed_hits = [
        (p, score)
        for p, score in hits
        if verify_capsule(p)
        and p.get("type") == CAPSULE_TYPE
        and (agent_id is None or p.get("agent_id") == agent_id)
    ]
    if len(signed_hits) < len(payloads) and collection not in _UNSIGNED_CAPSULES_WARNED:
        _UNSIGNED_CAPSULES_WARNED.add(collection)
        logger.warning(
            "Ignoring %d query capsule(s) in %r that are not signed with this "
            "deployment's key. Capsules embedded before signing was added are "
            "among them: run `nlqueries process-history --embed` again.",
            len(payloads) - len(signed_hits),
            collection,
        )
    # Confirmed top_k at a time, nearest first, and only as far as needed: one
    # batch of top_k texts on an ordinary question, not every candidate, and a
    # capsule left out has its place taken by the next.
    signed: list[dict[str, Any]] = []
    step = max(1, top_k)
    for start in range(0, len(signed_hits), step):
        confirmed = _found_by_their_own_text(signed_hits[start : start + step], vector, collection)
        if confirmed is None:
            break  # the embedder failed; asking again for the next batch would too
        signed.extend(confirmed)
        if len(signed) >= top_k:
            break
    return [
        QueryCapsule(
            template_sql=str(p.get("template_sql", "")),
            placeholders=[],
            tables=list(p.get("tables", [])),
            columns=[],
            frequency=int(p.get("frequency", 0)),
            auto_description=str(p.get("auto_description", "")),
            intent=str(p.get("intent", "")),
        )
        for p in signed[:top_k]
    ]


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

DOCUMENT_VECTOR_SIZE = config.EMBED_DIMENSIONS  # the embedding model's width


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
                    # Embedded above from the plaintext; stored sealed when
                    # NLQ_CACHE_ENCRYPTION is on.
                    "text": seal(chunk.text, chunk_text_purpose(chunk.source_id)),
                    "metadata": chunk.metadata,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
        client.upsert(collection_name=collection, points=points)


def chunk_from_payload(payload: dict[str, Any]) -> DocumentChunk | None:
    """The ``DocumentChunk`` a point's payload stores, or ``None`` if its text cannot be read.

    Text sealed under a key this process does not have is skipped rather than
    returned as ciphertext, which would otherwise reach the model as content.
    """
    text = unseal(
        str(payload.get("text", "")), chunk_text_purpose(str(payload.get("source_id", "")))
    )
    if text is None:
        return None
    return DocumentChunk(
        chunk_id=str(payload.get("chunk_id", "")),
        source_id=str(payload.get("source_id", "")),
        source_name=str(payload.get("source_name", "")),
        page_number=payload.get("page_number"),
        chunk_index=int(payload.get("chunk_index", 0)),
        text=text,
        metadata=dict(payload.get("metadata") or {}),
    )


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
            chunk = chunk_from_payload(hit.payload or {})
            if chunk is not None:
                result.append(chunk)

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


#: Points written per request by :func:`replace_chunks`. One request for a
#: whole source passed Qdrant's default 32 MB request limit at a few thousand
#: chunks (each point carries a 384-float vector and up to 800 characters).
_REPLACE_BATCH_SIZE = 256


class StaleChunksRemain(RuntimeError):
    """A replace stopped part-way, leaving new and old chunks side by side.

    Raised by :func:`replace_chunks` when a write fails after at least one
    batch of new chunks was stored, or when the delete after a complete write
    fails. The collection then holds some or all of the new chunks *and* the
    old version's remainder, so a search can return text from both until a
    later replace succeeds.
    """


def replace_chunks(collection: str, source_id: str, chunks: list[DocumentChunk]) -> None:
    """Make *chunks* the whole of *source_id*'s content in *collection*.

    The new chunks are stored first and the source's other chunks deleted
    after, so a failure before the store -- the embedder or Qdrant unavailable
    -- leaves the previous version untouched rather than nothing. A failure of
    the delete after a successful store raises :class:`StaleChunksRemain`: the
    new chunks are in place but so is the old version's tail. Chunk IDs are
    derived from the source and the chunk's position, so a shorter new version
    overwrites the start of the old one and the delete removes its tail. With
    no chunks every chunk of the source is removed: the document is now empty.

    Args:
        collection: Qdrant collection name (convention: ``doc_{source_id}_chunks``).
        source_id:  The source whose content *chunks* is.
        chunks:     The source's complete set of chunks.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        FilterSelector,
        HasIdCondition,
        MatchValue,
    )

    ensure_collection(collection, DOCUMENT_VECTOR_SIZE)
    for start in range(0, len(chunks), _REPLACE_BATCH_SIZE):
        try:
            upsert_chunks(collection, chunks[start : start + _REPLACE_BATCH_SIZE])
        except Exception as exc:
            if start == 0:
                raise  # nothing written yet: the earlier version is untouched
            raise StaleChunksRemain(
                f"{start} of {len(chunks)} new chunks were stored before the write "
                f"failed ({exc}), and the earlier version was not removed"
            ) from exc
    # One request whatever the size: a keep-list of a few thousand integer IDs
    # is tens of kilobytes, and splitting it across deletes would remove the
    # chunks each slice leaves out.
    filter_ = Filter(
        must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))],
        must_not=[HasIdCondition(has_id=[int(chunk.chunk_id, 16) for chunk in chunks])]
        if chunks
        else None,
    )
    try:
        _get_client().delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=filter_),
        )
    except Exception as exc:
        raise StaleChunksRemain(
            f"the new chunks were stored, but the earlier version's leftover chunks "
            f"could not be removed ({exc})"
        ) from exc
