"""
tests.integration.test_cache_prune_integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The cache sweep, against a real Qdrant.

This file exists because the sweep could not be written without it. Review of
#186 asked for a prune and I declined to ship one on inference: the delete is
expressed as a `DatetimeRange` over `created_at`, and `created_at` is stored as
an ISO string on collections that carry a payload index on `kind` alone. Whether
Qdrant matches that filter correctly without a datetime index is not something
to reason about, because the failure mode is a delete that matches the wrong
points and destroys live cache entries.

So the question is settled here rather than argued. Requires Docker; skipped
when `testcontainers` is absent, and a container that will not start is a skip
rather than a failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("testcontainers")

from nlqueries.cache.semantic_cache import _last_prune_at, _prune_expired  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as qm  # noqa: E402

#: Pinned, not `latest`. This file exists to establish a version-dependent
#: property of Qdrant's filtering, and evidence is only worth what it says about
#: the versions actually run.
#:
#: v1.10 is the floor -- `query_points`, which every search here goes through,
#: arrived then -- and v1.18.2 is what both `docker-compose.yml` and enterprise
#: pin, so that is what this measures. The sweep's filter was checked on v1.9.3
#: too, but that server cannot serve the searches this project makes, so
#: evidence from it says nothing about a deployment anyone can run.
#:
#: Bump this deliberately, alongside those compose files.
QDRANT_IMAGE = "qdrant/qdrant:v1.18.2"

COLLECTION = "cache_prune_probe"
TTL_HOURS = 24


@pytest.fixture(scope="module")
def qdrant() -> object:
    try:
        from testcontainers.community.qdrant import QdrantContainer
    except ImportError:  # pragma: no cover - older testcontainers layout
        from testcontainers.qdrant import QdrantContainer  # type: ignore[no-redef]

    try:
        container = QdrantContainer(QDRANT_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker / Qdrant container unavailable: {exc}")
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(
    params=[None, qm.PayloadSchemaType.DATETIME],
    ids=["created_at-unindexed", "created_at-datetime-indexed"],
)
def client(qdrant: object, request: pytest.FixtureRequest) -> QdrantClient:
    """A cache collection in each of the two shapes that exist in the field.

    **Unindexed** is every collection created before this change: only `kind`
    was ever indexed. **Datetime-indexed** is what `ensure_collection` builds
    from now on, since `put()` passes `datetime_indexes=["created_at"]`.

    Both are parameterised because the argument for this whole file is that the
    filter's behaviour must be measured rather than inferred, and the outcome to
    rule out is a delete matching live entries. Covering only the shape this
    change moves *away* from would leave the shape every new deployment has
    unmeasured.
    """
    c = QdrantClient(
        host=qdrant.get_container_host_ip(),  # type: ignore[attr-defined]
        port=int(qdrant.get_exposed_port(6333)),  # type: ignore[attr-defined]
        check_compatibility=False,
    )
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    c.create_collection(
        COLLECTION,
        vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE),
    )
    c.create_payload_index(COLLECTION, field_name="kind", field_schema=qm.PayloadSchemaType.KEYWORD)
    if request.param is not None:
        c.create_payload_index(COLLECTION, field_name="created_at", field_schema=request.param)
    _last_prune_at.clear()
    return c


def _seed(client: QdrantClient, ages_hours: dict[int, int]) -> None:
    """Insert one point per (id -> age in hours), stamped as the cache stamps them."""
    now = datetime.now(UTC)
    client.upsert(
        COLLECTION,
        wait=True,
        points=[
            qm.PointStruct(
                id=pid,
                vector=[0.1] * 4,
                payload={
                    "kind": "answer",
                    "question": f"q{pid}",
                    "created_at": (now - timedelta(hours=age)).isoformat(),
                },
            )
            for pid, age in ages_hours.items()
        ],
    )


def _ids(client: QdrantClient) -> set[int]:
    return {p.id for p in client.scroll(COLLECTION, limit=100)[0]}


def test_the_sweep_deletes_only_what_a_read_would_have_discarded(
    client: QdrantClient,
) -> None:
    """The question that blocked this change: does the filter match correctly?

    `created_at` is an ISO string and is not indexed. If Qdrant needed a datetime
    index for `DatetimeRange`, this filter would match nothing (the cache grows
    anyway, which is survivable) or -- far worse -- match everything.

    The boundary is the interesting part: 25 hours goes, 23 hours stays, against
    a 24-hour TTL. The cutoff is the same `ttl_hours` the read path applies, so
    the sweep can only remove what a read would already have refused.
    """
    _seed(client, {1: 100, 2: 25, 3: 23, 4: 0})
    assert _ids(client) == {1, 2, 3, 4}

    _prune_expired(client, COLLECTION, TTL_HOURS, wait=True)

    assert _ids(client) == {3, 4}, (
        "the sweep did not delete exactly the points past the TTL -- if this is "
        "empty, the filter matched everything and live entries were destroyed"
    )


def test_the_sweep_is_idempotent(client: QdrantClient) -> None:
    """Running it twice removes nothing further and does not error."""
    _seed(client, {1: 100, 2: 0})
    _prune_expired(client, COLLECTION, TTL_HOURS, wait=True)
    after_first = _ids(client)

    _last_prune_at.clear()  # the interval gate is unit-tested; exercise the delete
    _prune_expired(client, COLLECTION, TTL_HOURS, wait=True)

    assert _ids(client) == after_first == {2}


def test_an_entry_exactly_at_the_boundary_is_kept(client: QdrantClient) -> None:
    """`lt`, not `lte`: an entry the read path would still serve must survive.

    A sweep that removed the boundary entry would delete something a concurrent
    read could legitimately return, which is the one thing it must never do.
    """
    _seed(client, {1: TTL_HOURS - 1})
    _prune_expired(client, COLLECTION, TTL_HOURS, wait=True)
    assert _ids(client) == {1}


def test_the_sweep_leaves_a_collection_of_only_fresh_points_alone(
    client: QdrantClient,
) -> None:
    """The common case, and the control for the deletion tests above.

    Without it, every assertion here is satisfied by a sweep that deletes
    nothing at all.
    """
    _seed(client, {1: 0, 2: 1, 3: 2})
    _prune_expired(client, COLLECTION, TTL_HOURS, wait=True)
    assert _ids(client) == {1, 2, 3}
