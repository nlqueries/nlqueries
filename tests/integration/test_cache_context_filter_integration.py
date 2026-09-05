"""
tests.integration.test_cache_context_filter_integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Whether pushing the cache context into the Qdrant query actually stops a
context-free read being starved.

This needs a real Qdrant. The starvation is a property of *ranking* — how many
foreign entries sit above yours in a similarity search — and a stand-in client
returns whatever the test hands it, so it cannot exhibit the thing being fixed.
Two earlier attempts to prove this with a fake, and one against a server too old
to answer the query at all, established nothing.

Requires Docker; skipped when `testcontainers` is absent, and a container that
will not start is a skip rather than a failure.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("testcontainers")

from nlqueries.cache import semantic_cache as sc  # noqa: E402
from nlqueries.cache.envelope import CacheBinding, sign  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as qm  # noqa: E402

#: v1.18.2, matching `docker-compose.yml` and enterprise. v1.10 is the floor:
#: `query_points` arrived then, and against anything older every search here
#: returns 404 — which is exactly how an earlier version of this file managed to
#: report starvation that was really a missing API.
QDRANT_IMAGE = "qdrant/qdrant:v1.18.2"

TEST_KEY = b"cache-context-filter-test-key"
COLLECTION = "cache_agent1"
QUESTION = "how many orders were there"
DIM = 384

TEST_BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)

#: The query, and an entry at cosine 0.99 to it — comfortably above
#: `CACHE_ANSWER_THRESHOLD` (0.97), so a miss cannot be blamed on similarity.
QUERY_VECTOR = [1.0] + [0.0] * (DIM - 1)
NEARBY_VECTOR = [0.99, math.sqrt(1 - 0.99**2)] + [0.0] * (DIM - 2)


@pytest.fixture(autouse=True)
def _use_the_test_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: TEST_KEY)


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


@pytest.fixture
def client(qdrant: object) -> QdrantClient:
    c = QdrantClient(
        host=qdrant.get_container_host_ip(),  # type: ignore[attr-defined]
        port=int(qdrant.get_exposed_port(6333)),  # type: ignore[attr-defined]
        check_compatibility=False,
        timeout=30,
    )
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    c.create_collection(
        COLLECTION,
        vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
    )
    c.create_payload_index(COLLECTION, field_name="kind", field_schema=qm.PayloadSchemaType.KEYWORD)
    sc._known_collections.discard(COLLECTION)
    sc._missing_collections.pop(COLLECTION, None)
    return c


def _entry(context: dict[str, str] | None, answer: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": QUESTION,
        "resolved_question": QUESTION,
        "agent_type": "sql",
        "answer": answer,
        "sql": "SELECT 1",
        "created_at": datetime.now(UTC).isoformat(),
        "hit_count": 0,
        "kind": "answer",
        sc.CONTEXT_DIGEST_KEY: sc._context_digest(context),
    }
    if context:
        payload.update(context)
    return sign(payload, TEST_BINDING, TEST_KEY)


def _get(client: QdrantClient, context: dict[str, str] | None) -> Any:
    with (
        patch.object(sc, "_get_client", return_value=client),
        patch.object(sc, "embed_text", return_value=QUERY_VECTOR),
    ):
        return sc.SemanticCache("agent1", binding=TEST_BINDING).get(
            QUESTION, payload_filter=context
        )


def _seed_crowd(client: QdrantClient, scoped: int) -> None:
    """*scoped* foreign entries ranked above one unscoped entry of our own."""
    points = [
        qm.PointStruct(
            id=i + 1,
            vector=QUERY_VECTOR,  # cosine 1.0: every one of these outranks ours
            payload=_entry({"context_fingerprint": f"fp{i}"}, f"scoped {i}"),
        )
        for i in range(scoped)
    ]
    points.append(qm.PointStruct(id=999, vector=NEARBY_VECTOR, payload=_entry(None, "OURS")))
    client.upsert(COLLECTION, points=points, wait=True)


def test_a_context_free_read_is_not_starved_by_scoped_entries(
    client: QdrantClient,
) -> None:
    """The point of the change, measured rather than argued.

    Twenty context-scoped entries all rank above ours, and the candidate window
    is five. With the context applied client-side the window fills with entries
    belonging to other conversations and the lookup falls through; with the
    digest pushed into the query, Qdrant never returns them and the window is
    spent on candidates that could actually match.

    On `main` this same arrangement returns a miss.
    """
    _seed_crowd(client, scoped=20)

    entry = _get(client, None)

    assert entry is not None, "a context-free read was starved by scoped entries ranked above it"
    assert entry.answer == "OURS"


def test_the_crowd_can_exceed_the_candidate_window_by_any_margin(
    client: QdrantClient,
) -> None:
    """Eliminated, not bounded.

    Raising `NLQ_CACHE_COSINE_CANDIDATES` moves the cliff; pushing the equality
    into the query removes it. A hundred foreign entries against a window of
    five is a margin no setting would have covered.
    """
    _seed_crowd(client, scoped=100)

    entry = _get(client, None)

    assert entry is not None, "the crowd still starves the read, so it is bounded not fixed"
    assert entry.answer == "OURS"


def test_a_scoped_read_still_finds_its_own_entry(client: QdrantClient) -> None:
    """The control. Without it, everything above passes against a broken filter
    that simply excludes every scoped entry from every lookup."""
    _seed_crowd(client, scoped=20)

    entry = _get(client, {"context_fingerprint": "fp7"})

    assert entry is not None, "a scoped caller could not find its own entry"
    assert entry.answer == "scoped 7"


def test_a_scoped_read_does_not_see_another_context(client: QdrantClient) -> None:
    """The partition still holds — this is an optimisation over it, not a
    replacement for it. `_payload_matches` remains the authority."""
    _seed_crowd(client, scoped=20)

    entry = _get(client, {"context_fingerprint": "nobody-wrote-this"})

    assert entry is None, "a caller was served an entry from a context it did not write"


def test_an_entry_written_before_the_digest_existed_is_still_readable(
    client: QdrantClient, caplog: pytest.LogCaptureFixture
) -> None:
    """No cold start: legacy entries carry no digest key at all.

    A context-free read has to match both those and entries written since, which
    is why that side of the filter is a disjunction rather than an equality. Get
    it wrong and every entry in every existing deployment becomes unreadable at
    once, which is the kind of thing to measure rather than reason about.
    """
    legacy = {
        "question": QUESTION,
        "resolved_question": QUESTION,
        "agent_type": "sql",
        "answer": "WRITTEN BEFORE",
        "sql": "SELECT 1",
        "created_at": datetime.now(UTC).isoformat(),
        "hit_count": 0,
        "kind": "answer",
    }
    assert sc.CONTEXT_DIGEST_KEY not in legacy
    client.upsert(
        COLLECTION,
        points=[
            qm.PointStruct(id=1, vector=QUERY_VECTOR, payload=sign(legacy, TEST_BINDING, TEST_KEY))
        ],
        wait=True,
    )

    with caplog.at_level(logging.WARNING):
        entry = _get(client, None)

    assert entry is not None, (
        "an entry written before the digest key existed became unreadable, so "
        "upgrading would empty every cache in the field"
    )
    assert entry.answer == "WRITTEN BEFORE"
