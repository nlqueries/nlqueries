"""
tests.test_cache_partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The invariant that makes the semantic cache safe to share, guarded so that a
future feature cannot quietly break it.

**The invariant.** A cache collection is per agent, and every entry in it is
readable by everyone who may query that agent. That is sound today because
authorisation is granted at agent level: row filters are a property of the agent
record rather than of the caller, and cached SQL replays through the same
filtered connector. So "any user of agent X may see agent X's cached answers" is
a restatement of the permission model, not a hole in it.

It stops being sound the moment anything narrows what a caller may see *below*
the agent -- per-user row filters, per-user document ACLs, RLS keyed on caller
identity. Such a thing must put its distinguishing value into `cache_context` on
**both** `get()` and `put()`, or it must not be built.

`tests/`, not `tests/security/`: that directory's conftest calls
`pytest.importorskip("testcontainers.postgres")` at import time, so the whole
directory is skipped without that extra. A guard against a mistake nobody has
made yet is worth little if it only runs where Docker is installed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.cache.envelope import CacheBinding, sign
from nlqueries.cache.semantic_cache import (
    SemanticCache,
    _context_of,
    _mask_entities,
    _normalize_question,
    _payload_matches,
    _point_id_for_question,
)

TEST_KEY = b"cache-partitioning-test-key"

TEST_BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)

#: One tenant's context, another's, and the caller that forgot to pass one.
CONTEXT_A: dict[str, str] = {"tenant": "a"}
CONTEXT_B: dict[str, str] = {"tenant": "b"}

QUESTION = "orders after 2024-06-01"


@pytest.fixture(autouse=True)
def _use_the_test_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: TEST_KEY)


@pytest.fixture(autouse=True)
def _forget_cached_collection_state() -> None:
    """`_collection_exists` memoises both answers for the life of the process."""
    from nlqueries.cache import semantic_cache as sc

    sc._known_collections.discard("cache_agent1")
    sc._missing_collections.pop("cache_agent1", None)


def _entry(context: dict[str, str] | None, *, kind: str = "answer") -> dict[str, Any]:
    """A stored entry written under *context*, signed for `TEST_BINDING`."""
    question = _mask_entities(QUESTION) if kind == "template" else QUESTION
    sql = "SELECT * FROM orders WHERE d >= '[d:DATE]'" if kind == "template" else "SELECT 1"
    return sign(
        {
            **(context or {}),
            "question": question,
            "resolved_question": question,
            "agent_type": "sql",
            "answer": "There were 42 orders.",
            "sql": sql,
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": kind,
        },
        TEST_BINDING,
        TEST_KEY,
    )


def _client() -> MagicMock:
    client = MagicMock()
    coll = MagicMock()
    coll.name = "cache_agent1"
    client.get_collections.return_value.collections = [coll]
    client.retrieve.return_value = []
    client.query_points.return_value = MagicMock(points=[])
    return client


def _get(client: MagicMock, context: dict[str, str] | None) -> Any:
    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        return SemanticCache("agent1", binding=TEST_BINDING).get(QUESTION, payload_filter=context)


def _tier0_client(context: dict[str, str] | None) -> MagicMock:
    client = _client()
    point = MagicMock()
    point.payload = _entry(context)
    point.id = _point_id_for_question(_normalize_question(QUESTION))
    client.retrieve.return_value = [point]
    return client


def _tier1_client(context: dict[str, str] | None) -> MagicMock:
    client = _client()
    hit = MagicMock()
    hit.score = 0.99
    hit.payload = _entry(context)
    hit.id = 1
    client.query_points.return_value = MagicMock(points=[hit])
    return client


def _tier2_client(context: dict[str, str] | None) -> MagicMock:
    client = _client()
    tmpl = MagicMock()
    tmpl.score = 0.95
    tmpl.payload = _entry(context, kind="template")
    tmpl.id = 99
    client.query_points.side_effect = [
        MagicMock(points=[]),  # Tier 1 miss, so the request reaches Tier 2
        MagicMock(points=[tmpl]),
    ]
    return client


TIERS = [
    pytest.param(_tier0_client, id="tier0-exact"),
    pytest.param(_tier1_client, id="tier1-cosine"),
    pytest.param(_tier2_client, id="tier2-template"),
]


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_client", TIERS)
def test_a_context_scoped_entry_is_a_hit_for_the_same_context(make_client: Any) -> None:
    """The control, and it comes first deliberately.

    Every assertion below is that something is *not* served. Each would pass
    against a cache that served nothing at all, or against a harness that never
    reached the tier it names -- which has happened here before. This pins that
    the entry really is reachable in the matching context, so the misses that
    follow mean what they say.
    """
    entry = _get(make_client(CONTEXT_A), CONTEXT_A)
    assert entry is not None, "the entry was not reachable even in its own context"


@pytest.mark.parametrize("make_client", TIERS)
def test_a_context_scoped_entry_is_a_miss_for_another_context(make_client: Any) -> None:
    """One tenant's cached answer must not be served to another's question."""
    assert _get(make_client(CONTEXT_A), CONTEXT_B) is None


@pytest.mark.parametrize("make_client", TIERS)
def test_a_context_scoped_entry_is_a_miss_for_no_context(make_client: Any) -> None:
    """The direction that used to succeed, which is the one that mattered.

    `_payload_matches` asked whether the payload *contained* the caller's keys.
    A caller passing no context therefore matched an entry written under any
    context, while the reverse correctly missed. Since the value of
    `cache_context` rests on it being supplied on both `get()` and `put()`, the
    failure mode being guarded is exactly a caller that forgets it -- and that
    was the case that silently returned someone else's entry.
    """
    assert _get(make_client(CONTEXT_A), None) is None


@pytest.mark.parametrize("make_client", TIERS)
def test_an_unscoped_entry_is_a_miss_for_a_scoped_caller(make_client: Any) -> None:
    """The other direction, which already held. Kept so it cannot regress."""
    assert _get(make_client(None), CONTEXT_A) is None


@pytest.mark.parametrize("make_client", TIERS)
def test_an_unscoped_entry_is_a_hit_for_an_unscoped_caller(make_client: Any) -> None:
    """Standalone turns still share with each other, which is intended.

    Without this the invariant could be satisfied by never serving anything.
    """
    assert _get(make_client(None), None) is not None


def test_a_partial_context_match_is_not_enough() -> None:
    """Two keys stored, one supplied: a subset is not the same context.

    The failure this guards is a future caller that adds a second scoping
    dimension to `put()` and forgets it in one of the `get()` call sites.
    """
    payload = _entry({"tenant": "a", "user": "u1"})
    assert _payload_matches(payload, {"tenant": "a", "user": "u1"})
    assert not _payload_matches(payload, {"tenant": "a"})
    assert not _payload_matches(payload, {"user": "u1"})


def test_the_context_is_recovered_from_the_payload_not_a_marker_field() -> None:
    """How the context is known, stated so the reserved list stays honest.

    There is no field recording which keys were context. They are whatever is
    left after the keys the cache writes itself, which is what lets entries
    written before this check existed be read correctly rather than all missing
    for one TTL. If a new reserved key is ever added to a stored payload and not
    to `_RESERVED_PAYLOAD_KEYS`, it starts counting as context and every entry
    carrying it misses -- this is the test that says so.
    """
    assert _context_of(_entry(None)) == {}
    assert _context_of(_entry(CONTEXT_A)) == {"tenant": "a"}
    assert _context_of(_entry({"tenant": "a", "user": "u1"})) == {
        "tenant": "a",
        "user": "u1",
    }

    # Every key the cache writes itself, for both kinds of entry.
    for kind in ("answer", "template"):
        assert _context_of(_entry(None, kind=kind)) == {}, (
            f"a {kind} entry with no context reported one, so a key the cache "
            f"writes is missing from _RESERVED_PAYLOAD_KEYS"
        )
