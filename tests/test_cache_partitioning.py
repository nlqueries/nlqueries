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
from nlqueries.cache.envelope import CacheBinding, sign, verify
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


def test_a_foreign_tier1_hit_does_not_block_tier2() -> None:
    """A mismatch at one tier is not a verdict on the next.

    Tier 1's top-scoring neighbour belonging to another context says nothing
    about whether a template exists for this one, so the mismatch falls through
    rather than ending the lookup. Written because the alternative -- returning
    `None` on a foreign Tier 1 hit -- is the more obvious code and would make one
    caller's cached answer able to suppress another caller's template hit, which
    is a quiet denial rather than a leak but is still not what the partition
    means.
    """
    client = _client()
    foreign = MagicMock()
    foreign.score = 0.99
    foreign.payload = _entry(CONTEXT_B)
    foreign.id = 1

    ours = MagicMock()
    ours.score = 0.95
    ours.payload = _entry(CONTEXT_A, kind="template")
    ours.id = 99

    client.query_points.side_effect = [
        MagicMock(points=[foreign]),  # Tier 1: another context's entry on top
        MagicMock(points=[ours]),  # Tier 2: ours
    ]

    entry = _get(client, CONTEXT_A)
    assert entry is not None, "a foreign Tier 1 neighbour suppressed our own Tier 2 template"
    assert client.query_points.call_count == 2, "Tier 2 was never reached"


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
    """`_context_of` on hand-built payloads: the shape of the rule."""
    assert _context_of(_entry(None)) == {}
    assert _context_of(_entry(CONTEXT_A)) == {"tenant": "a"}
    assert _context_of(_entry({"tenant": "a", "user": "u1"})) == {
        "tenant": "a",
        "user": "u1",
    }


@pytest.mark.parametrize("with_context", [False, True], ids=["unscoped", "scoped"])
def test_reserved_keys_covers_everything_put_actually_writes(with_context: bool) -> None:
    """Asserted against `put()`'s own output, not against this file's copy of it.

    There is no field recording which keys were context; they are whatever is
    left over after the keys the cache writes itself. That is what lets entries
    written before this check existed still be read correctly instead of all
    missing for one TTL, and the cost is that `_RESERVED_PAYLOAD_KEYS` has to
    stay complete: a key added to a stored payload and not to that set starts
    counting as caller context, and every entry carrying it misses.

    An earlier version of this checked `_context_of` against `_entry()` above --
    this file's hand-written copy of the payload -- which could not detect that
    at all. Whoever adds a key to `put()` and forgets `_RESERVED_PAYLOAD_KEYS`
    has no reason to have added it here either, so the test would have stayed
    green while the hit rate quietly dropped. `docs/architecture.md` promises
    this is a red test; it has to read the real payload to be one.

    Both points are checked. The template payload is built separately in `put()`
    and could acquire a key the answer payload does not.
    """
    from unittest.mock import patch as _patch

    class _Result:
        resolved_question = "orders after 2024-06-01"
        agent_type = "sql"
        answer = "There were 42 orders."
        sql = "SELECT * FROM orders WHERE order_date >= '2024-06-01'"

    context = dict(CONTEXT_A) if with_context else None
    client = _client()
    with (
        _patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        _patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        _patch("nlqueries.cache.semantic_cache.ensure_collection"),
    ):
        SemanticCache("agent1", binding=TEST_BINDING).put(
            QUESTION, _Result(), payload_extra=context
        )

    points = client.upsert.call_args.kwargs["points"]
    assert points, "put() upserted nothing, so this asserts nothing"
    kinds = {p.payload["kind"] for p in points}
    assert kinds == {"answer", "template"}, (
        f"expected both point kinds so the template payload is covered too, got {kinds}"
    )

    for point in points:
        recovered = _context_of(point.payload)
        assert recovered == (context or {}), (
            f"the {point.payload['kind']} payload reports context {recovered}, "
            f"expected {context or {}} -- a key put() writes is missing from "
            f"_RESERVED_PAYLOAD_KEYS, so every entry carrying it will miss"
        )


# ---------------------------------------------------------------------------
# The context is part of what is signed
# ---------------------------------------------------------------------------


def _signed(**extra: object) -> dict[str, object]:
    payload = {
        "question": QUESTION,
        "resolved_question": QUESTION,
        "agent_type": "sql",
        "answer": "There were 42 orders.",
        "sql": "SELECT 1",
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "answer",
        "hit_count": 0,
        **extra,
    }
    return sign(payload, TEST_BINDING, TEST_KEY)


def test_the_context_cannot_be_relabelled_without_the_key() -> None:
    """A partition boundary the signature does not cover is not a boundary.

    `envelope.py` says it defends against "an attacker with write access to the
    vector store but not to the key". Reaching another context did not require
    forging a tag -- only moving a valid one, by editing the context keys the
    HMAC did not cover. All three edits below verified before this was fixed.
    """
    scoped = _signed(tenant="a")
    assert verify(scoped, TEST_BINDING, TEST_KEY), "the control: it verifies as written"

    relabelled = {**scoped, "tenant": "b"}
    assert not verify(relabelled, TEST_BINDING, TEST_KEY), (
        "an entry was moved to another context by editing the payload"
    )

    stripped = {k: v for k, v in scoped.items() if k != "tenant"}
    assert not verify(stripped, TEST_BINDING, TEST_KEY), (
        "a scoped entry was made readable by every caller by deleting its context"
    )

    unscoped = _signed()
    promoted = {**unscoped, "tenant": "b"}
    assert not verify(promoted, TEST_BINDING, TEST_KEY), (
        "an unscoped entry was given a context it was not signed with"
    )


def test_an_entry_written_without_a_context_still_verifies_after_the_change() -> None:
    """Why the context is appended only when there is one.

    Covering it unconditionally would change the signed message for every entry
    ever written, so the whole cache would miss for one TTL on upgrade. Appending
    only a non-empty context leaves the message byte-identical for unscoped
    entries -- which is nearly all of them -- so only context-carrying entries
    pay a one-off miss.

    This reproduces the old message format directly rather than trusting that
    claim.
    """
    import hashlib
    import hmac
    import json

    from nlqueries.cache.envelope import (
        ENVELOPE_VERSION,
        SIGNATURE_KEY,
        SIGNED_FIELDS,
        VERSION_KEY,
    )

    def _old_tag(payload: dict[str, object]) -> str:
        fields = {name: payload.get(name) for name in SIGNED_FIELDS}
        body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        message = f"{ENVELOPE_VERSION}\n{TEST_BINDING.canonical()}\n{body}".encode()
        return hmac.new(TEST_KEY, message, hashlib.sha256).hexdigest()

    written_before: dict[str, object] = {
        "question": QUESTION,
        "resolved_question": QUESTION,
        "agent_type": "sql",
        "answer": "There were 42 orders.",
        "sql": "SELECT 1",
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "answer",
        "hit_count": 0,
    }
    written_before[SIGNATURE_KEY] = _old_tag(written_before)
    written_before[VERSION_KEY] = ENVELOPE_VERSION

    assert verify(written_before, TEST_BINDING, TEST_KEY), (
        "an entry written before this change stopped verifying, so every cache "
        "in existence would go cold for a TTL on upgrade"
    )


def test_the_two_reserved_lists_agree() -> None:
    """`envelope` keeps its own copy to avoid an import cycle, so pin them.

    If they drift, a key one module treats as context the other treats as
    reserved: entries would be signed over one view of the context and verified
    against another, and every entry carrying that key would fail to verify.
    """
    from nlqueries.cache.envelope import _RESERVED_PAYLOAD_KEYS_FOR_SIGNING
    from nlqueries.cache.semantic_cache import _RESERVED_PAYLOAD_KEYS

    assert _RESERVED_PAYLOAD_KEYS_FOR_SIGNING == _RESERVED_PAYLOAD_KEYS, (
        "semantic_cache and envelope disagree about which payload keys are the caller's context"
    )


def test_a_foreign_neighbour_does_not_shadow_our_own_entry() -> None:
    """The cost of applying the equality here instead of pushing it down.

    Qdrant's filter can require the caller's keys but not the absence of others,
    so the equality runs client-side. Asking for a single point therefore let the
    nearest neighbour decide the outcome for everyone: an entry from another
    context, ranked top, consumed the only candidate slot and the lookup fell
    through even though a same-context entry sat just below it and above the
    threshold.

    It bites hardest for context-free reads, where the pushed-down filter is
    `kind` alone and every follow-up-scoped entry competes freely for that slot.
    "Standalone turns still share with each other" is only true if this holds.
    """
    client = _client()
    foreign = MagicMock()
    foreign.score = 0.99
    foreign.payload = _entry(CONTEXT_B)
    foreign.id = 1

    ours = MagicMock()
    ours.score = 0.98  # still above CACHE_ANSWER_THRESHOLD (0.97)
    ours.payload = _entry(CONTEXT_A)
    ours.id = 2

    client.query_points.return_value = MagicMock(points=[foreign, ours])

    entry = _get(client, CONTEXT_A)
    assert entry is not None, (
        "a nearer entry from another context shadowed ours, which was above the "
        "threshold and one rank below it"
    )

    # And that we actually asked Qdrant for more than one candidate. The mock
    # above returns both points whatever `limit` says, so the scan alone passes
    # just as well against `limit=1` -- against which the real client would
    # return only the foreign entry and there would be nothing to scan.
    limit = client.query_points.call_args.kwargs["limit"]
    assert limit > 1, (
        f"asked Qdrant for {limit} point(s), so the nearest neighbour still "
        f"decides the outcome no matter how the results are scanned"
    )


def test_a_foreign_neighbour_does_not_shadow_an_unscoped_reader() -> None:
    """The same, for the caller that passes no context.

    This is the direction the CHANGELOG's claim about standalone turns rests on:
    a follow-up-scoped entry ranking above an unscoped one must not stop the
    unscoped read finding its own.
    """
    client = _client()
    scoped = MagicMock()
    scoped.score = 0.995
    scoped.payload = _entry({"context_fingerprint": "fp1"})
    scoped.id = 1

    unscoped = MagicMock()
    unscoped.score = 0.98
    unscoped.payload = _entry(None)
    unscoped.id = 2

    client.query_points.return_value = MagicMock(points=[scoped, unscoped])

    assert _get(client, None) is not None, (
        "a follow-up-scoped entry shadowed a standalone turn's own cached answer"
    )
    assert client.query_points.call_args.kwargs["limit"] > 1


def test_the_scan_stops_at_the_threshold() -> None:
    """Below-threshold candidates are not rescued by matching the context.

    Points come back ranked, so the first one under the threshold ends the scan.
    Without that, widening the candidate window would quietly lower the
    similarity bar for anyone whose context happens to sit further down.
    """
    client = _client()
    foreign = MagicMock()
    foreign.score = 0.99
    foreign.payload = _entry(CONTEXT_B)
    foreign.id = 1

    ours_but_distant = MagicMock()
    ours_but_distant.score = 0.10  # far below CACHE_ANSWER_THRESHOLD
    ours_but_distant.payload = _entry(CONTEXT_A)
    ours_but_distant.id = 2

    client.query_points.side_effect = [
        MagicMock(points=[foreign, ours_but_distant]),  # Tier 1
        MagicMock(points=[]),  # Tier 2, so the miss is Tier 1's
    ]

    assert _get(client, CONTEXT_A) is None, (
        "an entry far below the similarity threshold was served because its context matched"
    )


# ---------------------------------------------------------------------------
# Two ways the partition could be lost without a call site being at fault
# ---------------------------------------------------------------------------


def test_a_context_naming_a_reserved_key_is_refused_on_both_sides(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fail-open case, reached by an unlucky key name.

    `put()` merges `payload_extra` first, so a reserved literal following it wins
    and the key never reaches storage. The entry is then stored with **no**
    context: readable by every context-free caller, and not readable by the
    caller that asked to be scoped, for its whole life. That is precisely the
    failure this module exists to close, and nothing reported it.

    Both sides refuse, so a context that cannot be stored cannot be used to read
    either — refusing only the write would leave the reader silently matching
    everything.
    """
    import logging

    from nlqueries.cache.semantic_cache import _RESERVED_CONTEXT_KEYS_LOGGED

    _RESERVED_CONTEXT_KEYS_LOGGED.clear()

    class _Result:
        resolved_question = QUESTION
        agent_type = "sql"
        answer = "There were 42 orders."
        sql = "SELECT 1"

    client = _client()
    with (
        caplog.at_level(logging.WARNING),
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache.ensure_collection"),
    ):
        cache = SemanticCache("agent1", binding=TEST_BINDING)
        cache.put(QUESTION, _Result(), payload_extra={"kind": "x"})
        assert not client.upsert.called, (
            "an entry was stored with a context key that cannot survive the write, "
            "so it is readable by every context-free caller"
        )

        assert cache.get(QUESTION, payload_filter={"kind": "x"}) is None

    # Both sides must *say so*, and the read side is the one that needs saying.
    # A lookup with such a context misses whether or not it is refused --
    # `_context_of` strips the key from the stored payload too, so the equality
    # compares {} against the caller's filter and never matches. Refusing early
    # is what turns a permanent silent miss into something diagnosable, so the
    # warning is the assertion here, not the None.
    said = [r.getMessage() for r in caplog.records if "kind" in r.getMessage()]
    assert any(m.startswith("Cache write refused") for m in said), (
        f"the write was refused without saying why: {said}"
    )
    assert any(m.startswith("Cache lookup refused") for m in said), (
        f"the lookup missed silently rather than reporting the unusable context: {said}"
    )

    # The control: an ordinary key is unaffected.
    client = _client()
    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache.ensure_collection"),
    ):
        SemanticCache("agent1", binding=TEST_BINDING).put(
            QUESTION, _Result(), payload_extra=dict(CONTEXT_A)
        )
    assert client.upsert.called, "an ordinary context key was refused"


def test_two_contexts_do_not_share_a_point_id() -> None:
    """Otherwise each write clobbers the other and neither reads it back.

    Not hypothetical and not a future feature: `cache_context` is already used
    for follow-up turns, so a scoped follow-up write and a context-free write of
    the same normalised question collide today. The partition still holds --
    neither reads the other's entry -- but both miss indefinitely, and a
    collapsed hit rate is harder to attribute than a wrong answer.
    """
    ids = {
        "unscoped": _point_id_for_question("q"),
        "tenant-a": _point_id_for_question("q", {"tenant": "a"}),
        "tenant-b": _point_id_for_question("q", {"tenant": "b"}),
    }
    assert len(set(ids.values())) == 3, f"point ids collide across contexts: {ids}"

    # Stable, and unchanged for entries written without a context -- the same
    # reasoning as the signature: do not invalidate what is already stored.
    assert _point_id_for_question("q") == _point_id_for_question("q", None)
    assert _point_id_for_question("q") == _point_id_for_question("q", {})
    assert _point_id_for_question("q", {"tenant": "a"}) == _point_id_for_question(
        "q", {"tenant": "a"}
    )


def test_an_expired_template_does_not_end_the_tier2_scan() -> None:
    """Tier 2 validates each candidate, as Tier 1 does, instead of picking one.

    Selecting the first context-matching candidate and validating it afterwards
    meant an expired or unverifiable template returned a miss even with a usable
    one ranked just below it and above the threshold. Expired points are never
    deleted -- the TTL is applied on read -- so they go on being ranked by the
    search, which makes this the shape that accumulates rather than a rare one.
    """
    from datetime import timedelta

    stale = sign(
        {
            "question": _mask_entities(QUESTION),
            "resolved_question": _mask_entities(QUESTION),
            "agent_type": "sql",
            "answer": "Stale.",
            "sql": "SELECT * FROM orders WHERE order_date >= '[d:DATE]'",
            # Older than the default ttl_hours=24, so `_verified_entry` drops it
            # -- but only after the search has already ranked it first.
            "created_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            "hit_count": 0,
            "kind": "template",
        },
        TEST_BINDING,
        TEST_KEY,
    )

    expired = MagicMock()
    expired.score = 0.99
    expired.payload = stale
    expired.id = 1

    usable = MagicMock()
    usable.score = 0.95  # still above CACHE_TEMPLATE_THRESHOLD (0.90)
    usable.payload = _entry(None, kind="template")
    usable.id = 2

    client = _client()
    client.query_points.side_effect = [
        MagicMock(points=[]),  # Tier 1 miss, so the request reaches Tier 2
        MagicMock(points=[expired, usable]),
    ]

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        entry = SemanticCache("agent1", binding=TEST_BINDING).get(QUESTION)

    assert entry is not None, (
        "an expired template ranked first ended the scan, discarding a usable one below it"
    )
    assert entry.answer != "Stale.", "the expired template was served"


def test_tier2_verifies_a_candidate_before_parsing_its_sql() -> None:
    """Order of work inside the loop, and it is not only about speed.

    Expired templates cluster at the front of the ranking -- nothing deletes them
    until the sweep runs -- so binding and parsing before verifying spends a full
    sqlglot parse on each one before discarding it. The other half matters more:
    it means taking SQL out of a payload whose signature has not been checked and
    handing it to the parser.

    `_bind_entities` is patched to record what it is asked to parse. An entry
    that fails verification must never reach it.
    """
    from datetime import timedelta

    expired = sign(
        {
            "question": _mask_entities(QUESTION),
            "resolved_question": _mask_entities(QUESTION),
            "agent_type": "sql",
            "answer": "Stale.",
            "sql": "SELECT 'never-parsed' FROM orders WHERE d >= '[d:DATE]'",
            "created_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            "hit_count": 0,
            "kind": "template",
        },
        TEST_BINDING,
        TEST_KEY,
    )
    stale_point = MagicMock()
    stale_point.score = 0.99
    stale_point.payload = expired
    stale_point.id = 1

    client = _client()
    client.query_points.side_effect = [
        MagicMock(points=[]),  # Tier 1 miss
        MagicMock(points=[stale_point]),
    ]

    seen: list[str] = []
    real_bind = __import__(
        "nlqueries.cache.semantic_cache", fromlist=["_bind_entities"]
    )._bind_entities

    def _recording(question: str, template_sql: str, dialect: str | None = None) -> str | None:
        seen.append(template_sql)
        return real_bind(question, template_sql, dialect)

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache._bind_entities", side_effect=_recording),
    ):
        assert SemanticCache("agent1", binding=TEST_BINDING).get(QUESTION) is None

    assert seen == [], f"SQL from an unverified, expired entry was handed to the binder: {seen}"


def test_put_and_get_agree_on_the_tier0_point_id() -> None:
    """The round trip, which `_point_id_for_question` alone cannot establish.

    `put()` and `get()` derive the Tier 0 id independently, and every other case
    in this file mocks `client.retrieve` to answer whatever id it is asked for --
    so dropping `payload_filter` from the derivation in `get()` would leave all
    of them green while scoped callers silently lost Tier 0 for good. That is the
    same shape of quiet hit-rate collapse this file exists to prevent, so the two
    sides are compared directly: write with a context, capture the id actually
    upserted, and require the lookup to ask for exactly that one.
    """

    class _Result:
        resolved_question = QUESTION
        agent_type = "sql"
        answer = "There were 42 orders."
        sql = "SELECT 1"

    for context in (None, dict(CONTEXT_A)):
        write_client = _client()
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=write_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
            patch("nlqueries.cache.semantic_cache.ensure_collection"),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).put(
                QUESTION, _Result(), payload_extra=context
            )

        points = write_client.upsert.call_args.kwargs["points"]
        answer_id = next(p.id for p in points if p.payload["kind"] == "answer")

        read_client = _client()
        with (
            patch("nlqueries.cache.semantic_cache._get_client", return_value=read_client),
            patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        ):
            SemanticCache("agent1", binding=TEST_BINDING).get(QUESTION, payload_filter=context)

        asked_for = read_client.retrieve.call_args.kwargs["ids"]
        assert asked_for == [answer_id], (
            f"context {context!r}: put() stored the answer point at {answer_id} but "
            f"get() looked for {asked_for} -- Tier 0 can never hit"
        )


def test_the_mismatch_warning_fires_once_and_omits_the_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The condition persists, so an unguarded warning repeats per request.

    A planted or corrupted entry does not go away by being noticed. The other two
    warnings in the module dedupe for this reason, and this one carried the
    asker's question text into every line as well -- so the log filled with user
    input at a rate set by traffic rather than by the fault.
    """
    import logging

    from nlqueries.cache.semantic_cache import _MISMATCHED_POINTS_LOGGED

    victim = "how many refunds were issued"
    point = MagicMock()
    point.payload = _entry(None)  # signed for QUESTION, not for `victim`
    point.id = _point_id_for_question(_normalize_question(victim))
    _MISMATCHED_POINTS_LOGGED.discard(point.id)

    client = _client()
    client.retrieve.return_value = [point]

    with (
        caplog.at_level(logging.WARNING),
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        cache = SemanticCache("agent1", binding=TEST_BINDING)
        for _ in range(3):
            assert cache.get(victim) is None

    warned = [r.getMessage() for r in caplog.records if "different question" in r.getMessage()]
    assert len(warned) == 1, f"three requests produced {len(warned)} warnings"
    assert victim not in warned[0], f"the asker's question was written to the log: {warned[0]}"


def test_tier0_refuses_an_entry_stored_under_another_question() -> None:
    """Tier 0 trusts an id, and the id is not part of the signed message.

    So an entry can be *relocated* without being forged. Copying a genuine,
    correctly signed entry onto the id another question hashes to made Tier 0
    answer that other question with it -- the signature verifies, because the
    payload really is untouched, and nothing compared the question. Before this
    check, "how many refunds were issued" was answered with "There were 42
    orders."

    Pre-existing rather than introduced by the context in the id, and cheap to
    close where the id is already being trusted.
    """
    victim = "how many refunds were issued"

    stolen = _entry(None)  # genuinely signed, for QUESTION rather than `victim`
    point = MagicMock()
    point.payload = stolen
    point.id = _point_id_for_question(_normalize_question(victim))

    client = _client()
    client.retrieve.return_value = [point]

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        entry = SemanticCache("agent1", binding=TEST_BINDING).get(victim)

    assert entry is None, (
        f"Tier 0 answered {victim!r} with an entry stored for {stolen['question']!r}"
    )
