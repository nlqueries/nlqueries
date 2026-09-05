"""
tests.test_cache_prune
~~~~~~~~~~~~~~~~~~~~~~
The sweep's control flow: when it runs, when it does not, and what happens when
it fails. No Docker needed.

That the filter deletes the right points is a question about Qdrant, not about
this code, and is settled against a real one in
`tests/integration/test_cache_prune_integration.py`. These are the parts a fake
client can answer honestly.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.cache.envelope import CacheBinding
from nlqueries.cache.semantic_cache import (
    SemanticCache,
    _last_prune_at,
    _prune_expired,
)

TEST_KEY = b"cache-prune-test-key"

TEST_BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)


@pytest.fixture(autouse=True)
def _use_the_test_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: TEST_KEY)


@pytest.fixture(autouse=True)
def _forget_when_each_collection_was_swept() -> None:
    _last_prune_at.clear()


class _Result:
    resolved_question = "how many orders"
    agent_type = "sql"
    answer = "There were 42 orders."
    sql = "SELECT count(*) FROM orders WHERE order_date >= '2024-06-01'"


def _client() -> MagicMock:
    client = MagicMock()
    coll = MagicMock()
    coll.name = "cache_agent1"
    client.get_collections.return_value.collections = [coll]
    return client


def test_the_sweep_runs_at_most_once_per_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Amortised, not paid per write.

    The sweep is a round trip to Qdrant. Running it on every `put` would put that
    on the critical path of every answer, for a collection that only needs
    clearing occasionally.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    for _ in range(5):
        _prune_expired(client, "cache_agent1", 24)

    assert client.delete.call_count == 1, (
        f"five writes swept {client.delete.call_count} times inside one interval"
    )


def test_the_interval_is_per_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """One agent's sweep must not suppress another's.

    The gate is keyed on the collection precisely so a busy agent cannot starve a
    quiet one of its only means of shrinking.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    _prune_expired(client, "cache_agent1", 24)
    _prune_expired(client, "cache_agent2", 24)

    assert client.delete.call_count == 2


def test_the_interval_elapsing_allows_another_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the gate: it delays the sweep, it does not cancel it."""
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    _prune_expired(client, "cache_agent1", 24)
    _last_prune_at["cache_agent1"] -= 7200  # two hours ago
    _prune_expired(client, "cache_agent1", 24)

    assert client.delete.call_count == 2


def test_zero_disables_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 0)
    client = _client()

    assert _prune_expired(client, "cache_agent1", 24) is False
    client.delete.assert_not_called()


def test_a_failing_sweep_does_not_fail_the_write(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A cache that grows is worse than one that does not, but not that much worse.

    The sweep is housekeeping. If Qdrant refuses the delete, the caller should
    still get the answer they asked for -- and somebody should be able to find
    out that the collection is no longer shrinking.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()
    client.delete.side_effect = RuntimeError("qdrant is unhappy")

    with caplog.at_level(logging.WARNING):
        assert _prune_expired(client, "cache_agent1", 24) is False

    assert any("sweep" in r.getMessage().lower() for r in caplog.records), (
        "the sweep failed silently, so a collection that stops shrinking is invisible"
    )


def test_put_sweeps_after_upserting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordering matters: the write must land even if the sweep then fails.

    A sweep before the upsert would let housekeeping cost the caller the entry
    that prompted it.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()
    order: list[str] = []
    client.upsert.side_effect = lambda *a, **k: order.append("upsert")
    client.delete.side_effect = lambda *a, **k: order.append("delete")

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache.ensure_collection"),
    ):
        SemanticCache("agent1", binding=TEST_BINDING).put("how many orders", _Result())

    assert order == ["upsert", "delete"], f"expected upsert then sweep, got {order}"


def test_a_failing_sweep_still_leaves_the_entry_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence of that ordering, asserted rather than implied."""
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()
    client.delete.side_effect = RuntimeError("qdrant is unhappy")

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache.ensure_collection"),
    ):
        SemanticCache("agent1", binding=TEST_BINDING).put("how many orders", _Result())

    client.upsert.assert_called_once()


def test_the_sweep_does_not_wait_for_the_delete_to_be_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wait=False`, because the sweep sits on the write path.

    The filter ranges over `created_at`, and on a collection created before this
    change that field is unindexed -- so Qdrant scans to find the matches.
    Waiting for that to be applied puts an unbounded server-side scan in front of
    the caller, on precisely the collections large enough for the sweep to matter,
    against qdrant-client's five-second default. Nothing here depends on the
    delete having landed by the time `put()` returns.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    _prune_expired(client, "cache_agent1", 24)

    assert client.delete.call_args.kwargs["wait"] is False, (
        "the sweep blocks the write until Qdrant has applied the delete"
    )


def test_the_collection_indexes_created_at_as_a_datetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyword index does not accelerate a range query over a timestamp.

    `ensure_collection`'s `payload_indexes` creates keyword indexes, so asking it
    to index `created_at` that way would look like a fix and do nothing for the
    scan the sweep provokes.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 0)
    client = _client()
    seen: dict[str, object] = {}

    def _capture(name: str, size: int, **kwargs: object) -> None:
        seen.update(kwargs)

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.semantic_cache.ensure_collection", side_effect=_capture),
    ):
        SemanticCache("agent1", binding=TEST_BINDING).put("how many orders", _Result())

    assert seen.get("datetime_indexes") == ["created_at"], (
        f"created_at is not indexed as a datetime: {seen}"
    )


def test_the_cutoff_is_the_same_ttl_the_read_path_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep must only remove what a read would already have discarded.

    A cutoff derived from anything other than the caller's own `ttl_hours` could
    delete an entry that is still servable, which is the one outcome that turns
    housekeeping into data loss.
    """
    from datetime import UTC, datetime, timedelta

    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    before = datetime.now(UTC)
    _prune_expired(client, "cache_agent1", 6)
    after = datetime.now(UTC)

    selector = client.delete.call_args.kwargs["points_selector"]
    cutoff = selector.filter.must[0].range.lt

    assert before - timedelta(hours=6) <= cutoff <= after - timedelta(hours=6), (
        f"cutoff {cutoff} does not correspond to a 6-hour TTL"
    )


def test_a_datetime_index_is_created_without_keyword_indexes() -> None:
    """`ensure_collection` must not need `payload_indexes` to honour the other.

    `PayloadSchemaType` was imported inside the keyword branch, which a caller
    passing only `datetime_indexes` never entered -- and the `suppress(Exception)`
    around index creation swallowed the resulting `NameError`. So the index was
    never created, no error surfaced, and the field was still recorded as done,
    so it was never retried either.

    This asserts the index is **created**, not that `ensure_collection` was called
    with the right argument. The earlier test asserted the latter, which is why
    it stayed green through all of that.
    """
    from nlqueries.embeddings import qdrant_store as qs

    qs._indexed_fields.clear()
    client = MagicMock()
    client.collection_exists.return_value = True

    with patch.object(qs, "_get_client", return_value=client):
        qs.ensure_collection("c", 4, datetime_indexes=["created_at"])

    assert client.create_payload_index.call_count == 1, (
        "a datetime index was silently not created when no keyword indexes were asked for"
    )
    call = client.create_payload_index.call_args
    assert call.kwargs["field_name"] == "created_at"
    assert "datetime" in str(call.kwargs["field_schema"]).lower()


def test_both_kinds_of_index_are_created_with_the_right_schema() -> None:
    """A keyword index does not accelerate a range query, so the schema matters."""
    from nlqueries.embeddings import qdrant_store as qs

    qs._indexed_fields.clear()
    client = MagicMock()
    client.collection_exists.return_value = True

    with patch.object(qs, "_get_client", return_value=client):
        qs.ensure_collection("c", 4, payload_indexes=["kind"], datetime_indexes=["created_at"])

    got = {
        c.kwargs["field_name"]: str(c.kwargs["field_schema"]).lower()
        for c in client.create_payload_index.call_args_list
    }
    assert "keyword" in got["kind"]
    assert "datetime" in got["created_at"], (
        f"created_at was indexed as {got['created_at']}, which does nothing for a DatetimeRange"
    )


def test_a_failed_index_creation_is_retried() -> None:
    """A transient failure must not disable the index for the process's life.

    `_indexed_fields` used to be stamped whether or not creation succeeded, so
    Qdrant being briefly unreachable during a process's first write left the
    sweep full-scanning exactly the collections it exists for -- with a debug
    line to show for it.

    Retrying is safe because creating an index that already exists succeeds
    rather than raising, measured on both versions this project ships.
    """
    from nlqueries.embeddings import qdrant_store as qs

    qs._indexed_fields.clear()
    client = MagicMock()
    client.collection_exists.return_value = True
    client.create_payload_index.side_effect = RuntimeError("qdrant unreachable")

    with patch.object(qs, "_get_client", return_value=client):
        qs.ensure_collection("c", 4, datetime_indexes=["created_at"])

    assert "c:created_at:datetime" not in qs._indexed_fields, (
        "a failed index creation was recorded as done, so it will never be retried"
    )

    # The next call tries again, and succeeds.
    client.create_payload_index.side_effect = None
    with patch.object(qs, "_get_client", return_value=client):
        qs.ensure_collection("c", 4, datetime_indexes=["created_at"])

    assert client.create_payload_index.call_count == 2
    assert "c:created_at:datetime" in qs._indexed_fields, (
        "a successful creation was not recorded, so every write will retry it"
    )


def test_the_first_write_in_a_process_always_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate, and the alternative is worse.

    `_last_prune_at` starts empty, so the first write in a process sweeps. Under
    the CLI, where a process answers one question and exits, that is one sweep
    per invocation rather than one per hour.

    Seeding it so the first sweep waited out an interval would mean a CLI-only
    deployment never swept at all -- no process lives long enough -- and those
    are exactly the deployments whose collections still grow. The cost is bounded
    elsewhere instead: `wait=False`, and `created_at` indexed as a datetime.
    """
    monkeypatch.setattr("nlqueries.config.CACHE_PRUNE_INTERVAL_SECONDS", 3600)
    client = _client()

    assert _prune_expired(client, "cache_agent1", 24) is True, (
        "a fresh process did not sweep, so a CLI-only deployment would never prune"
    )


def test_dropping_a_collection_forgets_its_indexes() -> None:
    """`invalidate()` deletes the collection, so its index record must go too.

    `_indexed_fields` exists so repeated `ensure_collection` calls are cheap, and
    nothing invalidated it when a collection was dropped. In a long-lived process
    that also serves the admin clear endpoint, the recreated collection then came
    back with neither the `kind` keyword index nor the `created_at` datetime
    index -- leaving the sweep to scan exactly the field it ranges over, for the
    life of that process.
    """
    from nlqueries.embeddings import qdrant_store as qs

    qs._indexed_fields.clear()
    client = MagicMock()
    client.collection_exists.return_value = True

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch.object(qs, "_get_client", return_value=client),
    ):
        cache = SemanticCache("agent1", binding=TEST_BINDING)
        cache.put("how many orders", _Result())
        assert client.create_payload_index.call_count == 2, "the fixture never indexed"

        cache.invalidate("agent1")

        client.reset_mock()
        cache.put("how many orders", _Result())

    assert client.create_payload_index.call_count == 2, (
        "the recreated collection was left without its indexes, because the "
        "process still believed they existed"
    )
