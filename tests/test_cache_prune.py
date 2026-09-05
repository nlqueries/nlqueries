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

    assert _prune_expired(client, "cache_agent1", 24) is None
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
        assert _prune_expired(client, "cache_agent1", 24) is None

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
