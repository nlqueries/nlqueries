"""Tests for connector reuse (W-7).

Every query used to build a new SQLAlchemy engine — a fresh TCP connection, TLS
handshake and authentication against the customer's database — and then never
dispose it. At any real concurrency that is visible connection churn on the
customer's side, and it defeats SQLAlchemy's pooling entirely: a pool discarded
after one query has pooled nothing.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
import yaml
from nlqueries import config
from nlqueries.connectors import loader
from nlqueries.connectors.base import (
    ConnectorClosed,
    DatabaseConnector,
    PermittedConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
)
from nlqueries.execution import ExecutionPolicy


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.invalidate_connector_cache()
    yield
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path, monkeypatch):
    path = tmp_path / "connectors.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "postgres:localhost:db": {
                    "db_type": "postgres",
                    "url": "postgresql://user:secret@localhost:5432/db",
                }
            }
        )
    )
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_TTL_SECONDS", 900.0)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_MAX_ENTRIES", 32)
    return path


@pytest.fixture
def built(monkeypatch):
    """Count how many connectors get built, and hand back disposable doubles."""
    made: list[MagicMock] = []

    class _Connector:
        def __init__(self) -> None:
            self.closed = False
            made.append(self)  # type: ignore[arg-type]

        def connect(self, _cfg: dict[str, Any]) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Connector)
    return made


def test_a_second_query_reuses_the_first_connector(connectors_file, built) -> None:
    """The pooled connector is shared; the handle around it is not.

    Each call returns a fresh per-request wrapper carrying that request's
    execution permission, because a policy stored on the shared object would be
    inherited by whoever got the connector next. What must be reused — and what
    this test is actually about — is the pool underneath, since a pool rebuilt
    per query has pooled nothing.
    """
    first = loader.open_connector_for_agent("postgres:localhost:db")
    second = loader.open_connector_for_agent("postgres:localhost:db")

    assert first is not second, "the permission-bearing handle should be per request"
    assert first._inner is second._inner, "the pooled connector should be shared"
    assert len(built) == 1, f"built {len(built)} connectors for two queries"


def test_a_changed_credential_rebuilds_without_a_restart(connectors_file, built) -> None:
    """The fingerprint covers the resolved password, so rotation is self-healing
    even before anything calls invalidate explicitly."""
    loader.open_connector_for_agent("postgres:localhost:db")

    connectors_file.write_text(
        yaml.safe_dump(
            {
                "postgres:localhost:db": {
                    "db_type": "postgres",
                    "url": "postgresql://user:rotated@localhost:5432/db",
                }
            }
        )
    )
    loader.open_connector_for_agent("postgres:localhost:db")

    assert len(built) == 2, "kept using a connector built with the old credential"
    assert built[0].closed is True, "the superseded connector was not disposed"


def test_an_expired_entry_is_rebuilt_and_disposed(connectors_file, built) -> None:
    loader.open_connector_for_agent("postgres:localhost:db")

    # Backdate the entry rather than setting the TTL to 0 and trusting the clock
    # to have moved. time.monotonic() has ~15.6 ms resolution on Windows, so two
    # calls inside one tick differ by exactly 0.0 — and `elapsed > 0.0` is then
    # False, so the entry was NOT considered expired and nothing was rebuilt.
    # The test passed on Linux and failed on Windows for a reason that had
    # nothing to do with the behaviour under test, which is the least useful
    # kind of failure there is. Making the entry genuinely old states the
    # premise directly and holds on any clock.
    entry = loader._cache["postgres:localhost:db"]
    entry.created_at -= config.CONNECTOR_CACHE_TTL_SECONDS + 1

    loader.open_connector_for_agent("postgres:localhost:db")

    assert len(built) == 2
    assert built[0].closed is True


def test_eviction_disposes_rather_than_waiting_for_the_collector(
    connectors_file, built, monkeypatch
) -> None:
    """An engine released only by garbage collection is not on a schedule any
    DBA would recognise as one."""
    monkeypatch.setattr(config, "CONNECTOR_CACHE_MAX_ENTRIES", 1)

    entries = {
        f"postgres:host{i}:db": {
            "db_type": "postgres",
            "url": f"postgresql://user:secret@host{i}:5432/db",
        }
        for i in range(2)
    }
    connectors_file.write_text(yaml.safe_dump(entries))

    loader.open_connector_for_agent("postgres:host0:db")
    loader.open_connector_for_agent("postgres:host1:db")

    assert built[0].closed is True, "the evicted connector's engine was left open"
    assert built[1].closed is False


def test_invalidation_closes_and_forgets(connectors_file, built) -> None:
    loader.open_connector_for_agent("postgres:localhost:db")
    loader.invalidate_connector_cache("postgres:localhost:db")

    assert built[0].closed is True
    loader.open_connector_for_agent("postgres:localhost:db")
    assert len(built) == 2


def test_invalidating_everything_closes_everything(connectors_file, built) -> None:
    entries = {
        f"postgres:host{i}:db": {
            "db_type": "postgres",
            "url": f"postgresql://user:secret@host{i}:5432/db",
        }
        for i in range(3)
    }
    connectors_file.write_text(yaml.safe_dump(entries))
    for i in range(3):
        loader.open_connector_for_agent(f"postgres:host{i}:db")

    loader.invalidate_connector_cache()

    assert all(connector.closed for connector in built)


def test_twenty_threads_build_at_most_a_handful(connectors_file, built) -> None:
    """Callers arrive from asyncio.to_thread, so the race is real. A lock held
    across the whole build would serialise every first connection on a cold
    cache, so a few duplicates under a simultaneous cold start are accepted —
    what must not happen is one per query, forever."""
    results: list[Any] = []

    def _open() -> None:
        results.append(loader.open_connector_for_agent("postgres:localhost:db"))

    threads = [threading.Thread(target=_open) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 20 and all(r is not None for r in results)
    assert len(built) < 20, f"built {len(built)} connectors for 20 concurrent opens"
    # Everything that lost the race must have been disposed, not leaked.
    survivors = [c for c in built if not c.closed]
    assert len(survivors) == 1, f"{len(survivors)} connectors left open"


def test_disabling_the_cache_restores_the_old_behaviour(
    connectors_file, built, monkeypatch
) -> None:
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)

    first = loader.open_connector_for_agent("postgres:localhost:db")
    second = loader.open_connector_for_agent("postgres:localhost:db")

    assert first is not second
    assert len(built) == 2


def test_the_fingerprint_does_not_store_a_readable_password(connectors_file) -> None:
    cfg = yaml.safe_load(connectors_file.read_text())["postgres:localhost:db"]
    fingerprint = loader._fingerprint("postgres:localhost:db", cfg)

    assert "secret" not in fingerprint
    assert len(fingerprint) == 64


# ---------------------------------------------------------------------------
# An eviction landing on a caller that already holds a wrapper
# ---------------------------------------------------------------------------


class _RealConnector(DatabaseConnector):
    """A double that holds a raw handle, so releasing it destroys something.

    The `built` fixture's double is a plain class with no `in_use`, which is
    fine for the tests that only count builds — but the recovery path runs
    through the real guard, so this one is a `DatabaseConnector`.
    """

    instances: ClassVar[list[_RealConnector]] = []

    def __init__(self) -> None:
        self._conn = object()
        self.queries = 0
        _RealConnector.instances.append(self)

    def connect(self, credentials: dict[str, Any]) -> None:
        return None

    def test_connection(self) -> bool:
        return True

    def extract_schema(self) -> SchemaSpec:  # pragma: no cover - unused
        raise NotImplementedError

    def extract_query_history(
        self, days: int = 30, limit: int = 500
    ) -> list[QueryRecord]:  # pragma: no cover - unused
        raise NotImplementedError

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> QueryResult:
        self.queries += 1
        return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)


@pytest.fixture
def real_connector(monkeypatch):
    _RealConnector.instances = []
    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _RealConnector)
    return _RealConnector.instances


def test_a_caller_holding_a_wrapper_survives_an_eviction(connectors_file, real_connector) -> None:
    """The case that made releasing raw handles unsafe.

    `_cache_put` disposes connectors that callers are already holding — a cold
    start builds several for one agent and closes all but one, with every
    wrapper still live — and a caller is outside `in_use` between the moment
    `open_connector_for_agent` returns and the `asyncio.to_thread` hop into its
    first query. Before the handles were released this cost nothing, because
    closing one did nothing. With them released, that caller's next query used
    to fail; now it rebuilds.
    """
    held = loader.open_connector_for_agent(
        "postgres:localhost:db", ExecutionPolicy.execute_read_only()
    )
    assert held is not None
    assert len(real_connector) == 1

    # The eviction: a credential change, a TTL lapse, an LRU replacement. All of
    # them arrive here, at a caller that is holding a wrapper and between calls.
    loader.invalidate_connector_cache("postgres:localhost:db")

    result = held.execute_query("select 1")

    assert result.row_count == 0, "the query did not run"
    assert len(real_connector) == 2, "the wrapper did not rebuild its connector"
    assert real_connector[0]._conn is None, "the evicted connector was not released"
    assert real_connector[1].queries == 1, "the query ran against the stale connector"


def test_the_rebuild_is_pinned_to_the_connector_and_not_the_agent(
    connectors_file, real_connector, monkeypatch
) -> None:
    """A caller must not be moved to a different database mid-request.

    Changing which connector an agent points at is a feature, so re-resolving
    from the agent would let that change land on a caller who obtained the
    wrapper under the old binding. The rebuild takes the connector id the
    wrapper was opened for and nothing else.
    """
    held = loader.open_connector_for_agent(
        "postgres:localhost:db", ExecutionPolicy.execute_read_only()
    )
    assert held is not None

    # The agent's entry now names a different connector entirely.
    connectors_file.write_text(
        yaml.safe_dump(
            {
                "postgres:localhost:db": {
                    "db_type": "postgres",
                    "url": "postgresql://user:secret@localhost:5432/db",
                },
                "postgres:elsewhere:other": {
                    "db_type": "postgres",
                    "url": "postgresql://user:secret@elsewhere:5432/other",
                },
            }
        )
    )
    loader.invalidate_connector_cache("postgres:localhost:db")

    held.execute_query("select 1")

    # Rebuilt under its own id, so the entry it used is the one it started with.
    assert held._connector_id == "postgres:localhost:db"


def test_a_connector_that_cannot_be_rebuilt_fails_closed(connectors_file, real_connector) -> None:
    """Never a silent fall back to the released connector.

    Falling back would mean querying with a credential the cache has already
    retired, which is precisely what `invalidate_connector_cache` exists to
    prevent — a worse outcome than the failed query.
    """
    held = loader.open_connector_for_agent(
        "postgres:localhost:db", ExecutionPolicy.execute_read_only()
    )
    assert held is not None

    # The entry is gone, so there is nothing to rebuild from.
    connectors_file.write_text(yaml.safe_dump({}))
    loader.invalidate_connector_cache("postgres:localhost:db")

    with pytest.raises(ConnectorClosed):
        held.execute_query("select 1")


def test_a_wrapper_from_the_cache_can_rebuild_too(connectors_file, real_connector) -> None:
    """The common path in production, and the one the first test misses.

    A first open builds and caches; every open after it returns a wrapper around
    the cached connector by a different line. A wrapper handed out there without
    its connector id cannot rebuild, so the recovery would work only for whoever
    happened to open first — which on a warm cache is nobody.
    """
    first = loader.open_connector_for_agent(
        "postgres:localhost:db", ExecutionPolicy.execute_read_only()
    )
    second = loader.open_connector_for_agent(
        "postgres:localhost:db", ExecutionPolicy.execute_read_only()
    )
    assert first is not None and second is not None
    assert len(real_connector) == 1, "the second open should have hit the cache"

    loader.invalidate_connector_cache("postgres:localhost:db")

    second.execute_query("select 1")

    assert len(real_connector) == 2, "the cached wrapper could not rebuild"
    assert real_connector[1].queries == 1


def test_the_operations_own_error_is_not_mistaken_for_an_eviction(
    connectors_file, real_connector
) -> None:
    """The recovery must not swallow a `ConnectorClosed` it did not cause.

    `_active` yields inside its own `try`, so an error thrown in by the caller's
    body lands in the same handler as one from entering the guard. Answering
    that with a second `yield` raises `generator didn't stop after throw()` and
    the real error is gone. Nothing raises it from a body today; this pins the
    scoping so nothing has to.
    """

    class _RaisesFromTheBody(_RealConnector):
        def test_connection(self) -> bool:
            raise ConnectorClosed("from the operation, not from the guard")

    # `connectors_file` is load-bearing, not scenery: it makes the rebuild
    # succeed. With no file `_reopen` returns None and the wrong spelling
    # re-raises too, so the test passes over the bug it exists for.
    inner = _RaisesFromTheBody()
    wrapper = PermittedConnector(
        inner, ExecutionPolicy.execute_read_only(), connector_id="postgres:localhost:db"
    )

    with pytest.raises(ConnectorClosed, match="from the operation"):
        wrapper.test_connection()

    assert len(real_connector) == 1, "no rebuild: the guard was never refused"
