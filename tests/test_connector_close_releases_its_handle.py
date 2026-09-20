"""
tests.test_connector_close_releases_its_handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`DatabaseConnector.close` used to dispose `self._engine` and nothing else.

That was written as "most connectors keep a SQLAlchemy engine and nothing else",
which was not true of five of them: BigQuery holds a `_client`, DuckDB, Redshift
and SQLite a `_conn`, Snowflake a `_connection`. None overrode `close`, so the
loader evicting a cached connector released nothing at all for any of them and
the server-side session survived until the object was collected -- on Redshift
and Snowflake, a real session on a real warehouse, held for as long as the
garbage collector felt like it.

The tests below are per attribute rather than per connector, because the
attribute is what `close` knows about. A connector that starts holding its handle
somewhere new is caught by `test_every_connector_holds_its_handle_where_close_looks`,
which is the one that will still be true after the others have been forgotten.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from nlqueries.connectors.base import (
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
)


class _Handle:
    """A driver handle that records being closed, and can refuse to be."""

    def __init__(self, *, raises: bool = False) -> None:
        self.closed = 0
        self._raises = raises

    def close(self) -> None:
        self.closed += 1
        if self._raises:
            raise RuntimeError("the driver refused to close")


class _Connector(DatabaseConnector):
    """The smallest thing the abstract base will let us instantiate."""

    def connect(self, credentials: dict[str, Any]) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def test_connection(self) -> bool:  # pragma: no cover - unused
        raise NotImplementedError

    def extract_schema(self) -> SchemaSpec:  # pragma: no cover - unused
        raise NotImplementedError

    def extract_query_history(
        self, days: int = 30, limit: int = 500
    ) -> list[QueryRecord]:  # pragma: no cover - unused
        raise NotImplementedError

    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.parametrize("attr", ["_conn", "_connection", "_client"])
def test_close_releases_the_handle_whatever_it_is_called(attr: str) -> None:
    connector = _Connector()
    handle = _Handle()
    setattr(connector, attr, handle)

    connector.close()

    assert handle.closed == 1, f"close() left {attr} open"


@pytest.mark.parametrize("attr", ["_conn", "_connection", "_client"])
def test_close_clears_the_handle_so_the_next_caller_is_told_plainly(attr: str) -> None:
    # A closed DBAPI handle is dead. Left in place, `_require_conn` hands it back
    # and the driver raises something opaque several frames later; cleared, the
    # connectors' own guard says "not connected", which is the true statement.
    connector = _Connector()
    setattr(connector, attr, _Handle())

    connector.close()

    assert getattr(connector, attr) is None


def test_a_driver_that_refuses_to_close_does_not_escape() -> None:
    # `_dispose` in the loader suppresses exceptions, so a raising close is
    # invisible rather than loud -- and the eviction would abandon the rest of
    # what the connector holds. Suppressed here instead, where the next handle
    # in the list still gets its turn.
    connector = _Connector()
    refuses = _Handle(raises=True)
    other = _Handle()
    connector._conn = refuses
    connector._client = other

    connector.close()

    # Held before the call, because `close` clears the attributes -- reading
    # them afterwards asks a question about None.
    assert refuses.closed == 1, "the refusing handle was asked"
    assert other.closed == 1, "and the one after it still got its turn"
    assert connector._conn is None
    assert connector._client is None


def test_close_is_safe_with_nothing_held_and_safe_twice() -> None:
    connector = _Connector()
    connector.close()

    handle = _Handle()
    connector._conn = handle
    connector.close()
    connector.close()

    assert handle.closed == 1


def test_an_engine_is_disposed_but_kept() -> None:
    """The asymmetry, asserted so it is not "tidied" into consistency.

    `engine.dispose()` returns the pool's connections and leaves the engine
    usable -- it builds a new pool on next use. Clearing it would turn a
    reusable object into a dead attribute for no gain.
    """

    class _Engine:
        def __init__(self) -> None:
            self.disposed = 0

        def dispose(self) -> None:
            self.disposed += 1

    connector = _Connector()
    engine = _Engine()
    connector._engine = engine

    connector.close()

    assert engine.disposed == 1
    assert connector._engine is engine


def test_every_connector_holds_its_handle_where_close_looks() -> None:
    """The guard that outlives the cases above.

    `close` releases named attributes, so a connector that keeps its handle
    under a different name leaks silently and nothing fails. Read from the
    source rather than by importing: several connectors import their driver at
    module scope, and the drivers are optional extras.
    """
    package = Path(DatabaseConnector.__module__.replace(".", "/")).parent
    root = Path(__file__).resolve().parents[1] / package
    known = set(DatabaseConnector._HANDLE_ATTRS) | {"_engine"}
    # Anything assigned from a call that looks like opening a connection.
    # The dotted prefix is optional: `from redshift_connector import connect`
    # then `self._session = connect(...)` is the same hazard written differently,
    # and requiring `driver.connect(` would not see it.
    assigned = re.compile(r"self\.(_[a-z_]+)\s*=\s*(?:[a-z_][a-z_0-9.]*\.)?(connect|Client)\(")

    offenders: list[str] = []
    found: set[str] = set()
    scanned = 0
    for path in sorted(root.glob("*.py")):
        if path.name in {"base.py", "loader.py", "__init__.py"}:
            continue
        scanned += 1
        for attr, _call in assigned.findall(path.read_text(encoding="utf-8")):
            found.add(attr)
            if attr not in known:
                offenders.append(f"{path.name}: self.{attr}")

    # Asserted before the offenders, because "no offenders" is also what an
    # empty scan says. `root.glob` on a directory that is not there yields
    # nothing at all -- which is what a run against an installed package rather
    # than this checkout would produce -- and a pattern that stops matching
    # reports the same clean result as a codebase with nothing wrong in it.
    assert scanned >= 5, f"the scan read {scanned} connector modules; it is not looking at them"
    assert found, "the scan matched no handle assignments at all, so it is proving nothing"

    assert not offenders, (
        "these hold a driver handle on an attribute `DatabaseConnector.close` does not "
        f"release, so evicting one from the cache leaks it: {sorted(set(offenders))}"
    )


class _SlowHandle(_Handle):
    """A handle that records whether it was closed while a query was running."""

    def __init__(self) -> None:
        super().__init__()
        self.closed_during_query = False
        self.querying = False

    def close(self) -> None:
        if self.querying:
            self.closed_during_query = True
        super().close()


def test_close_waits_for_a_caller_that_is_still_using_the_connector() -> None:
    """The regression releasing raw handles would otherwise introduce.

    The loader disposes cached entries other callers already hold -- `_cache_get`
    on a stale one, `_cache_put` on a replaced one, `invalidate_connector_cache`
    on demand. That was harmless while `close` only disposed an engine, because
    `dispose()` leaves checked-out connections alone. Closing a raw handle is
    not: it aborts the statement running on it. The ordinary case is a slow
    query, a TTL that lapses mid-flight, and the next request evicting the entry
    the first request is still reading from.
    """
    connector = _Connector()
    handle = _SlowHandle()
    connector._conn = handle

    with connector.in_use():
        handle.querying = True
        connector.close()  # the eviction, arriving mid-query
        assert handle.closed == 0, "the handle was closed under a running caller"
        assert connector._conn is handle, "the handle was taken away mid-query"
        handle.querying = False

    assert handle.closed == 1, "the deferred close never happened"
    assert handle.closed_during_query is False
    assert connector._conn is None


def test_the_last_caller_out_performs_the_deferred_close() -> None:
    # Two overlapping callers: the close must wait for both, not the first.
    connector = _Connector()
    handle = _Handle()
    connector._conn = handle

    with connector.in_use():
        with connector.in_use():
            connector.close()
            assert handle.closed == 0
        assert handle.closed == 0, "closed while one caller was still inside"

    assert handle.closed == 1


def test_a_connector_nobody_holds_closes_immediately() -> None:
    # The common case, and the control: deferral must not become "never".
    connector = _Connector()
    handle = _Handle()
    connector._conn = handle

    connector.close()

    assert handle.closed == 1


def test_a_caller_leaving_without_a_pending_close_closes_nothing() -> None:
    # Leaving `in_use` must not release a connector nobody asked to close, or
    # every finished query would drop the connection the cache just stored.
    connector = _Connector()
    handle = _Handle()
    connector._conn = handle

    with connector.in_use():
        pass

    assert handle.closed == 0
    assert connector._conn is handle


def test_a_failing_close_is_logged_rather_than_silently_swallowed(caplog) -> None:
    """The one case where the session is most likely still open on the server.

    `loader._dispose` suppresses too, so with nothing logged here the residual
    leak is invisible from both sides.
    """
    connector = _Connector()
    connector._conn = _Handle(raises=True)

    with caplog.at_level("WARNING"):
        connector.close()

    assert "_conn" in caplog.text
    assert "_Connector" in caplog.text


def test_a_query_through_the_wrapper_holds_the_connector_open() -> None:
    """The wiring everything else here depends on, and nothing else covers.

    The deferral only protects a real deployment because `PermittedConnector`
    takes `in_use()` around its delegated calls, and the loader hands out that
    wrapper exclusively. Every other test in this file calls `in_use` on a bare
    connector, so deleting the guard from `PermittedConnector._execute_query`
    leaves them all green while restoring the aborted-statement regression for
    every connector the loader returns.

    So this drives a query the way a request does and evicts the connector from
    inside it.
    """
    from nlqueries.connectors.base import PermittedConnector, QueryResult
    from nlqueries.execution import ExecutionPolicy

    handle = _Handle()
    observed: dict[str, object] = {}

    class _Inner(_Connector):
        def _execute_query(
            self,
            sql: str,
            timeout_seconds: float | None = None,
            max_rows: int | None = None,
        ) -> QueryResult:
            # The eviction, arriving while this statement is running.
            self.close()
            observed["closed_mid_query"] = handle.closed
            observed["handle_attached"] = self._conn is handle
            return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)

    inner = _Inner()
    inner._conn = handle
    wrapper = PermittedConnector(inner, ExecutionPolicy.execute_read_only())

    wrapper.execute_query("select 1")

    assert observed["closed_mid_query"] == 0, "the handle was closed under a running query"
    assert observed["handle_attached"] is True, "the handle was detached mid-query"
    assert handle.closed == 1, "the deferred close did not run when the query finished"
    assert inner._conn is None


def test_the_counter_is_created_once_and_shared_by_every_caller() -> None:
    """Repeated callers get the same lock and the same counter.

    Deterministic, and therefore weaker than it looks: it catches a `_use_state`
    that builds a fresh pair per call -- which would make every `in_use` count
    against a counter nobody reads -- but it cannot catch the interleaving that
    motivated publishing the pair atomically.

    That race was not reproducible here. A barrier over eight threads and 150
    fresh connectors failed to catch a deliberately two-step publication in five
    consecutive runs: under CPython the window between the two dict writes is too
    narrow to land in on demand. A test that cannot fail for the reason it names
    is worse than none, so it is not in this file. `setdefault` remains the right
    construction for the reason a lock is right whether or not a test can
    provoke contention -- the correctness argument is the interleaving, not the
    observation of one.
    """
    connector = _Connector()

    first_lock, first_state = connector._use_state()
    second_lock, second_state = connector._use_state()

    assert first_lock is second_lock
    assert first_state is second_state

    with connector.in_use():
        assert connector._use_state()[1]["count"] == 1, (
            "a second call built a new counter, so the count is invisible to close()"
        )
