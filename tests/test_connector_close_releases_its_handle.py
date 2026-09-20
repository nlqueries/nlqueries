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
    assigned = re.compile(r"self\.(_[a-z_]+)\s*=\s*[a-z_]+[a-z_0-9.]*\.(connect|Client)\(")

    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name in {"base.py", "loader.py", "__init__.py"}:
            continue
        for attr, _call in assigned.findall(path.read_text(encoding="utf-8")):
            if attr not in known:
                offenders.append(f"{path.name}: self.{attr}")

    assert not offenders, (
        "these hold a driver handle on an attribute `DatabaseConnector.close` does not "
        f"release, so evicting one from the cache leaks it: {sorted(set(offenders))}"
    )
