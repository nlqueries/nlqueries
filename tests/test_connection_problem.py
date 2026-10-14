"""
tests.test_connection_problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``connection_problem()``: why a connection is not usable, in the driver's words.

``test_connection()`` answers only yes or no. Every built-in connector caught
the driver's error inside it and logged it, so an application showing a person
why a connection failed -- a "Test connection" button -- had nothing to show
but "check host/credentials" and a log line it could not reach.
"""

from __future__ import annotations

import pytest
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import (
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
)
from nlqueries.connectors.postgres import PostgresConnector


@pytest.mark.parametrize("name", sorted(CONNECTOR_REGISTRY))
def test_every_built_in_connector_says_why_it_cannot_connect(name: str) -> None:
    """Each overrides the default, which can say only that the test failed."""
    connector = CONNECTOR_REGISTRY[name]()  # never connected

    problem = connector.connection_problem()

    assert problem is not None
    assert problem.startswith("RuntimeError: "), problem
    assert "connect() must be called before use" in problem
    assert connector.test_connection() is False


def test_the_drivers_own_message_comes_back() -> None:
    """What a "Test connection" button needs: the database's reason."""
    connector = PostgresConnector()
    connector.connect(
        {"host": "127.0.0.1", "port": 1, "database": "x", "user": "u", "password": "p"}
    )
    try:
        problem = connector.connection_problem()
    finally:
        connector.close()

    assert problem is not None
    assert problem.startswith("OperationalError: "), problem
    assert "127.0.0.1" in problem and "port 1" in problem
    assert "\n" not in problem, "later lines, where drivers quote what they sent, leaked"


def _minimal(ok: bool) -> DatabaseConnector:
    class _Minimal(DatabaseConnector):
        def connect(self, credentials: dict) -> None:
            return None

        def test_connection(self) -> bool:
            return ok

        def extract_schema(self) -> SchemaSpec:
            return SchemaSpec(database="db", tables=[], extracted_at="2026-01-01T00:00:00Z")

        def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
            return []

        def _execute_query(self, sql: str, timeout_seconds=None, max_rows=None) -> QueryResult:
            return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)

    return _Minimal()


def test_a_connector_that_implements_only_test_connection_still_answers() -> None:
    """Third-party connectors implement the abstract surface; the new method is
    not abstract, so they keep working, saying no more than that it failed."""
    assert _minimal(ok=True).connection_problem() is None
    assert _minimal(ok=False).connection_problem() == "the connection test failed"


def test_a_permitted_view_asks_the_connector_it_wraps() -> None:
    """The per-request wrapper the loader hands out. Without its own override it
    would fall back to the default and lose the driver's reason."""
    from nlqueries.connectors.base import PermittedConnector
    from nlqueries.execution import ExecutionPolicy

    connector = PostgresConnector()  # never connected
    view = PermittedConnector(connector, ExecutionPolicy.execute_read_only())

    assert view.connection_problem() == connector.connection_problem()
    assert view.connection_problem() != "the connection test failed"
