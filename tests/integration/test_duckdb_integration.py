"""Real (non-mocked) integration tests for DuckDBConnector.

DuckDB runs in-process, so these tests exercise a real ``:memory:`` database
with no containers or external services required. They must always pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from nlqueries import config  # noqa: E402
from nlqueries.connectors.duckdb import DuckDBConnector  # noqa: E402


@pytest.fixture
def connector() -> DuckDBConnector:
    conn = DuckDBConnector()
    conn.connect({"database": ":memory:"})
    return conn


def test_connect_and_test_connection(connector: DuckDBConnector) -> None:
    assert connector.test_connection() is True


def test_execute_real_query(connector: DuckDBConnector) -> None:
    result = connector.execute_query("SELECT 42 AS answer")

    assert result.error is None
    assert result.columns == ["answer"]
    assert result.rows == [[42]]


def test_extract_schema_from_real_tables(connector: DuckDBConnector) -> None:
    connector._conn.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name VARCHAR, price DOUBLE)"
    )

    spec = connector.extract_schema()

    tables = [t for t in spec.tables if t.name == "products"]
    assert len(tables) == 1
    table = tables[0]

    column_names = {c.name for c in table.columns}
    assert column_names == {"id", "name", "price"}

    id_column = next(c for c in table.columns if c.name == "id")
    assert id_column.is_primary_key is True


def test_execute_query_with_real_data(connector: DuckDBConnector) -> None:
    connector._conn.execute("CREATE TABLE items (id INTEGER, label VARCHAR)")
    connector._conn.execute("INSERT INTO items VALUES (1, 'a'), (2, 'b'), (3, 'c')")

    result = connector.execute_query("SELECT COUNT(*) AS n FROM items")

    assert result.row_count == 1
    assert result.rows == [[3]]


def test_execute_query_bad_sql_surfaces_error(connector: DuckDBConnector) -> None:
    result = connector.execute_query("SELECT * FROM nonexistent_table_xyz")

    assert result.error is not None
    assert result.row_count == 0


def test_extract_query_history_always_empty(connector: DuckDBConnector) -> None:
    assert connector.extract_query_history() == []


def test_execute_query_watchdog_interrupts_runaway_query(connector: DuckDBConnector) -> None:
    """A tiny timeout interrupts a long recursive query via the watchdog thread,
    surfacing an error instead of hanging."""
    slow = (
        "WITH RECURSIVE t(n) AS ("
        "  SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 100000000"
        ") SELECT count(*) FROM t"
    )
    result = connector.execute_query(slow, timeout_seconds=0.3)

    assert result.error is not None


def test_execute_query_no_watchdog_when_disabled(connector: DuckDBConnector, monkeypatch) -> None:
    """A 0 default disables the watchdog — a normal query runs to completion."""
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    result = connector.execute_query("SELECT 1 AS x")

    assert result.error is None
    assert result.rows == [[1]]
