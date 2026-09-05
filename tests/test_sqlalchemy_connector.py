"""
tests.test_sqlalchemy_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The generic URL-driven SQLAlchemyConnector, exercised against a file-backed
SQLite database (built in — no optional driver needed), so its dialect-agnostic
reflection and execution paths are covered end to end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.sqlalchemy_connector import (
    SQLAlchemyConnector,
    _apply_statement_timeout,
)

from tests.conftest import granted


def _connect(tmp_path: Path) -> SQLAlchemyConnector:
    db = tmp_path / "t.db"
    c = granted(SQLAlchemyConnector())
    c.connect({"url": f"sqlite:///{db}"})
    return c


def _seed(c: SQLAlchemyConnector, *statements: str) -> None:
    """Set up fixture data outside the answer path.

    ``execute_query`` is the path an answer takes and never commits, so a write
    sent through it is rolled back by design. Tests that need rows on disk have
    to put them there themselves.
    """
    from sqlalchemy import text

    engine = c._engine  # noqa: SLF001
    assert engine is not None
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _mock_conn(dialect_name: str, *, is_mariadb: bool = False) -> MagicMock:
    conn = MagicMock()
    conn.engine.dialect.name = dialect_name
    conn.engine.dialect._is_mariadb = is_mariadb
    return conn


def _emitted_sql(conn: MagicMock) -> str:
    """The SQL string of the single statement _apply_statement_timeout emitted."""
    return str(conn.execute.call_args[0][0])


def test_apply_statement_timeout_postgres_uses_set_local() -> None:
    conn = _mock_conn("postgresql")
    _apply_statement_timeout(conn, 5)
    sql = _emitted_sql(conn).lower()
    assert "set local statement_timeout" in sql and "5000" in sql


def test_apply_statement_timeout_mysql_uses_max_execution_time() -> None:
    conn = _mock_conn("mysql", is_mariadb=False)
    _apply_statement_timeout(conn, 5)
    sql = _emitted_sql(conn).lower()
    assert "max_execution_time" in sql and "5000" in sql


def test_apply_statement_timeout_mariadb_uses_max_statement_time() -> None:
    conn = _mock_conn("mysql", is_mariadb=True)
    _apply_statement_timeout(conn, 5)
    assert "max_statement_time" in _emitted_sql(conn).lower()


def test_apply_statement_timeout_sqlite_is_noop() -> None:
    conn = _mock_conn("sqlite")
    _apply_statement_timeout(conn, 5)
    conn.execute.assert_not_called()


def test_execute_query_on_sqlite_ignores_timeout(tmp_path: Path) -> None:
    """The default timeout is a no-op on SQLite — the query still runs cleanly."""
    c = _connect(tmp_path)
    assert c.execute_query("CREATE TABLE t (id INTEGER)").error is None
    result = c.execute_query("SELECT 1 AS one")
    assert result.error is None
    assert result.rows == [[1]]


def test_registered_under_sqlalchemy() -> None:
    assert CONNECTOR_REGISTRY.get("sqlalchemy") is SQLAlchemyConnector


def test_connect_requires_a_url() -> None:
    with pytest.raises(ValueError, match="url"):
        SQLAlchemyConnector().connect({})


def test_test_connection_ok(tmp_path: Path) -> None:
    assert _connect(tmp_path).test_connection() is True


def test_reflects_columns_pk_and_fk(tmp_path: Path) -> None:
    c = _connect(tmp_path)
    c.execute_query("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
    c.execute_query(
        "CREATE TABLE orders ("
        "  id INTEGER PRIMARY KEY,"
        "  customer_id INTEGER REFERENCES customers(id),"
        "  total REAL"
        ")"
    )

    schema = c.extract_schema()
    assert {t.name for t in schema.tables} == {"customers", "orders"}

    orders = next(t for t in schema.tables if t.name == "orders")
    cols = {col.name: col for col in orders.columns}
    assert cols["id"].is_primary_key is True
    assert cols["customer_id"].is_foreign_key is True
    assert cols["customer_id"].references == "customers.id"
    assert cols["total"].is_primary_key is False


def test_a_write_through_the_answer_path_does_not_survive(tmp_path: Path) -> None:
    """The point of the change: DML sent as an "answer" is rolled back.

    The generic connector cannot know what its engine offers, so it cannot ask
    for a read-only transaction. What it can do is never commit. A model that
    emits `INSERT` -- or a `SELECT` calling a function that writes -- gets its
    work undone whether the statement succeeded or failed.
    """
    c = _connect(tmp_path)
    _seed(c, "CREATE TABLE t (a INTEGER)", "INSERT INTO t VALUES (1)")

    # The write reports no error: it really did run, and was really undone.
    assert c.execute_query("INSERT INTO t VALUES (99)").error is None

    from sqlalchemy import text

    engine = c._engine  # noqa: SLF001
    assert engine is not None
    with engine.connect() as conn:
        assert [r[0] for r in conn.execute(text("SELECT a FROM t ORDER BY a"))] == [1]


def test_execute_query_returns_rows_and_surfaces_errors(tmp_path: Path) -> None:
    c = _connect(tmp_path)
    _seed(c, "CREATE TABLE t (a INTEGER, b TEXT)", "INSERT INTO t VALUES (1, 'x'), (2, 'y')")

    ok = c.execute_query("SELECT a, b FROM t ORDER BY a")
    assert ok.error is None
    assert ok.columns == ["a", "b"]
    assert ok.rows == [[1, "x"], [2, "y"]]
    assert ok.row_count == 2

    bad = c.execute_query("SELECT * FROM does_not_exist")
    assert bad.error is not None
    assert bad.rows == []


def test_query_history_is_empty(tmp_path: Path) -> None:
    assert _connect(tmp_path).extract_query_history() == []
