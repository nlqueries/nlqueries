"""
tests.test_sqlalchemy_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The generic URL-driven SQLAlchemyConnector, exercised against a file-backed
SQLite database (built in — no optional driver needed), so its dialect-agnostic
reflection and execution paths are covered end to end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors import sqlalchemy_connector as sqlalchemy_connector_module
from nlqueries.connectors.sqlalchemy_connector import (
    SQLAlchemyConnector,
    _apply_statement_timeout,
)
from sqlalchemy import inspect as sa_inspect

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
    _seed(
        c,
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE orders ("
        "  id INTEGER PRIMARY KEY,"
        "  customer_id INTEGER REFERENCES customers(id),"
        "  total REAL"
        ")",
    )

    schema = c.extract_schema()
    assert {t.name for t in schema.tables} == {"customers", "orders"}

    orders = next(t for t in schema.tables if t.name == "orders")
    cols = {col.name: col for col in orders.columns}
    assert cols["id"].is_primary_key is True
    assert cols["customer_id"].is_foreign_key is True
    assert cols["customer_id"].references == "customers.id"
    assert cols["total"].is_primary_key is False


class _RefusingInspector:
    """A real Inspector with some calls refused, as a restricted catalogue does.

    A Snowflake share, or a role without the grant, answers the constraint views
    with an error while still describing its tables and columns.
    """

    def __init__(self, inner: Any, refuse: tuple[str, ...]) -> None:
        self._inner = inner
        self._refuse = refuse

    def __getattr__(self, name: str) -> Any:
        if name in self._refuse:

            def _refuse(*_args: Any, **_kwargs: Any) -> Any:
                # Naming the call keeps assertions specific about which
                # reflection was refused, rather than that something was.
                raise RuntimeError(
                    f"{name} refused: Object "
                    "'SHARED_DB.INFORMATION_SCHEMA.KEY_COLUMN_USAGE' "
                    "does not exist or not authorized"
                )

            return _refuse
        return getattr(self._inner, name)


def _refusing(*calls: str) -> Any:
    """Patch the module's `inspect` so reflection meets a restricted catalogue."""
    return patch.object(
        sqlalchemy_connector_module,
        "inspect",
        lambda engine: _RefusingInspector(sa_inspect(engine), calls),
    )


def test_a_catalogue_that_refuses_keys_still_returns_the_tables(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The tables are the schema; the keys are an enrichment.

    Key reflection used to sit in the same `try` as the column reflection, so a
    catalogue that refuses constraint metadata cost the whole table. Every table
    failing the same way returned an empty schema -- which the caller cannot
    tell apart from a database that has no tables, and which the UI reported as
    "No tables found in this connector."
    """
    c = _connect(tmp_path)
    _seed(
        c,
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER"
        "  REFERENCES customers(id))",
    )

    with caplog.at_level(logging.WARNING), _refusing("get_pk_constraint", "get_foreign_keys"):
        schema = c.extract_schema()

    assert {t.name for t in schema.tables} == {"customers", "orders"}
    orders = next(t for t in schema.tables if t.name == "orders")
    assert {col.name for col in orders.columns} == {"id", "customer_id"}
    # Kept, but not claimed: no key it could not read is reported as present.
    assert all(not col.is_primary_key and not col.is_foreign_key for col in orders.columns)
    # And the reason is on the record, not swallowed.
    assert "KEY_COLUMN_USAGE" in caplog.text


def test_a_key_that_did_read_is_kept_when_only_the_other_kind_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Primary and foreign keys are separate catalogue reads, so separate `try`s.

    A role can hold one grant and not the other. Sharing one `try` discarded a
    primary key that had already been read because the foreign keys were refused
    after it -- degrading further than the evidence requires, which is the thing
    this change exists to stop doing.
    """
    c = _connect(tmp_path)
    _seed(
        c,
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER"
        "  REFERENCES customers(id))",
    )

    with caplog.at_level(logging.WARNING), _refusing("get_foreign_keys"):
        schema = c.extract_schema()

    orders = next(t for t in schema.tables if t.name == "orders")
    cols = {col.name: col for col in orders.columns}
    assert cols["id"].is_primary_key is True
    # Refused, so not claimed -- but it cost only itself.
    assert cols["customer_id"].is_foreign_key is False
    assert "foreign keys" in caplog.text
    assert "primary keys" not in caplog.text


def test_the_warning_for_a_skipped_table_says_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The column path is the only one that still drops a table, so it is the
    one an operator most needs a reason from."""
    c = _connect(tmp_path)
    _seed(c, "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")

    with caplog.at_level(logging.WARNING), _refusing("get_columns"):
        schema = c.extract_schema()

    assert schema.tables == []
    assert "get_columns refused" in caplog.text


def test_a_table_whose_columns_refuse_is_still_skipped(tmp_path: Path) -> None:
    """The existing behaviour, kept: without columns there is no table to return.

    Negative control for the split -- it must not have turned every reflection
    failure into a table with no columns.
    """
    c = _connect(tmp_path)
    _seed(c, "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")

    with _refusing("get_columns"):
        schema = c.extract_schema()

    assert schema.tables == []


def test_ddl_through_the_answer_path_is_not_undone_on_sqlite(tmp_path: Path) -> None:
    """The documented gap, asserted rather than relied upon.

    `capabilities.py`, `docs/database-hardening.md` and `docs/connectors.md` all
    now say the rollback does not reach DDL on every engine: MySQL, MariaDB and
    Oracle commit implicitly around it, and pysqlite does not open a transaction
    for it at all. Two fixtures in this file used to depend on that quietly, by
    creating their tables through `execute_query` -- so the accident was load
    bearing and the guarantee was not tested.

    It is stated here instead, on the engine the suite can actually reach. This
    is the case for which the database grant, not the connector, is the control.
    """
    c = _connect(tmp_path)

    assert c.execute_query("CREATE TABLE ddl_survives (id INTEGER)").error is None

    from sqlalchemy import inspect as _inspect

    engine = c._engine  # noqa: SLF001
    assert engine is not None
    assert "ddl_survives" in _inspect(engine).get_table_names(), (
        "SQLite DDL was undone by the rollback -- if this ever passes, the docs "
        "claiming the connector cannot undo DDL have become too pessimistic"
    )


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
