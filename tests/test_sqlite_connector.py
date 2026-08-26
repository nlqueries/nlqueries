"""
Tests for nlqueries.connectors.sqlite.SQLiteConnector — the stdlib file-based
SQLite connector. Runs against real in-memory / temp-file SQLite databases (no
mocks needed), mirroring test_sqlalchemy_connector.py's SQLite coverage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import SchemaSpec
from nlqueries.connectors.sqlite import SQLiteConnector, _quote_ident

from tests.conftest import granted


def _seeded(tmp_path: Path) -> SQLiteConnector:
    """A connector on a temp-file DB with a two-table FK schema and some rows."""
    db = tmp_path / "shop.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE customers (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            note TEXT
        );
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER,
            total       REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        INSERT INTO customers (id, name) VALUES (1, 'Ada'), (2, 'Grace');
        INSERT INTO orders (id, customer_id, total) VALUES (1, 1, 9.5), (2, 1, 3.0), (3, 2, 7.0);
        """
    )
    raw.commit()
    raw.close()

    c = granted(SQLiteConnector())
    c.connect({"database": str(db)})
    return c


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------


def test_connect_defaults_to_in_memory() -> None:
    c = granted(SQLiteConnector())
    c.connect({})  # no "database" key
    assert c._database == ":memory:"
    assert c.test_connection() is True


def test_connect_ignores_server_credential_keys() -> None:
    # host/port/user/password are accepted (same dict shape as server connectors)
    # and ignored — only "database" matters.
    c = granted(SQLiteConnector())
    c.connect({"database": ":memory:", "host": "x", "port": 5432, "user": "u", "password": "p"})
    assert c.test_connection() is True


def test_test_connection_false_before_connect() -> None:
    assert SQLiteConnector().test_connection() is False


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------


def test_extract_query_history_always_empty(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    assert c.extract_query_history() == []
    assert c.extract_query_history(days=90, limit=10) == []


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def test_execute_query_returns_rows(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT id, name FROM customers ORDER BY id")
    assert result.error is None
    assert result.columns == ["id", "name"]
    assert result.rows == [[1, "Ada"], [2, "Grace"]]
    assert result.row_count == 2
    assert result.execution_time_ms >= 0


def test_execute_query_non_select_has_no_columns(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    result = c.execute_query("UPDATE customers SET note = 'x' WHERE id = 1")
    assert result.error is None
    assert result.columns == []
    assert result.rows == []


def test_execute_query_surfaces_error(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT * FROM does_not_exist")
    assert result.error is not None
    assert "does_not_exist" in result.error
    assert result.rows == []


def test_execute_query_with_timeout_budget_still_returns(tmp_path: Path) -> None:
    # A fast query under a generous budget completes normally (watchdog cancelled).
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT COUNT(*) FROM orders", timeout_seconds=30)
    assert result.error is None
    assert result.rows == [[3]]


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------


def test_extract_schema_builds_spec(tmp_path: Path) -> None:
    spec = _seeded(tmp_path).extract_schema()
    assert isinstance(spec, SchemaSpec)
    by_name = {t.name: t for t in spec.tables}
    assert set(by_name) == {"customers", "orders"}

    customers = by_name["customers"]
    assert customers.schema == "main"
    assert customers.row_count == 2
    cols = {col.name: col for col in customers.columns}
    assert cols["id"].is_primary_key is True
    assert cols["id"].type == "INTEGER"
    assert cols["name"].nullable is False  # NOT NULL
    assert cols["note"].nullable is True

    orders = by_name["orders"]
    assert orders.row_count == 3
    ocols = {col.name: col for col in orders.columns}
    assert ocols["customer_id"].is_foreign_key is True
    assert ocols["customer_id"].references == "customers.id"
    assert ocols["total"].is_foreign_key is False


def test_extract_schema_skips_internal_tables(tmp_path: Path) -> None:
    # Force creation of an internal sqlite_sequence table via AUTOINCREMENT.
    db = tmp_path / "auto.db"
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    raw.execute("INSERT INTO t (v) VALUES ('a')")
    raw.commit()
    raw.close()
    c = granted(SQLiteConnector())
    c.connect({"database": str(db)})
    names = {t.name for t in c.extract_schema().tables}
    assert names == {"t"}
    assert not any(n.startswith("sqlite_") for n in names)


# ---------------------------------------------------------------------------
# registry + helpers
# ---------------------------------------------------------------------------


def test_registered_in_connector_registry() -> None:
    assert CONNECTOR_REGISTRY.get("sqlite") is SQLiteConnector


def test_quote_ident_escapes_double_quotes() -> None:
    assert _quote_ident("orders") == '"orders"'
    assert _quote_ident('we"ird') == '"we""ird"'


def test_build_url_for_sqlite() -> None:
    from nlqueries.cli.main import _build_url

    assert _build_url("sqlite", "", 0, "/data/app.db", "", "") == "sqlite:////data/app.db"
    assert _build_url("sqlite", "", 0, ":memory:", "", "") == "sqlite:///:memory:"
    assert _build_url("sqlite", "", 0, "", "", "") == "sqlite:///:memory:"
