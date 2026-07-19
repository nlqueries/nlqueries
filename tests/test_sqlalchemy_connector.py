"""
tests.test_sqlalchemy_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The generic URL-driven SQLAlchemyConnector, exercised against a file-backed
SQLite database (built in — no optional driver needed), so its dialect-agnostic
reflection and execution paths are covered end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector


def _connect(tmp_path: Path) -> SQLAlchemyConnector:
    db = tmp_path / "t.db"
    c = SQLAlchemyConnector()
    c.connect({"url": f"sqlite:///{db}"})
    return c


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


def test_execute_query_returns_rows_and_surfaces_errors(tmp_path: Path) -> None:
    c = _connect(tmp_path)
    c.execute_query("CREATE TABLE t (a INTEGER, b TEXT)")
    c.execute_query("INSERT INTO t VALUES (1, 'x'), (2, 'y')")

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
