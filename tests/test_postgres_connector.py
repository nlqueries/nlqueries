"""
Tests for PostgresConnector (nlqueries.connectors.postgres).

These are integration tests backed by a real, ephemeral PostgreSQL instance
spun up in Docker via testcontainers. They are skipped automatically when
Docker is not available (e.g. in CI environments without a Docker daemon).
"""

from __future__ import annotations

import logging

import pytest
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import ColumnSpec, QueryResult, SchemaSpec, TableSpec
from nlqueries.connectors.postgres import PostgresConnector

testcontainers_postgres = pytest.importorskip(
    "testcontainers.postgres", reason="testcontainers[postgres] is not installed"
)
from testcontainers.postgres import PostgresContainer  # noqa: E402


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker is not available in this environment"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_container():
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="module")
def credentials(pg_container) -> dict:
    return {
        "host": pg_container.get_container_host_ip(),
        "port": int(pg_container.get_exposed_port(5432)),
        "database": pg_container.dbname,
        "user": pg_container.username,
        "password": pg_container.password,
    }


@pytest.fixture()
def connector(credentials) -> PostgresConnector:
    c = PostgresConnector()
    c.connect(credentials)
    return c


@pytest.fixture(scope="module")
def seeded_connector(credentials):
    """A connected PostgresConnector with a small schema (PK + FK) created."""
    c = PostgresConnector()
    c.connect(credentials)

    setup_statements = [
        "DROP TABLE IF EXISTS order_items",
        "DROP TABLE IF EXISTS orders",
        "DROP TABLE IF EXISTS customers",
        """
        CREATE TABLE customers (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            name TEXT
        )
        """,
        """
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            total NUMERIC(10, 2) NOT NULL
        )
        """,
        """
        CREATE TABLE order_items (
            id SERIAL PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1
        )
        """,
        "INSERT INTO customers (email, name) "
        "VALUES ('a@example.com', 'Alice'), ('b@example.com', 'Bob')",
        "INSERT INTO orders (customer_id, total) VALUES (1, 19.99), (2, 5.00)",
        "INSERT INTO order_items (order_id, sku, quantity) VALUES (1, 'WIDGET-1', 2)",
        "ANALYZE",
    ]
    for stmt in setup_statements:
        result = c.execute_query(stmt)
        assert result.error is None, f"setup statement failed: {stmt!r} -> {result.error}"

    return c


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_postgres_is_registered_under_postgres_key():
    assert CONNECTOR_REGISTRY["postgres"] is PostgresConnector


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------

def test_connect_builds_engine_and_test_connection_succeeds(credentials):
    connector = PostgresConnector()
    connector.connect(credentials)
    assert connector.test_connection() is True


def test_test_connection_returns_false_for_bad_credentials(credentials):
    bad_credentials = {**credentials, "password": "definitely-not-the-password"}
    connector = PostgresConnector()
    connector.connect(bad_credentials)
    assert connector.test_connection() is False


def test_methods_behave_before_connect_is_called():
    connector = PostgresConnector()

    # _require_engine() raises directly...
    with pytest.raises(RuntimeError):
        connector._require_engine()

    # ...but test_connection() and execute_query() catch it and surface it
    # gracefully (False / QueryResult.error) rather than propagating.
    assert connector.test_connection() is False

    result = connector.execute_query("SELECT 1")
    assert result.error is not None
    assert "connect()" in result.error


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------

def test_execute_query_returns_columns_and_rows(connector):
    result = connector.execute_query("SELECT 1 AS one, 'two' AS two")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert result.columns == ["one", "two"]
    assert result.rows == [[1, "two"]]
    assert result.row_count == 1
    assert result.execution_time_ms >= 0


def test_execute_query_captures_errors_without_raising(connector):
    result = connector.execute_query("SELECT * FROM this_table_does_not_exist")

    assert isinstance(result, QueryResult)
    assert result.error is not None
    assert "this_table_does_not_exist" in result.error
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------

def test_extract_schema_returns_full_schema_spec(seeded_connector):
    schema = seeded_connector.extract_schema()

    assert isinstance(schema, SchemaSpec)
    assert schema.database == seeded_connector._require_engine().url.database
    assert schema.extracted_at  # non-empty ISO timestamp

    tables_by_name = {t.name: t for t in schema.tables}
    assert {"customers", "orders", "order_items"} <= set(tables_by_name)

    customers = tables_by_name["customers"]
    assert isinstance(customers, TableSpec)
    assert customers.schema == "public"

    customers_columns = {c.name: c for c in customers.columns}
    assert isinstance(customers_columns["id"], ColumnSpec)
    assert customers_columns["id"].is_primary_key is True
    assert customers_columns["id"].is_foreign_key is False
    assert customers_columns["email"].nullable is False

    orders = tables_by_name["orders"]
    orders_columns = {c.name: c for c in orders.columns}
    assert orders_columns["customer_id"].is_foreign_key is True
    assert orders_columns["customer_id"].references == "customers.id"
    assert orders_columns["id"].is_primary_key is True


def test_extract_schema_row_counts_are_present_after_analyze(seeded_connector):
    schema = seeded_connector.extract_schema()
    tables_by_name = {t.name: t for t in schema.tables}

    # row_count comes from pg_class.reltuples (an estimate refreshed by ANALYZE),
    # so it should be a non-negative number once the table has been analysed.
    for name in ("customers", "orders", "order_items"):
        assert tables_by_name[name].row_count is not None
        assert tables_by_name[name].row_count >= 0


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------

def test_extract_query_history_returns_empty_list_when_extension_missing(connector, caplog):
    # The default postgres:16-alpine image does not ship pg_stat_statements,
    # so this exercises the "extension missing" graceful-degradation path.
    with caplog.at_level(logging.WARNING, logger="nlqueries.connectors.postgres"):
        history = connector.extract_query_history(days=30)

    assert history == []
    assert any("pg_stat_statements" in record.message for record in caplog.records)
