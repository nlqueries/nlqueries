"""
Tests for SnowflakeConnector (nlqueries.connectors.snowflake).

Unlike the PostgresConnector tests (which spin up a real database via
testcontainers), these tests mock the ``snowflake-connector-python`` driver
with ``unittest.mock`` — there is no free, ephemeral Snowflake instance to
test against, so we exercise the connector's logic against a fake driver
surface (connections, cursors, and query results).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import ColumnSpec, QueryRecord, QueryResult, SchemaSpec, TableSpec
from nlqueries.connectors.snowflake import SnowflakeConnector

from tests.conftest import granted

CREDENTIALS = {
    "account": "acme-prod",
    "user": "alice",
    "password": "s3cr3t",
    "warehouse": "COMPUTE_WH",
    "database": "ANALYTICS",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_snowflake_is_registered_under_snowflake_key():
    assert CONNECTOR_REGISTRY["snowflake"] is SnowflakeConnector


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_builds_connection_with_expected_kwargs(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect(CREDENTIALS)

    mock_connect.assert_called_once_with(
        account="acme-prod",
        user="alice",
        password="s3cr3t",
        warehouse="COMPUTE_WH",
        database="ANALYTICS",
    )
    assert connector._connection is mock_connect.return_value
    assert connector._database == "ANALYTICS"


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_includes_schema_only_when_provided(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect({**CREDENTIALS, "schema": "PUBLIC"})

    _, kwargs = mock_connect.call_args
    assert kwargs["schema"] == "PUBLIC"
    assert connector._db_schema == "PUBLIC"


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_omits_schema_when_not_provided(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect(CREDENTIALS)

    _, kwargs = mock_connect.call_args
    assert "schema" not in kwargs


def test_methods_behave_before_connect_is_called():
    connector = granted(SnowflakeConnector())

    # _require_connection() raises directly...
    with pytest.raises(RuntimeError):
        connector._require_connection()

    # ...but test_connection() and execute_query() catch it and surface it
    # gracefully (False / QueryResult.error) rather than propagating.
    assert connector.test_connection() is False

    result = connector.execute_query("SELECT 1")
    assert result.error is not None
    assert "connect()" in result.error


def _connector_with_mock_connection() -> tuple[SnowflakeConnector, MagicMock]:
    """Build a connector whose ``_connection`` is a fully-mocked driver connection."""
    connector = granted(SnowflakeConnector())
    connector._connection = MagicMock()
    connector._database = "ANALYTICS"
    return connector, connector._connection


def test_test_connection_returns_true_when_query_succeeds():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor

    assert connector.test_connection() is True
    mock_cursor.execute.assert_called_once_with("SELECT CURRENT_VERSION()")
    mock_cursor.close.assert_called_once()


def test_test_connection_returns_false_on_driver_error(caplog):
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("boom")
    mock_connection.cursor.return_value = mock_cursor

    with caplog.at_level(logging.ERROR, logger="nlqueries.connectors.snowflake"):
        assert connector.test_connection() is False

    assert any("test_connection failed" in r.message for r in caplog.records)
    # Cursor must still be closed even when execute() raises.
    mock_cursor.close.assert_called_once()


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def test_execute_query_returns_columns_and_rows():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.description = [("ONE", None), ("TWO", None)]
    mock_cursor.fetchall.return_value = [(1, "two")]
    mock_cursor.__iter__.return_value = iter([(1, "two")])
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("SELECT 1 AS one, 'two' AS two")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert result.columns == ["ONE", "TWO"]
    assert result.rows == [[1, "two"]]
    assert result.row_count == 1
    assert result.execution_time_ms >= 0
    mock_cursor.close.assert_called_once()


def _mock_cursor() -> tuple[SnowflakeConnector, MagicMock]:
    connector, mock_connection = _connector_with_mock_connection()
    cursor = MagicMock()
    cursor.description = None
    mock_connection.cursor.return_value = cursor
    return connector, cursor


def test_execute_query_passes_explicit_timeout_to_cursor():
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1", timeout_seconds=30)
    assert cursor.execute.call_args.kwargs.get("timeout") == 30


def test_execute_query_applies_default_timeout(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 45)
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1")
    assert cursor.execute.call_args.kwargs.get("timeout") == 45


def test_execute_query_no_timeout_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1")
    assert "timeout" not in cursor.execute.call_args.kwargs


def test_execute_query_handles_statements_with_no_result_set():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.description = None
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("CREATE TABLE foo (id INT)")

    assert result.error is None
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


def test_execute_query_captures_errors_without_raising():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("SQL compilation error: table does not exist")
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("SELECT * FROM this_table_does_not_exist")

    assert isinstance(result, QueryResult)
    assert result.error is not None
    assert "does not exist" in result.error
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0
    mock_cursor.close.assert_called_once()


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------

_TABLES_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "ROW_COUNT": 2,
        "COMMENT": "Customer accounts",
    },
    {"TABLE_SCHEMA": "PUBLIC", "TABLE_NAME": "ORDERS", "ROW_COUNT": 5, "COMMENT": None},
]

_COLUMNS_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "COLUMN_NAME": "ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": "Primary key",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "COLUMN_NAME": "EMAIL",
        "DATA_TYPE": "TEXT",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "COLUMN_NAME": "ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "COLUMN_NAME": "CUSTOMER_ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
]

_CONSTRAINT_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "CONSTRAINT_TYPE": "PRIMARY KEY",
        "COLUMN_NAME": "ID",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "CONSTRAINT_TYPE": "PRIMARY KEY",
        "COLUMN_NAME": "ID",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "CONSTRAINT_TYPE": "FOREIGN KEY",
        "COLUMN_NAME": "CUSTOMER_ID",
    },
]


def _query_side_effect(_connection, sql):
    if "INFORMATION_SCHEMA.TABLES" in sql:
        return list(_TABLES_ROWS)
    if "INFORMATION_SCHEMA.COLUMNS" in sql:
        return list(_COLUMNS_ROWS)
    if "TABLE_CONSTRAINTS" in sql:
        return list(_CONSTRAINT_ROWS)
    raise AssertionError(f"unexpected query: {sql}")


def test_extract_schema_returns_full_schema_spec():
    connector, _ = _connector_with_mock_connection()

    with patch.object(SnowflakeConnector, "_query", side_effect=_query_side_effect):
        schema = connector.extract_schema()

    assert isinstance(schema, SchemaSpec)
    assert schema.database == "ANALYTICS"
    assert schema.extracted_at  # non-empty ISO timestamp

    tables_by_name = {t.name: t for t in schema.tables}
    assert {"CUSTOMERS", "ORDERS"} == set(tables_by_name)

    customers = tables_by_name["CUSTOMERS"]
    assert isinstance(customers, TableSpec)
    assert customers.schema == "PUBLIC"
    assert customers.row_count == 2
    assert customers.description == "Customer accounts"

    customers_columns = {c.name: c for c in customers.columns}
    assert isinstance(customers_columns["ID"], ColumnSpec)
    assert customers_columns["ID"].is_primary_key is True
    assert customers_columns["ID"].is_foreign_key is False
    assert customers_columns["ID"].nullable is False
    assert customers_columns["EMAIL"].type == "TEXT"

    orders = tables_by_name["ORDERS"]
    orders_columns = {c.name: c for c in orders.columns}
    assert orders_columns["CUSTOMER_ID"].is_foreign_key is True
    assert orders_columns["CUSTOMER_ID"].is_primary_key is False
    # Resolving the referenced table/column requires REFERENTIAL_CONSTRAINTS,
    # which is intentionally out of scope — `references` stays None.
    assert orders_columns["CUSTOMER_ID"].references is None
    assert orders_columns["ID"].is_primary_key is True


def test_extract_schema_queries_use_the_connected_database():
    connector, _ = _connector_with_mock_connection()
    seen_sql: list[str] = []

    def record_and_delegate(_connection, sql):
        seen_sql.append(sql)
        return _query_side_effect(_connection, sql)

    with patch.object(SnowflakeConnector, "_query", side_effect=record_and_delegate):
        connector.extract_schema()

    assert any("ANALYTICS.INFORMATION_SCHEMA.TABLES" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.COLUMNS" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in sql for sql in seen_sql)


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------

_HISTORY_ROW = {
    "QUERY_TEXT": "SELECT * FROM customers",
    "EXECUTION_COUNT": 42,
    "AVG_DURATION_MS": 123.4,
    "LAST_EXECUTED": "2026-06-01 00:00:00",
}


def test_extract_query_history_prefers_account_usage():
    connector, _ = _connector_with_mock_connection()

    with patch.object(SnowflakeConnector, "_query", return_value=[_HISTORY_ROW]) as mock_query:
        history = connector.extract_query_history(days=30)

    assert history == [
        QueryRecord(
            sql="SELECT * FROM customers",
            execution_count=42,
            avg_duration_ms=123.4,
            last_executed="2026-06-01 00:00:00",
        )
    ]
    sql_used = mock_query.call_args[0][1]
    assert "SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY" in sql_used


def test_extract_query_history_falls_back_to_information_schema(caplog):
    connector, _ = _connector_with_mock_connection()

    def side_effect(_connection, sql):
        if "ACCOUNT_USAGE" in sql:
            raise RuntimeError("SQL access control error: ACCOUNT_USAGE not authorized")
        return [_HISTORY_ROW]

    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=side_effect) as mock_query,
    ):
        history = connector.extract_query_history(days=7)

    assert len(history) == 1
    assert history[0].execution_count == 42
    assert any(
        "falling back to INFORMATION_SCHEMA.QUERY_HISTORY" in r.message for r in caplog.records
    )

    # First call hit ACCOUNT_USAGE, second hit the INFORMATION_SCHEMA fallback.
    assert mock_query.call_count == 2
    second_sql = mock_query.call_args_list[1][0][1]
    assert "INFORMATION_SCHEMA.QUERY_HISTORY" in second_sql


def test_extract_query_history_returns_empty_list_when_both_sources_fail(caplog):
    connector, _ = _connector_with_mock_connection()

    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=RuntimeError("nope")),
    ):
        history = connector.extract_query_history(days=30)

    assert history == []
    assert any("returning an empty query history" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# list_security_policies (row-access + masking introspection — Block G)
# ---------------------------------------------------------------------------


def test_list_security_policies_maps_masking_and_row_access():
    connector, _ = _connector_with_mock_connection()
    refs = [
        {
            "POLICY_NAME": "mask_email",
            "POLICY_KIND": "MASKING_POLICY",
            "REF_SCHEMA_NAME": "PUBLIC",
            "REF_ENTITY_NAME": "CUSTOMERS",
            "REF_COLUMN_NAME": "EMAIL",
        },
        {
            "POLICY_NAME": "region_rap",
            "POLICY_KIND": "ROW_ACCESS_POLICY",
            "REF_SCHEMA_NAME": "PUBLIC",
            "REF_ENTITY_NAME": "ORDERS",
            "REF_COLUMN_NAME": None,
        },
        # A policy on the INFORMATION_SCHEMA is a system artifact → skipped.
        {
            "POLICY_NAME": "sys",
            "POLICY_KIND": "MASKING_POLICY",
            "REF_SCHEMA_NAME": "INFORMATION_SCHEMA",
            "REF_ENTITY_NAME": "X",
            "REF_COLUMN_NAME": "Y",
        },
    ]
    with patch.object(SnowflakeConnector, "_query", return_value=refs):
        report = connector.list_security_policies()

    assert report.supported is True
    by_name = {p.name: p for p in report.policies}
    assert set(by_name) == {"mask_email", "region_rap"}  # system schema dropped

    mask = by_name["mask_email"]
    assert mask.kind == "column"
    assert mask.columns == ["EMAIL"]
    assert mask.table == "CUSTOMERS"

    rap = by_name["region_rap"]
    assert rap.kind == "row"
    assert rap.columns == []
    # Row-access bodies aren't fetched → empty expression → caller marks untranslatable.
    assert rap.expression == ""


def test_list_security_policies_degrades_when_account_usage_unavailable(caplog):
    connector, _ = _connector_with_mock_connection()
    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=RuntimeError("not granted")),
    ):
        report = connector.list_security_policies()

    assert report.supported is True
    assert report.policies == []
    assert any("POLICY_REFERENCES unavailable" in r.message for r in caplog.records)
