"""
Tests for BigQueryConnector (nlqueries.connectors.bigquery).

Like the Snowflake connector tests, these mock the ``google-cloud-bigquery``
driver with ``unittest.mock`` — there is no free, ephemeral BigQuery project
to test against, so we exercise the connector's logic against a fake client
surface (``Client``, query jobs, datasets, and tables).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import ColumnSpec, QueryRecord, QueryResult, SchemaSpec, TableSpec
from nlqueries.connectors.bigquery import BigQueryConnector

from tests.conftest import granted

CREDENTIALS = {
    "project_id": "acme-prod",
    "dataset_id": "analytics",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_bigquery_is_registered_under_bigquery_key():
    assert CONNECTOR_REGISTRY["bigquery"] is BigQueryConnector


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_uses_application_default_credentials_when_no_key_given(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock(location=None)
    mock_client_cls.return_value = mock_client

    connector = granted(BigQueryConnector())
    connector.connect({"project_id": "acme-prod"})

    mock_client_cls.assert_called_once_with(project="acme-prod")
    assert connector._client is mock_client
    assert connector._project_id == "acme-prod"
    assert connector._dataset_id is None


@patch("nlqueries.connectors.bigquery.service_account.Credentials.from_service_account_info")
@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_builds_credentials_from_dict(mock_client_cls, mock_from_info):
    mock_creds = MagicMock()
    mock_from_info.return_value = mock_creds
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock(location="US")
    mock_client_cls.return_value = mock_client

    key_dict = {"type": "service_account", "project_id": "acme-prod"}
    connector = granted(BigQueryConnector())
    connector.connect({**CREDENTIALS, "service_account_json": key_dict})

    mock_from_info.assert_called_once_with(key_dict)
    mock_client_cls.assert_called_once_with(project="acme-prod", credentials=mock_creds)


@patch("nlqueries.connectors.bigquery.service_account.Credentials.from_service_account_file")
@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_builds_credentials_from_path(mock_client_cls, mock_from_file):
    mock_creds = MagicMock()
    mock_from_file.return_value = mock_creds
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock(location="US")
    mock_client_cls.return_value = mock_client

    connector = granted(BigQueryConnector())
    connector.connect({**CREDENTIALS, "service_account_json": "/path/to/key.json"})

    mock_from_file.assert_called_once_with("/path/to/key.json")
    mock_client_cls.assert_called_once_with(project="acme-prod", credentials=mock_creds)


@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_resolves_region_qualifier_from_dataset_location(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock(location="EUROPE-WEST1")
    mock_client_cls.return_value = mock_client

    connector = granted(BigQueryConnector())
    connector.connect(CREDENTIALS)

    mock_client.get_dataset.assert_called_once_with("analytics")
    assert connector._region_qualifier == "region-europe-west1"


@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_falls_back_to_default_region_without_dataset_id(mock_client_cls):
    mock_client_cls.return_value = MagicMock()

    connector = granted(BigQueryConnector())
    connector.connect({"project_id": "acme-prod"})

    assert connector._region_qualifier == "region-us"


@patch("nlqueries.connectors.bigquery.bigquery.Client")
def test_connect_falls_back_to_default_region_when_lookup_fails(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_dataset.side_effect = RuntimeError("not found")
    mock_client_cls.return_value = mock_client

    connector = granted(BigQueryConnector())
    connector.connect(CREDENTIALS)

    assert connector._region_qualifier == "region-us"


def test_methods_behave_before_connect_is_called():
    connector = granted(BigQueryConnector())

    # _require_client() raises directly...
    with pytest.raises(RuntimeError):
        connector._require_client()

    # ...but test_connection() and execute_query() catch it and surface it
    # gracefully (False / QueryResult.error) rather than propagating.
    assert connector.test_connection() is False

    result = connector.execute_query("SELECT 1")
    assert result.error is not None
    assert "connect()" in result.error


def _connector_with_mock_client() -> tuple[BigQueryConnector, MagicMock]:
    """Build a connector whose ``_client`` is a fully-mocked BigQuery client."""
    connector = granted(BigQueryConnector())
    connector._client = MagicMock()
    connector._project_id = "acme-prod"
    connector._dataset_id = "analytics"
    connector._region_qualifier = "region-us"
    return connector, connector._client


def _mock_client_for_timeout() -> tuple[BigQueryConnector, MagicMock]:
    connector, client = _connector_with_mock_client()
    result = MagicMock()
    result.schema = []
    result.__iter__.return_value = iter([])
    job = MagicMock()
    job.result.return_value = result
    job.total_bytes_processed = 0
    job.job_id = "j"
    client.query.return_value = job
    return connector, client


def test_execute_query_sets_job_timeout_ms():
    connector, client = _mock_client_for_timeout()
    connector.execute_query("SELECT 1", timeout_seconds=30)
    job_config = client.query.call_args.kwargs.get("job_config")
    assert job_config is not None
    # BigQuery stores job_timeout_ms as a string on the config object.
    assert int(job_config.job_timeout_ms) == 30000


def test_execute_query_applies_default_job_timeout(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 45)
    connector, client = _mock_client_for_timeout()
    connector.execute_query("SELECT 1")
    assert int(client.query.call_args.kwargs["job_config"].job_timeout_ms) == 45000


def test_execute_query_no_job_timeout_when_disabled(monkeypatch):
    """No timeout, asserted on the timeout rather than on the job config.

    A job config is always sent now -- it carries `use_legacy_sql=False` and
    `create_session=False` regardless of the timeout -- so "no job_config" no
    longer means "no timeout". The property this test is about is unchanged.
    """
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    connector, client = _mock_client_for_timeout()
    connector.execute_query("SELECT 1")
    assert client.query.call_args.kwargs["job_config"].job_timeout_ms is None


def test_execute_query_pins_the_dialect_and_refuses_a_session(monkeypatch):
    """BigQuery has no transaction to roll back, so the little that can be set
    on the job is set: the SQL dialect every validator in front of this one
    parsed, and no session for a later statement to join."""
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    connector, client = _mock_client_for_timeout()
    connector.execute_query("SELECT 1")

    job_config = client.query.call_args.kwargs["job_config"]
    assert job_config.use_legacy_sql is False
    assert job_config.create_session is False


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------


def test_test_connection_returns_true_when_query_succeeds():
    connector, mock_client = _connector_with_mock_client()
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job

    assert connector.test_connection() is True
    mock_client.query.assert_called_once_with("SELECT 1")
    mock_job.result.assert_called_once()


def test_test_connection_returns_false_on_driver_error(caplog):
    connector, mock_client = _connector_with_mock_client()
    mock_client.query.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="nlqueries.connectors.bigquery"):
        assert connector.test_connection() is False

    assert any("test_connection failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def _make_row(values: dict):
    row = MagicMock()
    row.values.return_value = list(values.values())
    return row


def test_execute_query_returns_columns_and_rows_and_logs_bytes_processed(caplog):
    connector, mock_client = _connector_with_mock_client()

    field_one = MagicMock(name="ONE")
    field_one.name = "one"
    field_two = MagicMock(name="TWO")
    field_two.name = "two"

    mock_result = MagicMock()
    mock_result.schema = [field_one, field_two]
    mock_result.__iter__.return_value = iter([_make_row({"one": 1, "two": "two"})])

    mock_job = MagicMock()
    mock_job.result.return_value = mock_result
    mock_job.total_bytes_processed = 12345
    mock_job.job_id = "job-abc-123"
    mock_client.query.return_value = mock_job

    with caplog.at_level(logging.INFO, logger="nlqueries.connectors.bigquery"):
        result = connector.execute_query("SELECT 1 AS one, 'two' AS two")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert result.columns == ["one", "two"]
    assert result.rows == [[1, "two"]]
    assert result.row_count == 1
    assert result.execution_time_ms >= 0

    # total_bytes_processed is surfaced as a logged metadata note, since
    # QueryResult has no dedicated field for job metadata.
    assert any(
        "total_bytes_processed=12345" in r.message and "job-abc-123" in r.message
        for r in caplog.records
    )


def test_execute_query_handles_statements_with_no_result_set():
    connector, mock_client = _connector_with_mock_client()

    mock_result = MagicMock()
    mock_result.schema = []
    mock_result.__iter__.return_value = iter([])

    mock_job = MagicMock()
    mock_job.result.return_value = mock_result
    mock_job.total_bytes_processed = 0
    mock_job.job_id = "job-ddl-1"
    mock_client.query.return_value = mock_job

    result = connector.execute_query("CREATE TABLE foo (id INT64)")

    assert result.error is None
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


def test_execute_query_captures_errors_without_raising():
    connector, mock_client = _connector_with_mock_client()
    mock_client.query.side_effect = RuntimeError("Not found: Table acme-prod:analytics.missing")

    result = connector.execute_query("SELECT * FROM analytics.missing")

    assert isinstance(result, QueryResult)
    assert result.error is not None
    assert "Not found" in result.error
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------


def _make_field(name: str, field_type: str, mode: str, description=None):
    field = MagicMock()
    field.name = name
    field.field_type = field_type
    field.mode = mode
    field.description = description
    return field


def _make_table(table_id: str, dataset_id: str, num_rows, description, schema_fields):
    table = MagicMock()
    table.table_id = table_id
    table.dataset_id = dataset_id
    table.num_rows = num_rows
    table.description = description
    table.schema = schema_fields
    return table


_CUSTOMERS_TABLE = _make_table(
    "customers",
    "analytics",
    2,
    "Customer accounts",
    [
        _make_field("id", "INTEGER", "REQUIRED", "Primary key"),
        _make_field("email", "STRING", "REQUIRED"),
        _make_field("tags", "STRING", "REPEATED"),
    ],
)

_ORDERS_TABLE = _make_table(
    "orders",
    "analytics",
    5,
    None,
    [
        _make_field("id", "INTEGER", "REQUIRED"),
        _make_field("customer_id", "INTEGER", "NULLABLE"),
    ],
)


def test_extract_schema_returns_full_schema_spec_for_single_dataset():
    connector, mock_client = _connector_with_mock_client()

    dataset_ref = MagicMock()
    mock_client.dataset.return_value = dataset_ref
    mock_client.list_tables.return_value = [MagicMock(), MagicMock()]
    mock_client.get_table.side_effect = [_CUSTOMERS_TABLE, _ORDERS_TABLE]

    schema = connector.extract_schema()

    mock_client.dataset.assert_called_once_with("analytics")
    mock_client.list_tables.assert_called_once_with(dataset_ref)
    mock_client.list_datasets.assert_not_called()

    assert isinstance(schema, SchemaSpec)
    assert schema.database == "acme-prod"
    assert schema.extracted_at  # non-empty ISO timestamp

    tables_by_name = {t.name: t for t in schema.tables}
    assert {"customers", "orders"} == set(tables_by_name)

    customers = tables_by_name["customers"]
    assert isinstance(customers, TableSpec)
    assert customers.schema == "analytics"
    assert customers.row_count == 2
    assert customers.description == "Customer accounts"

    customers_columns = {c.name: c for c in customers.columns}
    assert isinstance(customers_columns["id"], ColumnSpec)
    # REQUIRED -> not nullable
    assert customers_columns["id"].nullable is False
    # NULLABLE/REPEATED -> nullable
    assert customers_columns["tags"].nullable is True
    assert customers_columns["email"].nullable is False
    assert customers_columns["id"].description == "Primary key"

    # BigQuery has no enforced PK/FK constraint system exposed via these APIs.
    assert customers_columns["id"].is_primary_key is False
    assert customers_columns["id"].is_foreign_key is False
    assert customers_columns["id"].references is None

    orders = tables_by_name["orders"]
    orders_columns = {c.name: c for c in orders.columns}
    assert orders_columns["customer_id"].nullable is True
    assert orders.row_count == 5
    assert orders.description is None


def test_extract_schema_iterates_every_dataset_when_no_dataset_id_given():
    connector = granted(BigQueryConnector())
    connector._client = MagicMock()
    connector._project_id = "acme-prod"
    connector._dataset_id = None
    connector._region_qualifier = "region-us"
    mock_client = connector._client

    dataset_one = MagicMock(reference="ref-one")
    dataset_two = MagicMock(reference="ref-two")
    mock_client.list_datasets.return_value = [dataset_one, dataset_two]
    mock_client.list_tables.side_effect = [[MagicMock()], [MagicMock()]]
    mock_client.get_table.side_effect = [_CUSTOMERS_TABLE, _ORDERS_TABLE]

    schema = connector.extract_schema()

    mock_client.dataset.assert_not_called()
    mock_client.list_datasets.assert_called_once()
    assert mock_client.list_tables.call_count == 2
    assert {t.name for t in schema.tables} == {"customers", "orders"}


def test_field_to_column_spec_maps_modes_to_nullable():
    required = _make_field("id", "INTEGER", "REQUIRED")
    nullable = _make_field("name", "STRING", "NULLABLE")
    repeated = _make_field("tags", "STRING", "REPEATED")

    assert BigQueryConnector._field_to_column_spec(required).nullable is False
    assert BigQueryConnector._field_to_column_spec(nullable).nullable is True
    assert BigQueryConnector._field_to_column_spec(repeated).nullable is True


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------

_HISTORY_ROW = {
    "query": "SELECT * FROM customers",
    "job_count": 42,
    "avg_duration_ms": 123.4,
    "last_executed": "2026-06-01 00:00:00+00:00",
}


def test_extract_query_history_queries_information_schema_with_region_qualifier():
    connector, _ = _connector_with_mock_client()

    with patch.object(BigQueryConnector, "_query", return_value=[_HISTORY_ROW]) as mock_query:
        history = connector.extract_query_history(days=30)

    assert history == [
        QueryRecord(
            sql="SELECT * FROM customers",
            execution_count=42,
            avg_duration_ms=123.4,
            last_executed="2026-06-01 00:00:00+00:00",
        )
    ]

    sql_used = mock_query.call_args[0][1]
    assert "`region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT" in sql_used
    assert "statement_type = 'SELECT'" in sql_used
    assert "LIMIT 500" in sql_used


def test_extract_query_history_handles_null_aggregates():
    connector, _ = _connector_with_mock_client()
    row = {"query": "SELECT 1", "job_count": 1, "avg_duration_ms": None, "last_executed": None}

    with patch.object(BigQueryConnector, "_query", return_value=[row]):
        history = connector.extract_query_history(days=7)

    assert history == [
        QueryRecord(
            sql="SELECT 1",
            execution_count=1,
            avg_duration_ms=None,
            last_executed=None,
        )
    ]


def test_extract_query_history_returns_empty_list_when_view_is_not_accessible(caplog):
    connector, _ = _connector_with_mock_client()

    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.bigquery"),
        patch.object(BigQueryConnector, "_query", side_effect=RuntimeError("Access Denied")),
    ):
        history = connector.extract_query_history(days=30)

    assert history == []
    assert any("is not accessible" in r.message for r in caplog.records)
    assert any("returning an empty query history" in r.message for r in caplog.records)
