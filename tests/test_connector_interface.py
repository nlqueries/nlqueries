"""Tests for the DatabaseConnector abstract interface (nlqueries.connectors.base)."""

from __future__ import annotations

import pytest
from nlqueries.connectors import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)
from nlqueries.connectors.base import DatabaseConnector as DatabaseConnectorFromBase


def test_database_connector_cannot_be_instantiated_directly():
    """ABC with abstract methods must reject direct instantiation."""
    with pytest.raises(TypeError, match="abstract"):
        DatabaseConnector()


def test_base_module_export_is_same_class_as_package_export():
    assert DatabaseConnectorFromBase is DatabaseConnector


@pytest.mark.parametrize(
    "missing_methods, expected",
    [
        ([], True),
    ],
)
def test_subclass_implementing_all_methods_can_be_instantiated(missing_methods, expected):
    class CompleteConnector(DatabaseConnector):
        def connect(self, credentials: dict) -> None:
            return None

        def test_connection(self) -> bool:
            return True

        def extract_schema(self) -> SchemaSpec:
            return SchemaSpec(database="db", tables=[], extracted_at="2026-01-01T00:00:00Z")

        def extract_query_history(self, days: int = 30) -> list[QueryRecord]:
            return []

        def execute_query(self, sql: str) -> QueryResult:
            return QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None
            )

    connector = CompleteConnector()
    assert isinstance(connector, DatabaseConnector) is expected
    assert connector.test_connection() is True


@pytest.mark.parametrize(
    "method_to_omit",
    [
        "connect",
        "test_connection",
        "extract_schema",
        "extract_query_history",
        "execute_query",
    ],
)
def test_subclass_missing_any_abstract_method_cannot_be_instantiated(method_to_omit):
    """Omitting any single abstract method should still prevent instantiation."""
    methods = {
        "connect": lambda self, credentials: None,
        "test_connection": lambda self: True,
        "extract_schema": lambda self: SchemaSpec(
            database="db", tables=[], extracted_at="2026-01-01T00:00:00Z"
        ),
        "extract_query_history": lambda self, days=30: [],
        "execute_query": lambda self, sql: QueryResult(
            columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None
        ),
    }
    methods.pop(method_to_omit)

    IncompleteConnector = type("IncompleteConnector", (DatabaseConnector,), methods)

    with pytest.raises(TypeError, match="abstract"):
        IncompleteConnector()


def test_dataclasses_are_constructible_and_hold_values():
    column = ColumnSpec(
        name="id",
        type="integer",
        nullable=False,
        is_primary_key=True,
        is_foreign_key=False,
        references=None,
        description="Primary key",
    )
    table = TableSpec(
        name="users",
        schema="public",
        row_count=100,
        columns=[column],
        description="User accounts",
    )
    schema = SchemaSpec(database="app", tables=[table], extracted_at="2026-06-06T00:00:00Z")

    assert schema.tables[0].columns[0].name == "id"
    assert schema.tables[0].columns[0].is_primary_key is True

    fk_column = ColumnSpec(
        name="user_id",
        type="integer",
        nullable=False,
        is_primary_key=False,
        is_foreign_key=True,
        references="users.id",
        description=None,
    )
    assert fk_column.references == "users.id"

    record = QueryRecord(
        sql="SELECT 1",
        execution_count=5,
        avg_duration_ms=12.3,
        last_executed="2026-06-05T00:00:00Z",
    )
    assert record.execution_count == 5

    result = QueryResult(
        columns=["id"], rows=[[1], [2]], row_count=2, execution_time_ms=1.5, error=None
    )
    assert result.row_count == 2
    assert result.error is None
