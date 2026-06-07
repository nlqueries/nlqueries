"""
nlqueries.connectors.base
~~~~~~~~~~~~~~~~~~~~~~~~~
Defines the public connector interface for nlqueries-core.

Every database integration (Postgres, Snowflake, BigQuery, ...) implements
``DatabaseConnector``. This is the contract the rest of the OSS package (CLI,
MCP server, knowledge base) is written against, so it can work with any
database without knowing the underlying driver.

This module is part of the public OSS API: it ships in the open-source
``nlqueries-core`` package and has no dependency on the enterprise layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnSpec:
    """Describes a single column within a table."""

    name: str
    type: str
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    references: str | None  # "table.column" if FK
    description: str | None


@dataclass
class TableSpec:
    """Describes a single table (or view) within a schema."""

    name: str
    schema: str
    row_count: int | None
    columns: list[ColumnSpec]
    description: str | None


@dataclass
class SchemaSpec:
    """The full extracted schema for a database."""

    database: str
    tables: list[TableSpec]
    extracted_at: str  # ISO timestamp


@dataclass
class QueryRecord:
    """A historical query, as surfaced by query-history extraction."""

    sql: str
    execution_count: int
    avg_duration_ms: float | None
    last_executed: str | None


@dataclass
class QueryResult:
    """The result of executing a query against the connected database."""

    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    execution_time_ms: float
    error: str | None


class DatabaseConnector(ABC):
    """Abstract base class for all database connectors.

    Concrete subclasses (e.g. a Postgres or Snowflake connector) must
    implement every method below. This is the minimal surface area the
    rest of nlqueries-core relies on to connect to a database, introspect
    its schema and query history, and run queries against it.
    """

    @abstractmethod
    def connect(self, credentials: dict) -> None:
        """Establish a connection to the database using the given credentials."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify that the current connection is alive and usable."""
        ...

    @abstractmethod
    def extract_schema(self) -> SchemaSpec:
        """Introspect and return the full schema of the connected database."""
        ...

    @abstractmethod
    def extract_query_history(self, days: int = 30) -> list[QueryRecord]:
        """Return recent query history covering the last ``days`` days."""
        ...

    @abstractmethod
    def execute_query(self, sql: str) -> QueryResult:
        """Execute ``sql`` against the connected database and return the result."""
        ...
