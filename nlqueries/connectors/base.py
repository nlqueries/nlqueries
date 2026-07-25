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


# Policy kinds surfaced by :meth:`DatabaseConnector.list_security_policies`.
POLICY_ROW = "row"  # restricts which rows are visible (RLS / row-access policy)
POLICY_COLUMN = "column"  # masks/hides a column (column masking policy)


@dataclass
class SecurityPolicy:
    """A row- or column-level security artifact discovered in the source database.

    Read-only introspection surfaces these so the caller can *suggest* equivalent
    NLQueries row filters / column exclusions — it never enforces them here.
    ``expression`` is the raw predicate the database reports (e.g. a Postgres RLS
    ``USING`` clause); it may be empty when the source doesn't expose a translatable
    body (e.g. a Snowflake policy defined as an opaque function), in which case the
    caller must treat the policy as "cannot translate — review manually" rather
    than guess. ``columns`` names the masked column(s) for :data:`POLICY_COLUMN`
    policies and is empty for :data:`POLICY_ROW` policies.
    """

    name: str
    kind: str  # POLICY_ROW | POLICY_COLUMN
    table_schema: str
    table: str
    columns: list[str]
    expression: str
    roles: list[str]


@dataclass
class SecurityPolicyReport:
    """The result of :meth:`DatabaseConnector.list_security_policies`.

    ``supported`` is ``False`` when the connector type can't introspect security
    policies at all (the default) — distinct from a supported connector that
    simply found none (``supported=True, policies=[]``).
    """

    supported: bool
    policies: list[SecurityPolicy]


class DatabaseConnector(ABC):
    """Abstract base class for all database connectors.

    Concrete subclasses (e.g. a Postgres or Snowflake connector) must
    implement every method below. This is the minimal surface area the
    rest of nlqueries-core relies on to connect to a database, introspect
    its schema and query history, and run queries against it.
    """

    @abstractmethod
    def connect(self, credentials: dict[str, Any]) -> None:
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
    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return recent query history covering the last ``days`` days.

        At most ``limit`` records are returned, ordered by execution count
        descending so the most-used queries are always included.
        """
        ...

    @abstractmethod
    def execute_query(self, sql: str, timeout_seconds: float | None = None) -> QueryResult:
        """Execute ``sql`` against the connected database and return the result.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: Optional server-side execution budget (Task 26.5
                — Sprint 26), so a runaway query is aborted by the database
                itself rather than left running — and holding locks/connections
                — after the caller has already given up waiting. Connectors
                that don't support a statement-level timeout ignore this.
        """
        ...

    def get_schema_summary(self) -> tuple[int, int]:
        """Return *(table_count, column_count)* for the connected database.

        Delegates to :meth:`extract_schema` by default.  Subclasses may
        override this method with a faster implementation (e.g. a single
        ``COUNT`` query against ``information_schema``) if introspecting the
        full schema would be too slow.
        """
        spec = self.extract_schema()
        return len(spec.tables), sum(len(t.columns) for t in spec.tables)

    def list_security_policies(self) -> SecurityPolicyReport:
        """Introspect row-/column-level security policies in the source database.

        An **optional, read-only** capability: the default reports it unsupported,
        so connectors that can't (or don't yet) introspect their security catalog
        keep working unchanged. Connectors that can (Postgres RLS, Snowflake
        row-access / masking) override this.

        Implementations must be **best-effort and degrade gracefully**: if the
        metadata views aren't readable by the connector's role, return
        ``SecurityPolicyReport(supported=True, policies=[])`` rather than raising —
        a missing grant is not an error, it just means "nothing to suggest".
        Introspection reuses the existing connection; it needs no new credentials.
        """
        return SecurityPolicyReport(supported=False, policies=[])
