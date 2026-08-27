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

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nlqueries.execution import (
    DEFAULT_POLICY,
    ExecutionNotPermitted,
    ExecutionPolicy,
)


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
    # Both default so every existing constructor keeps working. A result that
    # stopped early has to say so: silently returning the first N rows of a
    # larger answer is a wrong answer, not a partial one.
    truncated: bool = False
    truncation_reason: str | None = None  # "row_budget" | "byte_budget" | None


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

    def close(self) -> None:
        """Release whatever this connector holds open.

        A default rather than an abstract method: most connectors keep a
        SQLAlchemy engine on ``_engine`` and nothing else, and the ones that do
        not should not be made to write an empty override. Called when a cached
        connector is evicted — without it, engines would be released only by
        garbage collection, which is not a schedule a customer's DBA would
        recognise as one.
        """
        engine = getattr(self, "_engine", None)
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.dispose()

    def bind_execution_policy(self, policy: ExecutionPolicy) -> None:
        """Grant this connector permission to execute statements.

        Bound to the connector rather than passed per call. A per-call
        parameter must be supplied at every call site, and can therefore be
        supplied incorrectly at any of them, which is the defect this replaces
        rather than a reduced form of it. A connector in an executable state is
        one that a caller was deliberately permitted to open.
        """
        self._execution_policy = policy

    @property
    def execution_policy(self) -> ExecutionPolicy:
        """The permission held by this connector. Denied unless granted."""
        return getattr(self, "_execution_policy", DEFAULT_POLICY)

    def execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute ``sql`` if this connector holds permission to do so.

        Deliberately concrete: the check belongs to the interface rather than
        to each implementation, so that no connector -- including one added
        later -- can omit it. Implementations override :meth:`_execute_query`,
        which is reached only through this method.

        The orchestration layer performs its own check before reaching here.
        This layer does not depend on that check being correct.
        """
        policy = self.execution_policy
        if not policy.may_execute:
            raise ExecutionNotPermitted(f"{type(self).__name__}.execute_query", policy)
        return self._execute_query(sql, timeout_seconds, max_rows)

    @abstractmethod
    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute ``sql`` against the connected database and return the result.

        Reached only through :meth:`execute_query`, which checks permission
        first. Do not call it directly, and do not re-implement the check here.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: Optional server-side execution budget (Task 26.5
                — Sprint 26), so a runaway query is aborted by the database
                itself rather than left running — and holding locks/connections
                — after the caller has already given up waiting. Connectors
                that don't support a statement-level timeout ignore this.
            max_rows: Most rows to materialise. ``None`` uses
                ``CONNECTOR_MAX_FETCH_ROWS``; the effective budget is never
                larger than that, so a caller cannot ask for an unbounded read.

                This is a memory bound, not a LIMIT: the row cap that shapes the
                *answer* lives above the connector, and injecting a LIMIT into
                the SQL here would change aggregate semantics and produce
                silently wrong results. What it prevents is one ``SELECT *``
                over a large table materialising every row in the worker before
                anything upstream gets to discard them.
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


class PermittedConnector(DatabaseConnector):
    """A per-request view of a shared connector, carrying that request's permission.

    Connectors are pooled and reused across concurrent requests, so permission
    cannot be stored on the connector itself: a grant made by one request would
    apply to every other request holding the same object, and the resulting
    escalation would not be observable from the connector's state.

    The permission is held here instead, on a wrapper created per request,
    while the pooled connection remains shared. Enforcement remains the
    inherited :meth:`DatabaseConnector.execute_query`, so the rule has a single
    implementation; this class determines only which policy that rule consults.

    Opening a connector is deliberately not gated. Schema extraction, history
    mining and connection tests read metadata that generation-only callers
    require; refusing construction would prevent a ``--no-execute`` run from
    describing the tables it generates SQL against. Execution is what is
    permitted, and execution is therefore what is checked.
    """

    def __init__(self, inner: DatabaseConnector, policy: ExecutionPolicy) -> None:
        self._inner = inner
        self._execution_policy = policy

    # -- delegation ------------------------------------------------------
    def connect(self, credentials: dict[str, Any]) -> None:
        self._inner.connect(credentials)

    def test_connection(self) -> bool:
        return self._inner.test_connection()

    def extract_schema(self) -> SchemaSpec:
        return self._inner.extract_schema()

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        return self._inner.extract_query_history(days, limit)

    def get_schema_summary(self) -> tuple[int, int]:
        return self._inner.get_schema_summary()

    def list_security_policies(self) -> SecurityPolicyReport:
        return self._inner.list_security_policies()

    def close(self) -> None:
        """Intentionally a no-op.

        The wrapped connector is pooled and shared; closing it here would
        dispose of an engine that concurrent requests are using. The loader's
        contract already states that callers must not close what it returns,
        and this enforces that contract rather than only documenting it.
        """

    # -- the one thing that is gated -------------------------------------
    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        # Reached only through the inherited `execute_query`, which has
        # already checked this wrapper's policy. The inner private method is
        # called directly to keep enforcement in one place: the inner public
        # method would consult its own unbound, denying policy.
        return self._inner._execute_query(sql, timeout_seconds, max_rows)
