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
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from nlqueries.execution import (
    DEFAULT_POLICY,
    ExecutionNotPermitted,
    ExecutionPolicy,
)

logger = logging.getLogger(__name__)


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
    #: Why the result stopped short, and the authoritative list of values:
    #:
    #: - ``"row_budget"``   -- more rows existed than the connector was asked for
    #: - ``"byte_budget"``  -- the result was too large to hold in memory
    #: - ``"orchestrator_row_cap"`` -- set downstream, not by a connector: the
    #:   orchestrator's ``sql_table`` frame returns at most ``_MAX_RESULT_ROWS``
    #:   rows to MCP and CLI callers, and says so through this same field
    #: - ``None``           -- not truncated
    #:
    #: A caller may branch on these, so a new value belongs here first.
    truncation_reason: str | None = None


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

    #: Attributes that hold a live handle to the database, in the order they are
    #: released. Named rather than discovered, so adding a connector that holds
    #: its handle somewhere new is a deliberate edit here and not a silent leak.
    _HANDLE_ATTRS: ClassVar[tuple[str, ...]] = ("_conn", "_connection", "_client")

    def _use_state(self) -> tuple[threading.Lock, dict[str, Any]]:
        """This instance's lock and use counter, created on first need.

        Lazily, because connectors are not required to call ``super().__init__``
        and several do not; a counter that existed only for the well-behaved
        ones would protect exactly the connectors that did not need it.

        Created atomically, because the first two callers of a pooled connector
        can arrive on different threads at the same moment -- which is the only
        situation any of this matters in.
        """
        # Keyed distinctly from this method's own name: an entry in the
        # instance dict shadows the bound method, so storing the pair under
        # `_use_state` makes the second call find a tuple where it expects a
        # method and fail with "'tuple' object is not callable".
        existing = self.__dict__.get("_use_state_pair")
        if existing is None:
            # One `setdefault`, publishing the lock and the counter together.
            # Two statements could not do it safely: writing the dict before the
            # lock lets a second thread see the counter and miss the lock, and
            # two threads both taking a creation branch end up with a state each
            # -- so a count incremented against one is invisible to the `close`
            # reading the other, and the handle is closed under a running
            # statement after all. That is the failure this counter exists to
            # prevent, reintroduced by the counter's own initialisation.
            #
            # `dict.setdefault` is a single C-level operation, so the loser of
            # the race gets the winner's pair rather than its own. The pair it
            # allocated and did not install is simply collected.
            existing = self.__dict__.setdefault(
                "_use_state_pair", (threading.Lock(), {"count": 0, "deferred": False})
            )
        lock, state = existing
        return lock, state

    @contextlib.contextmanager
    def in_use(self) -> Iterator[None]:
        """Hold this connector open for the duration of one operation.

        Taken by :class:`PermittedConnector`, the per-request view the loader
        hands out, around everything that touches the database. A ``close``
        arriving while the count is non-zero is deferred to whichever caller
        leaves last, so an eviction cannot pull the handle out from under a
        statement that is still running.
        """
        lock, state = self._use_state()
        with lock:
            state["count"] += 1
        try:
            yield
        finally:
            release = False
            with lock:
                state["count"] -= 1
                if state["count"] <= 0 and state["deferred"]:
                    state["deferred"] = False
                    release = True
            if release:
                self._release()

    def close(self) -> None:
        """Release whatever this connector holds open.

        A default rather than an abstract method: a connector that holds nothing
        should not have to write an empty override. Called when the loader
        evicts a cached connector — without it, what the connector holds is
        released only by garbage collection, which is not a schedule a
        customer's DBA would recognise as one.

        This used to dispose ``_engine`` alone, on the reading that the others
        "keep a SQLAlchemy engine and nothing else". Five did not: BigQuery
        holds a ``_client``, DuckDB, Redshift and SQLite a ``_conn``, Snowflake a
        ``_connection``. None of them overrode this, so eviction released
        nothing for any of them and the server-side session stayed open until
        the object was collected — on Redshift and Snowflake, a real session
        against a real warehouse.

        The engine and the raw handles are treated differently on purpose.
        ``engine.dispose()`` returns the pool's connections and leaves the engine
        usable — it builds a new pool on next use — so the attribute stays. A raw
        DBAPI handle is dead once closed, so it is cleared: the connectors guard
        it with ``_require_conn``, which raises a plain "not connected" for
        ``None`` and would otherwise hand the caller a closed handle to fail on
        further in.
        """
        lock, state = self._use_state()
        with lock:
            if state["count"] > 0:
                # Someone is mid-call. The loader disposes entries other callers
                # already hold -- `_cache_get` on a stale one, `_cache_put` on a
                # replaced one, `invalidate_connector_cache` on demand -- and
                # that was harmless while this only disposed an engine, because
                # `dispose()` leaves checked-out connections alone. Closing a raw
                # handle is not harmless: it aborts the statement running on it,
                # and with the attribute cleared the caller is told "connect()
                # must be called before use", which points at configuration
                # rather than at the eviction that actually happened.
                state["deferred"] = True
                return
        self._release()

    def _release(self) -> None:
        """Close the handles, unconditionally. :meth:`close` is the gate."""
        engine = getattr(self, "_engine", None)
        if engine is not None:
            # Disposed, not cleared: `dispose()` returns the pool's connections
            # and leaves the engine usable, building a new pool on next use.
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001 - releasing must not raise
                logger.warning(
                    "%s could not dispose its engine; its pooled connections may still be open",
                    type(self).__name__,
                    exc_info=True,
                )

        for attr in self._HANDLE_ATTRS:
            handle = getattr(self, attr, None)
            if handle is None:
                continue
            # Cleared before the close, not after: a driver that raises on close
            # would otherwise leave the attribute pointing at a handle this
            # method has already given up on, and the next caller would use it.
            setattr(self, attr, None)
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - releasing must not raise
                # Logged, because this is exactly the case where the session this
                # method exists to release is probably still open on the server.
                # The loader's `_dispose` suppresses as well, so without this the
                # leak left behind is invisible from both sides.
                logger.warning(
                    "%s could not close %s; the server-side session may still be open",
                    type(self).__name__,
                    attr,
                    exc_info=True,
                )

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
        with self._inner.in_use():
            return self._inner.test_connection()

    def extract_schema(self) -> SchemaSpec:
        with self._inner.in_use():
            return self._inner.extract_schema()

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        with self._inner.in_use():
            return self._inner.extract_query_history(days, limit)

    def get_schema_summary(self) -> tuple[int, int]:
        with self._inner.in_use():
            return self._inner.get_schema_summary()

    def list_security_policies(self) -> SecurityPolicyReport:
        with self._inner.in_use():
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
        # The longest thing this wrapper does, and the one the deferred
        # close matters most for: a query outliving its connector's
        # eviction is the case that turned a leak into an aborted statement.
        with self._inner.in_use():
            return self._inner._execute_query(sql, timeout_seconds, max_rows)
