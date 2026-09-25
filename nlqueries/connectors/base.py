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


#: How many times `PermittedConnector._active` will take the guard before
#: giving up. Three, because the case it exists for is a rebuilt connector being
#: evicted before its holder can enter -- one more eviction than that, on one
#: request, is churn the caller should hear about rather than wait through.
_REOPEN_ATTEMPTS = 3


class ConnectorClosed(RuntimeError):
    """Raised when a caller reaches for a connector the cache has released.

    Distinct from the connectors' own "connect() must be called before use",
    which says the caller never connected. This says the opposite: it was
    connected, and the loader evicted it underneath — the caller should ask the
    loader for another rather than look at its own configuration, and the two
    are not distinguishable from the message the guard used to produce.
    """


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


def qualified_table_sql(name: str, schema: str | None, dialect: str | None = None) -> str:
    """*name* as a SQL table reference, schema-qualified and quoted for *dialect*.

    Every connector reports each table's schema (`TableSpec.schema`), and a query
    that names the table bare only works while the connection's default schema
    happens to hold it. On Snowflake it did not: `SELECT * FROM CALL_CENTER`
    failed where `TPCDS_SF10TCL.CALL_CENTER` worked, and the code sampling tables
    for column descriptions skipped every table without saying so.

    Quoted, because the names come from the catalogue exactly as stored -- an
    unquoted `CallCenter` folds to `callcenter` on Postgres and misses. *dialect*
    is a connector `db_type` or a sqlglot dialect name; without one, or with one
    sqlglot does not know, ANSI double quotes, which Postgres, Snowflake, DuckDB,
    Redshift, SQLite and SQL Server accept. MySQL and BigQuery need theirs, so
    pass it. An empty *schema* (a knowledge base written before the field
    existed) gives the bare, quoted name.
    """
    table, grammar = _table_expression(name, schema, dialect)
    return str(table.sql(dialect=grammar))


def table_sample_sql(name: str, schema: str | None, limit: int, dialect: str | None = None) -> str:
    """``SELECT * FROM <schema>.<table>`` bounded to *limit* rows, for *dialect*.

    The table is referenced as :func:`qualified_table_sql` does. The row bound is
    rendered by sqlglot rather than written as ``LIMIT``, which SQL Server does
    not accept: there it becomes ``TOP``.
    """
    from sqlglot import exp  # noqa: PLC0415

    table, grammar = _table_expression(name, schema, dialect)
    return str(exp.select(exp.Star()).from_(table).limit(limit).sql(dialect=grammar))


def _table_expression(name: str, schema: str | None, dialect: str | None) -> tuple[Any, str | None]:
    """The quoted table reference both helpers render, and the grammar to render it in.

    A *dialect* sqlglot does not know -- ``sqlalchemy``, the generic connector's
    ``db_type``, names no grammar -- renders as ANSI rather than raising, so a
    caller building SQL for such a connector gets the default quoting.
    """
    from sqlglot import exp  # noqa: PLC0415 -- keep sqlglot off this module's import path
    from sqlglot.dialects.dialect import Dialect  # noqa: PLC0415

    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    grammar = _sqlglot_dialect(dialect) if dialect else None
    if grammar is not None:
        try:
            Dialect.get_or_raise(grammar)
        except ValueError:
            grammar = None
    return exp.table_(name, db=schema or None, quoted=True), grammar


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
                "_use_state_pair",
                (threading.Lock(), {"count": 0, "deferred": False, "closing": False}),
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
            if state.get("closing"):
                # Refused rather than counted, and only for a connector whose
                # release destroys what it holds -- see `close`. The guard spans
                # one call, so an eviction landing between two of them finds the
                # count at zero and closes; and `open_connector_for_agent`
                # returns before the `asyncio.to_thread` hop into
                # `execute_query`, so that window is on the ordinary path.
                # Entering here after a close was requested also lets a statement
                # start between the moment `close` decides to release and the
                # release itself, which is the abort the deferral exists to
                # prevent.
                raise ConnectorClosed(
                    f"{type(self).__name__} was released by the connector cache; "
                    "ask the loader for a new one."
                )
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
            # Only when the release actually destroys something. A connector
            # holding a raw handle is finished once it is closed; one holding
            # only an engine is not, because `dispose()` returns the pool and
            # leaves the object usable — it builds a new pool on next use.
            #
            # Refusing for both was a regression, and a broad one: Postgres,
            # MSSQL and the generic SQLAlchemy connector are engine-only, so a
            # routine eviction — a TTL lapse in `_cache_get`, an LRU replacement
            # in `_cache_put`, a credential change calling
            # `invalidate_connector_cache` — would fail the next call made by
            # any request still holding that wrapper. All of those were harmless
            # for those connectors before this branch.
            #
            # Read now rather than at release: by then the attributes have been
            # cleared, and "does this connector hold a handle" would answer no
            # for exactly the connectors it needs to answer yes for.
            # Monotonic. Recomputing it from the live attributes lets the flag
            # fall back to False once `_release` has cleared them -- so a second
            # `close()`, or a subclass that closes its own handle before
            # delegating to `super().close()`, reopens the door on a connector
            # whose handle is already gone and the next caller gets "connect()
            # must be called before use", which is the message this flag exists
            # to replace. `EnterpriseRedshiftConnector` has exactly that shape
            # today, so it is a trap with a caller rather than only in theory.
            state["closing"] = state["closing"] or any(
                getattr(self, attr, None) is not None for attr in self._HANDLE_ATTRS
            )
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

    def __init__(
        self,
        inner: DatabaseConnector,
        policy: ExecutionPolicy,
        connector_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._execution_policy = policy
        #: Which connector this wrapper is for, so a released one can be rebuilt
        #: under the caller. The ID and not the agent: an agent's binding can be
        #: changed, and a caller that obtained this wrapper for one database must
        #: not silently be moved to another. None disables recovery, which is
        #: what a directly-constructed wrapper gets.
        self._connector_id = connector_id

    def _reopen(self) -> DatabaseConnector | None:
        """Rebuild the connector this wrapper is for. Imported late: the loader
        imports this module, so a module-level import would be circular."""
        if self._connector_id is None:
            return None
        from nlqueries.connectors.loader import reopen_connector  # noqa: PLC0415

        return reopen_connector(self._connector_id)

    @contextlib.contextmanager
    def _active(self) -> Iterator[DatabaseConnector]:
        """The inner connector, held open for one operation.

        Recovers from an eviction exactly once. `_cache_put` disposes connectors
        that callers already hold, so finding this one released is ordinary
        rather than exceptional -- and before the raw handles were released it
        cost nothing, because closing them did nothing. The rebuild happens here
        and not per call: `_load_connectors` parses the file every time, so
        resolving eagerly would put a YAML parse on every query.

        Fails closed. If the rebuild yields nothing the original `ConnectorClosed`
        is raised, never a silent fall back to the released connector -- which
        would mean querying with a credential the cache has already retired, and
        is the failure `invalidate_connector_cache` exists to prevent.
        """
        with contextlib.ExitStack() as stack:
            # Only entering the guard is guarded. An ExitStack rather than a
            # `with` so the `yield` sits outside the `try`: a `ConnectorClosed`
            # thrown in by the caller's body would otherwise be caught here and
            # answered with a second `yield`, which is a `generator didn't stop
            # after throw()` in place of the real error.
            for attempt in range(_REOPEN_ATTEMPTS):
                try:
                    stack.enter_context(self._inner.in_use())
                    break
                except ConnectorClosed:
                    # The replacement can be evicted too, in the window between
                    # `_reopen` returning it and the guard being taken. A single
                    # retry left that window open, and it is widest exactly when
                    # this path is busiest: after `invalidate_connector_cache`
                    # several holders rebuild at once and each one's `_cache_put`
                    # disposes the connector the previous one has just been
                    # handed -- the shape `test_twenty_threads_build_at_most_a_
                    # handful` already demonstrates for the open path.
                    #
                    # Bounded rather than `while True`: persistent churn should
                    # surface as a failed query rather than as a request that
                    # never returns, and re-raising the last `ConnectorClosed`
                    # keeps the fail-closed behaviour exactly as it was.
                    if attempt == _REOPEN_ATTEMPTS - 1:
                        raise
                    replacement = self._reopen()
                    if replacement is None:
                        raise
                    self._inner = replacement
            yield self._inner

    # -- delegation ------------------------------------------------------
    def connect(self, credentials: dict[str, Any]) -> None:
        self._inner.connect(credentials)

    def test_connection(self) -> bool:
        with self._active() as inner:
            return inner.test_connection()

    def extract_schema(self) -> SchemaSpec:
        with self._active() as inner:
            return inner.extract_schema()

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        with self._active() as inner:
            return inner.extract_query_history(days, limit)

    def get_schema_summary(self) -> tuple[int, int]:
        with self._active() as inner:
            return inner.get_schema_summary()

    def list_security_policies(self) -> SecurityPolicyReport:
        with self._active() as inner:
            return inner.list_security_policies()

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
        with self._active() as inner:
            return inner._execute_query(sql, timeout_seconds, max_rows)
