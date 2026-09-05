"""
nlqueries.connectors.snowflake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Snowflake implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on the official ``snowflake-connector-python`` driver. This module is
part of the public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import snowflake.connector

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    POLICY_COLUMN,
    POLICY_ROW,
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    SecurityPolicy,
    SecurityPolicyReport,
    TableSpec,
)

logger = logging.getLogger(__name__)

# Schemas that are part of Snowflake / the catalog itself, never user data.
_SYSTEM_SCHEMAS = ("INFORMATION_SCHEMA",)


#: Longest a query waits for the connector's transaction lock when no statement
#: timeout applies. With a timeout, the wait is derived from it: whoever holds
#: the lock cannot legitimately outlast their own timeout by much, so waiting a
#: little longer and then failing is the honest outcome.
#:
#: Without one -- `CONNECTOR_STATEMENT_TIMEOUT_SECONDS=0`, which is documented
#: and supported -- there is nothing bounding the holder, so this bounds the
#: waiter instead. An operator who disables the statement timeout is saying a
#: *query* may run unbounded; they are not asking for every later query on the
#: same connector to queue behind it indefinitely on a pool thread.
_LOCK_WAIT_WITHOUT_TIMEOUT_SECONDS = 300.0

#: Added to the statement timeout when deriving the lock wait, to cover the row
#: fetch and the ROLLBACK that happen inside the span after the query returns.
_LOCK_WAIT_MARGIN_SECONDS = 30.0


class SnowflakeConnector(DatabaseConnector):
    """Connector for Snowflake.

    Usage::

        connector = SnowflakeConnector()
        connector.connect({
            "account": "acme-prod",
            "user": "alice",
            "password": "YOUR_PASSWORD",
            "warehouse": "COMPUTE_WH",
            "database": "ANALYTICS",
            "schema": "PUBLIC",       # optional
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._connection: Any = None
        #: Serialises the BEGIN ... ROLLBACK span below. A Snowflake transaction
        #: is scoped to the *session*, not the cursor, and this connector holds
        #: one connection for its lifetime while `loader.py` caches instances and
        #: callers arrive through `asyncio.to_thread`. Two overlapping queries
        #: would otherwise interleave as BEGIN(A), BEGIN(B) -- ignored, a
        #: transaction is already open -- query(A), ROLLBACK(A), leaving B
        #: running under autocommit with nothing left to roll back and its own
        #: ROLLBACK a no-op. The guard would vanish precisely where it is
        #: documented to hold.
        self._txn_lock = threading.Lock()
        self._database: str | None = None
        self._db_schema: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Open a Snowflake connection from ``credentials``.

        Required keys: ``account``, ``user``, ``password``, ``warehouse``,
        ``database``. ``schema`` is optional.
        """
        self._database = credentials["database"]
        self._db_schema = credentials.get("schema")

        connect_kwargs: dict[str, Any] = {
            "account": credentials["account"],
            "user": credentials.get("user"),
            "password": credentials.get("password"),
            "warehouse": credentials.get("warehouse"),
            "database": self._database,
        }
        if self._db_schema:
            connect_kwargs["schema"] = self._db_schema

        self._connection = snowflake.connector.connect(**connect_kwargs)

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("SnowflakeConnector.connect() must be called before use.")
        return self._connection

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query(connection: Any, sql: str) -> list[dict[str, Any]]:
        """Run ``sql`` and return rows as a list of ``{column_name: value}`` dicts.

        Snowflake's driver returns column names in upper case by default
        (identifiers are folded to upper case unless quoted), so callers
        should look up keys like ``"TABLE_NAME"`` rather than ``"table_name"``.
        """
        cursor = connection.cursor(snowflake.connector.DictCursor)
        try:
            cursor.execute(sql)
            return list(cursor.fetchall())
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT CURRENT_VERSION()`` succeeds against Snowflake."""
        try:
            connection = self._require_connection()
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT CURRENT_VERSION()")
                cursor.fetchone()
            finally:
                cursor.close()
            return True
        except Exception:  # noqa: BLE001 — any failure means "not connected"
            logger.exception("SnowflakeConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via ``INFORMATION_SCHEMA``.

        Builds a full :class:`SchemaSpec` describing every base table outside
        the ``INFORMATION_SCHEMA`` schema: its columns, primary keys, foreign
        keys (flagged — see note below), and row count.

        Row counts come from ``INFORMATION_SCHEMA.TABLES.ROW_COUNT``.
        ``TABLE_STORAGE_METRICS`` only exposes storage-billing metrics
        (active/time-travel/fail-safe byte counts); it has no row-count
        column, so it cannot serve as a row-count source.

        Note: primary/foreign-key *membership* is derived from
        ``TABLE_CONSTRAINTS`` joined with ``KEY_COLUMN_USAGE``. Resolving
        *which* table/column a foreign key references would additionally
        require ``REFERENTIAL_CONSTRAINTS`` (to map the FK constraint to its
        referenced unique/primary-key constraint) — Snowflake has no
        ``CONSTRAINT_COLUMN_USAGE`` equivalent. That view is intentionally
        out of scope here, so ``ColumnSpec.references`` is left ``None`` for
        foreign-key columns.
        """
        connection = self._require_connection()
        database = self._database or ""

        tables_meta = self._fetch_tables(connection, database)
        columns_by_table = self._fetch_columns(connection, database)
        primary_keys, foreign_keys = self._fetch_constraint_columns(connection, database)

        tables: list[TableSpec] = []
        for (schema_name, table_name), meta in tables_meta.items():
            key = (schema_name, table_name)
            pk_columns = primary_keys.get(key, set())
            fk_columns = foreign_keys.get(key, set())

            columns = [
                ColumnSpec(
                    name=col["COLUMN_NAME"],
                    type=col["DATA_TYPE"],
                    nullable=col["IS_NULLABLE"] == "YES",
                    is_primary_key=col["COLUMN_NAME"] in pk_columns,
                    is_foreign_key=col["COLUMN_NAME"] in fk_columns,
                    references=None,
                    description=col["COMMENT"],
                )
                for col in columns_by_table.get(key, [])
            ]

            tables.append(
                TableSpec(
                    name=table_name,
                    schema=schema_name,
                    row_count=meta["row_count"],
                    columns=columns,
                    description=meta["description"],
                )
            )

        return SchemaSpec(
            database=database,
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    @classmethod
    def _fetch_tables(cls, connection: Any, database: str) -> dict[tuple[str, str], dict[str, Any]]:
        """Return ``{(schema, table): {row_count, description}}`` from ``TABLES``."""
        rows = cls._query(
            connection,
            f"""
            SELECT table_schema, table_name, row_count, comment
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ({_in_clause(_SYSTEM_SCHEMAS)})
            ORDER BY table_schema, table_name
            """,
        )
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            row_count = row["ROW_COUNT"]
            result[(row["TABLE_SCHEMA"], row["TABLE_NAME"])] = {
                "row_count": int(row_count) if row_count is not None else None,
                "description": row["COMMENT"],
            }
        return result

    @classmethod
    def _fetch_columns(
        cls, connection: Any, database: str
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return ``{(schema, table): [column dicts in ordinal order]}`` from ``COLUMNS``."""
        rows = cls._query(
            connection,
            f"""
            SELECT table_schema, table_name, column_name, data_type, is_nullable, comment
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema NOT IN ({_in_clause(_SYSTEM_SCHEMAS)})
            ORDER BY table_schema, table_name, ordinal_position
            """,
        )
        result: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (row["TABLE_SCHEMA"], row["TABLE_NAME"])
            result.setdefault(key, []).append(row)
        return result

    @classmethod
    def _fetch_constraint_columns(
        cls, connection: Any, database: str
    ) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
        """Return ``(primary_key_columns, foreign_key_columns)``, each keyed by ``(schema, table)``.

        Joins ``TABLE_CONSTRAINTS`` to ``KEY_COLUMN_USAGE`` on constraint
        identity to determine which columns participate in a PRIMARY KEY or
        FOREIGN KEY constraint.
        """
        rows = cls._query(
            connection,
            f"""
            SELECT tc.table_schema, tc.table_name, tc.constraint_type, kcu.column_name
            FROM {database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN {database}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON kcu.constraint_name = tc.constraint_name
               AND kcu.constraint_schema = tc.constraint_schema
               AND kcu.table_name = tc.table_name
            WHERE tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
              AND tc.table_schema NOT IN ({_in_clause(_SYSTEM_SCHEMAS)})
            """,
        )
        primary_keys: dict[tuple[str, str], set[str]] = {}
        foreign_keys: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            key = (row["TABLE_SCHEMA"], row["TABLE_NAME"])
            bucket = primary_keys if row["CONSTRAINT_TYPE"] == "PRIMARY KEY" else foreign_keys
            bucket.setdefault(key, set()).add(row["COLUMN_NAME"])
        return primary_keys, foreign_keys

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return the top queries (by execution count) from the last ``days`` days.

        Tries ``SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`` first (account-wide,
        up to one year of retention, requires the ``ACCOUNT_USAGE`` schema to
        be accessible to the current role). If that fails — e.g. the role
        lacks the ``IMPORTED PRIVILEGES`` grant on the ``SNOWFLAKE``
        database — falls back to ``INFORMATION_SCHEMA.QUERY_HISTORY``, which
        is always accessible but only retains the last 7 days and is scoped
        to the current account/session context.

        Returns up to ``limit`` records, ordered by execution count descending.
        Returns an empty list (with a logged warning) if both sources are
        inaccessible.
        """
        connection = self._require_connection()

        try:
            return self._fetch_query_history_account_usage(connection, days, limit)
        except Exception:
            logger.warning(
                "extract_query_history: SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY is not "
                "accessible — falling back to INFORMATION_SCHEMA.QUERY_HISTORY "
                "(requires the role to have IMPORTED PRIVILEGES on the SNOWFLAKE "
                "database for full account-usage history).",
                exc_info=True,
            )

        try:
            return self._fetch_query_history_information_schema(connection, days, limit)
        except Exception:
            logger.exception(
                "extract_query_history: INFORMATION_SCHEMA.QUERY_HISTORY is also "
                "inaccessible — returning an empty query history."
            )
            return []

    @classmethod
    def _fetch_query_history_account_usage(
        cls, connection: Any, days: int, limit: int
    ) -> list[QueryRecord]:
        rows = cls._query(
            connection,
            f"""
            SELECT
                query_text,
                COUNT(*) AS execution_count,
                AVG(total_elapsed_time) AS avg_duration_ms,
                MAX(start_time) AS last_executed
            FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
            WHERE start_time >= DATEADD('day', -{int(days)}, CURRENT_TIMESTAMP())
            GROUP BY query_text
            ORDER BY execution_count DESC
            LIMIT {limit}
            """,
        )
        return [cls._row_to_query_record(row) for row in rows]

    @classmethod
    def _fetch_query_history_information_schema(
        cls, connection: Any, days: int, limit: int
    ) -> list[QueryRecord]:
        rows = cls._query(
            connection,
            f"""
            SELECT
                query_text,
                COUNT(*) AS execution_count,
                AVG(total_elapsed_time) AS avg_duration_ms,
                MAX(start_time) AS last_executed
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
                END_TIME_RANGE_START => DATEADD('day', -{int(days)}, CURRENT_TIMESTAMP()),
                RESULT_LIMIT => 10000
            ))
            GROUP BY query_text
            ORDER BY execution_count DESC
            LIMIT {limit}
            """,
        )
        return [cls._row_to_query_record(row) for row in rows]

    @staticmethod
    def _row_to_query_record(row: dict[str, Any]) -> QueryRecord:
        avg_duration = row["AVG_DURATION_MS"]
        last_executed = row["LAST_EXECUTED"]
        return QueryRecord(
            sql=row["QUERY_TEXT"],
            execution_count=int(row["EXECUTION_COUNT"]),
            avg_duration_ms=float(avg_duration) if avg_duration is not None else None,
            last_executed=str(last_executed) if last_executed is not None else None,
        )

    # ------------------------------------------------------------------
    # execute_query
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute ``sql`` and return a :class:`QueryResult`.

        The query is bounded by *timeout_seconds* when given, else the
        ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS`` default — passed to the driver's
        ``cursor.execute(timeout=…)`` so Snowflake cancels the query server-side
        rather than letting it run indefinitely. A budget of 0 disables it.

        Any exception raised during execution is caught and surfaced via
        ``QueryResult.error`` rather than propagating, so callers can treat
        query execution as always returning a result object.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
        )
        start = time.perf_counter()
        try:
            connection = self._require_connection()
            # Held across the whole span, not just BEGIN: the transaction belongs
            # to the session, so releasing it before the ROLLBACK would let a
            # second query run inside this one's transaction. The cost is that
            # queries on one Snowflake connector serialise; a shared session
            # cannot give both concurrency and a per-query transaction, and a
            # guard that silently stops holding under load is worth less than the
            # throughput.
            # Bounded, so a wedged session degrades to a failed query rather
            # than blocking every later query on this connector. The holder
            # cannot legitimately outlast its own statement timeout by more than
            # the fetch and the ROLLBACK, so that plus a margin is the wait.
            lock_wait = (
                effective_timeout + _LOCK_WAIT_MARGIN_SECONDS
                if effective_timeout is not None and effective_timeout > 0
                else _LOCK_WAIT_WITHOUT_TIMEOUT_SECONDS
            )
            if not self._txn_lock.acquire(timeout=lock_wait):
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "SnowflakeConnector: gave up after %.0fs waiting for the "
                    "connector's transaction lock. A Snowflake transaction is "
                    "session-scoped and this connector holds one session, so "
                    "queries on it run one at a time.",
                    lock_wait,
                )
                return QueryResult(
                    columns=[],
                    rows=[],
                    row_count=0,
                    execution_time_ms=elapsed_ms,
                    error=(
                        f"The Snowflake connector was busy for {lock_wait:.0f}s and the "
                        f"query was not run. Queries on one connector run one at a time, "
                        f"because a Snowflake transaction belongs to the session rather "
                        f"than the cursor."
                    ),
                )
            try:
                cursor = connection.cursor()
                try:
                    # An explicit transaction that is never committed.
                    #
                    # Snowflake has no read-only transaction mode to ask for, and by
                    # default each statement autocommits -- so a write that reached
                    # here was permanent the moment it ran. Every validator in front
                    # of this one asks only whether the root node is a Select, and
                    # `SELECT some_volatile_function(...)` satisfies that while still
                    # writing; an audit reproduced exactly that shape through another
                    # connector twice, eight weeks apart.
                    #
                    # BEGIN turns autocommit off for what follows, and the ROLLBACK in
                    # the `finally` below undoes it whether the statement succeeded or
                    # not. DDL is not transactional on Snowflake, so a `CREATE` or
                    # `DROP` still stands: the control for that is a role holding only
                    # USAGE and SELECT, which only the operator can grant. See
                    # docs/database-hardening.md.
                    cursor.execute("BEGIN")
                    if effective_timeout is not None and effective_timeout > 0:
                        cursor.execute(sql, timeout=max(1, int(effective_timeout)))
                    else:
                        cursor.execute(sql)
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    _truncated, _reason = False, None
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        rows, _truncated, _reason = collect(cursor, max_rows)
                    else:
                        columns = []
                        rows = []

                    return QueryResult(
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        truncated=_truncated,
                        truncation_reason=_reason,
                        execution_time_ms=elapsed_ms,
                        error=None,
                    )
                finally:
                    # Before the cursor closes, and on the success path too: the
                    # point is that a statement which *worked* is still undone.
                    #
                    # Logged, not suppressed. A failed rollback must not replace
                    # the query's own error -- that is why it is caught -- but the
                    # justification used in `mssql.py` and
                    # `sqlalchemy_connector.py`, that the connection is reset when
                    # the pool takes it back, does not hold here: this connector
                    # keeps one session for its lifetime, so a rollback that fails
                    # can leave an explicit transaction open on that session until
                    # some later query happens to end it. Nothing is committed, so
                    # it is a visibility gap rather than an exposure -- which is
                    # exactly the kind that should not be silent.
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "SnowflakeConnector: ROLLBACK after the query failed. "
                            "Nothing was committed, but this connector holds one "
                            "session, so a transaction may remain open on it until "
                            "the next query ends it.",
                            exc_info=True,
                        )
                    cursor.close()
            finally:
                # Paired with the bounded `acquire` above. In a `finally` so the
                # next caller is not left waiting out its whole timeout because
                # this one raised on the way through.
                self._txn_lock.release()
        except Exception as exc:  # noqa: BLE001 — surfaced via QueryResult.error, not raised
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("SnowflakeConnector.execute_query failed")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )

    def list_security_policies(self) -> SecurityPolicyReport:
        """Introspect Snowflake row-access + masking policies via ``ACCOUNT_USAGE``.

        Uses ``SNOWFLAKE.ACCOUNT_USAGE.POLICY_REFERENCES``, which lists every place
        a policy is attached, scoped here to the connected database. A **masking
        policy** maps cleanly to a column exclusion, so it becomes a
        :data:`POLICY_COLUMN` policy naming the masked column. A **row-access
        policy** is an opaque SQL function whose body we do not fetch here, so it
        becomes a :data:`POLICY_ROW` policy with an empty ``expression`` — the
        caller treats an empty expression as "cannot translate, review manually",
        which is the safe default for a security policy (never guess a predicate).

        Reading ``ACCOUNT_USAGE`` needs ``IMPORTED PRIVILEGES`` on the ``SNOWFLAKE``
        database; without it (or on any read error) this degrades to an empty but
        supported report rather than raising.

        NOTE: pending validation against a live Snowflake account (Block G entry
        criterion) — the ``ACCOUNT_USAGE`` column names follow Snowflake's
        documented ``POLICY_REFERENCES`` view.
        """
        connection = self._require_connection()
        database = self._database or ""
        try:
            rows = self._query(
                connection,
                "SELECT POLICY_NAME, POLICY_KIND, REF_SCHEMA_NAME, REF_ENTITY_NAME, "
                "REF_COLUMN_NAME "
                "FROM SNOWFLAKE.ACCOUNT_USAGE.POLICY_REFERENCES "
                f"WHERE REF_DATABASE_NAME = '{database}'",
            )
        except Exception:  # noqa: BLE001 — ACCOUNT_USAGE may be ungranted; degrade to empty
            logger.warning(
                "SnowflakeConnector.list_security_policies: POLICY_REFERENCES unavailable "
                "(needs IMPORTED PRIVILEGES on the SNOWFLAKE database).",
                exc_info=True,
            )
            return SecurityPolicyReport(supported=True, policies=[])

        policies: list[SecurityPolicy] = []
        for row in rows:
            schema = str(row.get("REF_SCHEMA_NAME") or "")
            if schema in _SYSTEM_SCHEMAS:
                continue
            is_masking = str(row.get("POLICY_KIND") or "").upper() == "MASKING_POLICY"
            column = row.get("REF_COLUMN_NAME")
            policies.append(
                SecurityPolicy(
                    name=str(row.get("POLICY_NAME") or ""),
                    kind=POLICY_COLUMN if is_masking else POLICY_ROW,
                    table_schema=schema,
                    table=str(row.get("REF_ENTITY_NAME") or ""),
                    columns=[str(column)] if (is_masking and column) else [],
                    # bodies aren't fetched → the caller marks row policies untranslatable
                    expression="",
                    roles=[],
                )
            )
        return SecurityPolicyReport(supported=True, policies=policies)


def _in_clause(values: tuple[str, ...]) -> str:
    """Render ``('A', 'B', ...)`` -> ``'A', 'B', ...`` for use inside a SQL ``IN (...)``."""
    return ", ".join(f"'{value}'" for value in values)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
