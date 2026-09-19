"""
nlqueries.connectors.redshift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Amazon Redshift implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Uses Amazon's ``redshift-connector`` pure-Python driver.  Install the optional
extra before use::

    pip install "nlqueries-core[redshift]"

Redshift is PostgreSQL-wire-protocol-compatible, so standard
``information_schema`` queries work.  Row counts are sourced from the
Redshift-specific ``SVV_TABLE_INFO`` system view; query history from
``STL_QUERY``.  Neither ``pg_description`` nor ``pg_class.reltuples`` are
available, so column/table descriptions and exact tuple counts are omitted.
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)

_SYSTEM_SCHEMAS = frozenset(
    {
        "information_schema",
        "pg_catalog",
        "pg_internal",
        "pg_toast",
        "pg_temp_1",
        "pg_bitmapindex",
    }
)


class RedshiftConnector(DatabaseConnector):
    """Connector for Amazon Redshift.

    Usage::

        connector = RedshiftConnector()
        connector.connect({
            "host": "my-cluster.abc123.us-east-1.redshift.amazonaws.com",
            "port": 5439,
            "database": "dev",
            "user": "awsuser",
            "password": "YOUR_PASSWORD",
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._conn: Any = None
        self._database: str = ""

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Open a Redshift connection from ``credentials``.

        Required keys: ``host``, ``database``, ``user``, ``password``.
        ``port`` defaults to ``5439``.
        """
        try:
            import redshift_connector  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'redshift-connector' package is required for Redshift connections.\n"
                "Install it with:  pip install 'nlqueries-core[redshift]'"
            ) from exc

        self._database = credentials["database"]
        self._conn = redshift_connector.connect(
            host=credentials.get("host", "localhost"),
            port=int(credentials.get("port") or 5439),
            database=self._database,
            user=credentials.get("user", ""),
            password=credentials.get("password", ""),
            ssl=True,
            # Named a connect timeout by the driver, but it calls settimeout
            # once on the socket and never clears it, so this bounds every
            # later read too. It therefore has to sit above the statement
            # timeout, or a long query dies here instead of being cancelled by
            # the server. See config.REDSHIFT_SOCKET_TIMEOUT_SECONDS.
            timeout=config.REDSHIFT_SOCKET_TIMEOUT_SECONDS or None,
        )

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("RedshiftConnector.connect() must be called before use.")
        return self._conn

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds against the cluster."""
        try:
            with contextlib.closing(self._require_conn().cursor()) as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("RedshiftConnector.test_connection failed")
            return False
        finally:
            # `SELECT 1` opens a transaction like any other statement, and the
            # connection outlives this call -- `loader.py` caches connectors
            # across requests. Left open, what fails is the next query's
            # `SET TRANSACTION READ ONLY`, for a reason the user cannot see.
            # `contextlib.closing` also closes the cursor on the failure path,
            # which the earlier `cur.close()` inside the `try` did not.
            self._end_transaction()

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via ``information_schema`` and ``SVV_TABLE_INFO``.

        Returns a :class:`SchemaSpec` for every user table.  Row counts come
        from ``SVV_TABLE_INFO.tbl_rows`` (requires table-owner or superuser);
        a ``None`` count is returned if the view is inaccessible.
        Column/table descriptions are not available in Redshift and are
        returned as ``None``.
        """
        conn = self._require_conn()
        try:
            row_counts = self._fetch_row_counts(conn)
            cols_by_table = self._fetch_columns(conn)
            # Keys are an enrichment, tables and columns are the schema. The
            # constraint views are restricted on a datashare consumer database, and
            # `_fetch_row_counts` above already falls back rather than failing for
            # the same reason -- this extends that to the rest of the enrichment,
            # so a database that can describe its tables is not refused outright.
            pks = self._enrichment("primary keys", lambda: self._fetch_primary_keys(conn), {})
            fks = self._enrichment("foreign keys", lambda: self._fetch_foreign_keys(conn), {})

            tables: list[TableSpec] = []
            for (schema, name), row_count in row_counts.items():
                key = (schema, name)
                pk_cols = pks.get(key, set())
                fk_cols = fks.get(key, {})
                columns = [
                    ColumnSpec(
                        name=col["column_name"],
                        type=col["data_type"],
                        nullable=col["is_nullable"],
                        is_primary_key=col["column_name"] in pk_cols,
                        is_foreign_key=col["column_name"] in fk_cols,
                        references=fk_cols.get(col["column_name"]),
                        description=None,
                    )
                    for col in cols_by_table.get(key, [])
                ]
                tables.append(
                    TableSpec(
                        name=name,
                        schema=schema,
                        row_count=row_count,
                        columns=columns,
                        description=None,
                    )
                )

            return SchemaSpec(
                database=self._database,
                tables=tables,
                extracted_at=_utc_now_iso(),
            )
        finally:
            # Every `cur.execute` above opened a transaction implicitly, and
            # nothing here commits or rolls back -- so on the way out of a
            # *successful* extract the connection still carries an open
            # transaction. `_execute_query` opens the next user query with
            # `SET TRANSACTION READ ONLY`, which the server only accepts as
            # the first statement of a transaction, so that query would fail
            # on a reused connection. The failure paths roll back for their
            # own reasons; this covers the ones that succeed, and the raise.
            self._end_transaction()

    @staticmethod
    def _fetch_row_counts(conn: Any) -> dict[tuple[str, str], int | None]:
        """Return ``{(schema, table): row_count}`` from ``SVV_TABLE_INFO``.

        Falls back to a plain ``information_schema`` list (with ``None`` counts)
        if the user lacks access to ``SVV_TABLE_INFO`` -- which is the common
        case, since reading it needs a grant most analyst roles are not given.

        The fallback runs on a fresh cursor, after rolling back. `Cursor.execute`
        opens an explicit transaction when autocommit is off, so the failed
        `SVV_TABLE_INFO` statement leaves the block aborted; reusing the same
        cursor inside it made the fallback query fail with 25P02 and propagate,
        so the documented fallback never actually ran and a role without that
        grant got no schema at all.
        """
        with contextlib.closing(conn.cursor()) as cur:
            try:
                cur.execute(
                    """
                    SELECT schema, "table", tbl_rows
                    FROM SVV_TABLE_INFO
                    WHERE schema NOT IN (
                        'information_schema','pg_catalog','pg_internal',
                        'pg_toast','pg_temp_1','pg_bitmapindex'
                    )
                    ORDER BY schema, "table"
                    """
                )
                return {
                    (r[0], r[1]): (int(r[2]) if r[2] is not None else None) for r in cur.fetchall()
                }
            except Exception as exc:  # noqa: BLE001 — falls back, and logs
                logger.warning(
                    "RedshiftConnector: could not read SVV_TABLE_INFO (%s). Listing "
                    "tables without row counts; this usually means the role lacks the "
                    "grant that view requires.",
                    exc,
                )

        # Outside the `with`, and after the rollback: the aborted block has to be
        # ended before anything else will run on this connection.
        with contextlib.suppress(Exception):
            conn.rollback()

        # Rolled back on the way out too. If this one fails as well -- a dropped
        # connection, or a role that cannot read `information_schema.tables`
        # either -- the exception propagates with the block still aborted, and
        # the next `execute_query` on this connector then fails at
        # `SET TRANSACTION READ ONLY` for a reason that has nothing to do with
        # the query the user ran.
        try:
            with contextlib.closing(conn.cursor()) as cur:
                cur.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                      AND table_schema NOT IN (
                          'information_schema','pg_catalog','pg_internal',
                          'pg_toast','pg_temp_1','pg_bitmapindex'
                      )
                    ORDER BY table_schema, table_name
                    """
                )
                return {(r[0], r[1]): None for r in cur.fetchall()}
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise

    @staticmethod
    def _fetch_columns(conn: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT table_schema, table_name, column_name, data_type,
                   (is_nullable = 'YES') AS is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN (
                'information_schema','pg_catalog','pg_internal',
                'pg_toast','pg_temp_1','pg_bitmapindex'
            )
            ORDER BY table_schema, table_name, ordinal_position
            """
        )
        result: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in cur.fetchall():
            key = (r[0], r[1])
            result.setdefault(key, []).append(
                {"column_name": r[2], "data_type": r[3], "is_nullable": bool(r[4])}
            )
        cur.close()
        return result

    def _enrichment(self, label: str, fetch: Any, default: Any) -> Any:
        """Run an optional schema enrichment, or roll back, log and return *default*.

        Broad on purpose, matching `_fetch_row_counts`'s existing fallback: the
        driver raises a plain `Exception` subclass for both "no such view" and
        "permission denied", and the two are one outcome here. The warning
        carries the reason, because the cost of a broad catch is that a real
        fault becomes a quieter schema rather than a failure -- so it has to be
        visible in the log.

        The rollback is the part that is easy to miss. `Cursor.execute` opens an
        explicit transaction whenever one is not already open and autocommit is
        off, which is the default -- so a statement that fails leaves the block
        in the aborted state. Swallowing the error without ending it makes every
        later statement on this connection fail with 25P02: the foreign-key
        fetch below would be logged as though the datashare caused it, and the
        `SET TRANSACTION READ ONLY` that opens the next user query would fail
        for no reason that reaches the user.
        """
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 — degraded, rolled back, and logged
            self._end_transaction()
            logger.warning(
                "RedshiftConnector: could not read %s (%s). The schema will omit them; "
                "this is expected on a database imported from a datashare.",
                label,
                exc,
            )
            return default

    @staticmethod
    def _fetch_primary_keys(conn: Any) -> dict[tuple[str, str], set[str]]:
        # `closing`, not a `finally` around the fetch. The driver reports a
        # missing or restricted view from `execute`, so a guard beginning after
        # it covered only the path that does not fail -- and with the caller now
        # swallowing the error, every degraded extraction leaked a cursor.
        with contextlib.closing(conn.cursor()) as cur:
            cur.execute(
                """
                SELECT tc.table_schema, tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON kcu.constraint_name  = tc.constraint_name
                   AND kcu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema NOT IN (
                      'information_schema','pg_catalog','pg_internal'
                  )
                """
            )
            result: dict[tuple[str, str], set[str]] = {}
            for r in cur.fetchall():
                result.setdefault((r[0], r[1]), set()).add(r[2])
        return result

    @staticmethod
    def _fetch_foreign_keys(conn: Any) -> dict[tuple[str, str], dict[str, str]]:
        with contextlib.closing(conn.cursor()) as cur:
            cur.execute(
                """
                SELECT tc.table_schema, tc.table_name, kcu.column_name,
                       ccu.table_name AS ref_table, ccu.column_name AS ref_col
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON kcu.constraint_name  = tc.constraint_name
                   AND kcu.constraint_schema = tc.constraint_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name  = tc.constraint_name
                   AND ccu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema NOT IN (
                      'information_schema','pg_catalog','pg_internal'
                  )
                """
            )
            result: dict[tuple[str, str], dict[str, str]] = {}
            for r in cur.fetchall():
                result.setdefault((r[0], r[1]), {})[r[2]] = f"{r[3]}.{r[4]}"
        return result

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return top queries from ``STL_QUERY``, grouped by query text.

        Requires access to ``STL_QUERY`` (superuser or granted via
        ``pg_read_all_stats``).  Returns an empty list if the view is
        inaccessible or the user lacks permission.
        """
        conn = self._require_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT
                    TRIM(querytxt)                              AS query,
                    COUNT(*)                                    AS execution_count,
                    AVG(DATEDIFF(ms, starttime, endtime))       AS avg_duration_ms,
                    MAX(starttime)::text                        AS last_executed
                FROM STL_QUERY
                WHERE userid > 1
                  AND aborted = 0
                  AND starttime >= DATEADD(day, %s, GETDATE())
                  AND querytxt NOT LIKE 'PADB%%'
                  AND querytxt NOT LIKE 'SET %%'
                  AND querytxt NOT LIKE 'BEGIN%%'
                  AND querytxt NOT LIKE 'COMMIT%%'
                  AND LEN(TRIM(querytxt)) > 10
                GROUP BY TRIM(querytxt)
                ORDER BY execution_count DESC
                LIMIT %s
                """,
                (-days, limit),
            )
            rows = cur.fetchall()
        except Exception:  # noqa: BLE001
            logger.warning(
                "RedshiftConnector.extract_query_history: could not query STL_QUERY "
                "(requires superuser or pg_read_all_stats). Returning empty history."
            )
            rows = []
        finally:
            cur.close()
            # The `STL_QUERY` statement opened a transaction, and the swallowed
            # failure leaves the block aborted on top of that. Same reasoning as
            # `extract_schema`: the connection is reused, so this must not be
            # the next caller's problem.
            self._end_transaction()

        return [
            QueryRecord(
                sql=str(r[0]),
                execution_count=int(r[1]),
                avg_duration_ms=float(r[2]) if r[2] is not None else None,
                last_executed=str(r[3]) if r[3] is not None else None,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # execute_query
    # ------------------------------------------------------------------

    def _end_transaction(self) -> None:
        """Roll back, so the next query can open a read-only transaction.

        Nothing is ever committed here: the transaction is read-only, so there
        is nothing to keep. Failures are ignored because the connection may
        already be gone, and this runs on the error path too.
        """
        with contextlib.suppress(Exception):
            self._require_conn().rollback()

    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute ``sql`` in a read-only transaction and return a :class:`QueryResult`.

        *timeout_seconds* bounds the query, falling back to
        ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS``.

        Both guards were measured against Redshift Serverless (which reports
        itself as PostgreSQL 8.0.2):

        - ``SET TRANSACTION READ ONLY`` is accepted, ``SELECT`` still runs, and
          ``INSERT`` is refused with SQLSTATE 25006, ``transaction is
          read-only``. ``BEGIN READ ONLY`` behaves identically; this form is used
          because it matches the PostgreSQL connector.
        - ``SET statement_timeout TO`` milliseconds is accepted, and a query
          exceeding it is cancelled with SQLSTATE 57014. At 2000 ms a cross join
          was cancelled after 2.2 s; the same query ran for 60.2 s at 60000 ms.

        The transaction is closed after every query. ``SET TRANSACTION`` applies
        to the transaction it opens and must be the first statement in it, so a
        transaction left open by the previous query would prevent the next one
        being made read-only. Verified over a reused connection: three
        consecutive queries each read successfully and each refused a write.

        Exceptions are caught and surfaced via ``QueryResult.error``.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
        )
        start = time.perf_counter()
        try:
            conn = self._require_conn()
            cur = conn.cursor()
            cur.execute("SET TRANSACTION READ ONLY")
            if effective_timeout is not None and effective_timeout > 0:
                cur.execute(f"SET statement_timeout TO {max(1, int(effective_timeout * 1000))}")
            cur.execute(sql)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _truncated, _reason = False, None
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows, _truncated, _reason = collect(cur, max_rows)
            else:
                columns, rows = [], []
            cur.close()
            self._end_transaction()
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=_truncated,
                truncation_reason=_reason,
                execution_time_ms=elapsed_ms,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("RedshiftConnector.execute_query failed")
            self._end_transaction()
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
