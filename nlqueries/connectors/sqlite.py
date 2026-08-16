# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.sqlite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLite implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Uses the standard-library ``sqlite3`` driver, so it needs no optional extra.
Like DuckDB, a SQLite "connection" is just a file path (``/data/app.db``) or the
special string ``:memory:`` for an in-process transient database — there is no
host, port, or credentials. It is handy for local / dev analytics and for
evaluating NLQueries against a self-contained file.

SQLite keeps no persistent query history, so ``extract_query_history`` always
returns an empty list.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
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


class SQLiteConnector(DatabaseConnector):
    """Connector for SQLite.

    Usage::

        connector = SQLiteConnector()

        # File-based database:
        connector.connect({"database": "/data/app.db"})

        # In-memory (transient, good for testing):
        connector.connect({"database": ":memory:"})

        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._database: str = ":memory:"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Open a SQLite connection.

        The only recognised credential key is ``database``: a file-system path
        or ``:memory:`` (default when omitted or empty). Other keys (``host``,
        ``port``, ``user``, ``password``) are accepted and ignored so the same
        credential dict shape as server connectors works unchanged.

        ``check_same_thread=False`` lets the timeout watchdog call ``interrupt``
        from its thread (and keeps the connector usable across Celery worker
        threads); access is otherwise single-threaded per query.
        """
        self._database = str(credentials.get("database") or ":memory:")
        self._conn = sqlite3.connect(self._database, check_same_thread=False)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteConnector.connect() must be called before use.")
        return self._conn

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds."""
        try:
            self._require_conn().execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("SQLiteConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via SQLite ``PRAGMA`` calls.

        Columns and primary keys come from ``PRAGMA table_info``, foreign keys
        from ``PRAGMA foreign_key_list``, and row counts from ``COUNT(*)`` (SQLite
        keeps no cheap row-count estimate). Internal ``sqlite_*`` tables are
        skipped. Every table lives in the single implicit ``main`` schema.
        """
        conn = self._require_conn()
        tables: list[TableSpec] = []
        for name in self._table_names(conn):
            fk_by_col = self._foreign_keys(conn, name)
            columns = [
                ColumnSpec(
                    name=str(col["name"]),
                    type=str(col["type"] or ""),
                    nullable=not bool(col["notnull"]),
                    is_primary_key=bool(col["pk"]),
                    is_foreign_key=str(col["name"]) in fk_by_col,
                    references=fk_by_col.get(str(col["name"])),
                    description=None,
                )
                for col in self._columns(conn, name)
            ]
            tables.append(
                TableSpec(
                    name=name,
                    schema="main",
                    row_count=self._row_count(conn, name),
                    columns=columns,
                    description=None,
                )
            )

        return SchemaSpec(
            database=self._database,
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    @staticmethod
    def _table_names(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [str(r[0]) for r in rows]

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        # PRAGMA takes no bind params; quote the identifier from the trusted catalog.
        rows = conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
        # table_info columns: (cid, name, type, notnull, dflt_value, pk)
        return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in rows]

    @staticmethod
    def _foreign_keys(conn: sqlite3.Connection, table: str) -> dict[str, str]:
        """Return ``{local_column: "ref_table.ref_column"}`` for *table*'s FKs."""
        rows = conn.execute(f"PRAGMA foreign_key_list({_quote_ident(table)})").fetchall()
        # foreign_key_list columns: (id, seq, table, from, to, on_update, on_delete, match)
        fks: dict[str, str] = {}
        for r in rows:
            ref_table, from_col, to_col = r[2], r[3], r[4]
            if from_col:
                fks[str(from_col)] = f"{ref_table}.{to_col}" if to_col else str(ref_table)
        return fks

    @staticmethod
    def _row_count(conn: sqlite3.Connection, table: str) -> int | None:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """SQLite has no persistent query history — always returns an empty list."""
        return []

    # ------------------------------------------------------------------
    # execute_query
    # ------------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute ``sql`` and return a :class:`QueryResult`.

        SQLite has no server-side statement timeout, so a runaway query is bounded
        best-effort by a watchdog thread that calls ``conn.interrupt()`` after the
        budget — *timeout_seconds* when given, else the
        ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS`` default (0 disables). The interrupt
        surfaces as an error rather than a hang.

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
            watchdog: threading.Timer | None = None
            if effective_timeout is not None and effective_timeout > 0:
                watchdog = threading.Timer(effective_timeout, conn.interrupt)
                watchdog.daemon = True
                watchdog.start()
            try:
                cursor = conn.execute(sql)
                _truncated, _reason = False, None
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows, _truncated, _reason = collect(cursor, max_rows)
                else:
                    columns, rows = [], []
            finally:
                if watchdog is not None:
                    watchdog.cancel()
            elapsed_ms = (time.perf_counter() - start) * 1000
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
            logger.exception("SQLiteConnector.execute_query failed")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )


def _quote_ident(identifier: str) -> str:
    """Double-quote a SQLite identifier, escaping embedded double-quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
