"""
nlqueries.connectors.duckdb
~~~~~~~~~~~~~~~~~~~~~~~~~~~
DuckDB implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Uses the official ``duckdb`` Python package directly (no SQLAlchemy required).
Install the optional extra before use::

    pip install "nlqueries-core[duckdb]"

DuckDB is primarily used for local / dev analytics — useful for evaluating
NLQueries without a running database server.  The ``database`` credential is a
file path (``/data/warehouse.db``) or the special string ``:memory:`` for an
in-process transient database.

DuckDB does not persist query history across connections.  ``extract_query_history``
always returns an empty list.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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

# DuckDB internal schemas that contain no user data.
_SYSTEM_SCHEMAS = frozenset({"information_schema", "pg_catalog"})

#: Applied to every DuckDB connection, with no way to switch it off.
#:
#: DuckDB is unlike the other engines here. It reads files through ordinary
#: table functions -- ``read_csv_auto``, ``read_text``, ``read_blob``, ``glob``
#: -- so a file read arrives inside a perfectly well-formed ``SELECT``, and the
#: database file is not the boundary. Opening the database read-only does
#: nothing about any of them. That was measured rather than assumed.
#:
#: Measured against DuckDB 1.5.5 with this absent: ``read_csv_auto``,
#: ``read_text``, ``read_blob``, ``glob``, ``ATTACH``, ``COPY TO`` (a file
#: *write*) and ``INSTALL httpfs`` (the network) all succeeded. With it, every
#: one is refused and ordinary queries are unaffected.
#:
#: The cost is real and worth stating: a database whose tables are views over
#: external parquet or CSV will stop working, because reading those files is
#: the thing being refused. Such a deployment has to materialise the data into
#: the DuckDB file. That is the trade this makes, deliberately.
# Typed as duckdb's own `config` parameter rather than dict[str, str]:
# dict is invariant, so the narrower annotation does not fit the call.
_SANDBOX: dict[str, str | bool | int | float | list[str]] = {
    "enable_external_access": "false",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
    # Last, and the reason the others hold: no later SET can reopen them.
    "lock_configuration": "true",
}


class DuckDBConnector(DatabaseConnector):
    """Connector for DuckDB.

    Usage::

        connector = DuckDBConnector()

        # File-based database:
        connector.connect({"database": "/data/warehouse.db"})

        # In-memory (transient, good for testing):
        connector.connect({"database": ":memory:"})

        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._conn: Any = None
        self._database: str = ":memory:"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Open a DuckDB connection.

        The only recognised credential key is ``database``: a file-system
        path or ``:memory:`` (default when omitted or empty).

        Raises :class:`ImportError` when ``duckdb`` is not installed.
        """
        try:
            import duckdb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'duckdb' package is required for DuckDB connections.\n"
                "Install it with:  pip install 'nlqueries-core[duckdb]'"
            ) from exc

        self._database = str(credentials.get("database") or ":memory:")
        in_memory = self._database == ":memory:"

        if not in_memory and not Path(self._database).exists():
            # DuckDB would otherwise create an empty database here, and the
            # deployment would get "no tables" instead of "that path is wrong"
            # -- a misconfiguration reported as a success.
            raise FileNotFoundError(
                f"DuckDB database {self._database!r} does not exist. This connector "
                f"reads an existing database; it does not create one. Use ':memory:' "
                f"for a transient database."
            )

        self._conn = duckdb.connect(
            database=self._database,
            # A database on disk is somebody's data, so it is opened for
            # reading. An in-memory database cannot be opened read-only at all
            # -- DuckDB refuses -- and it is private to this process and gone
            # when it exits, so there is nothing there to protect.
            read_only=not in_memory,
            config=_SANDBOX,
        )

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("DuckDBConnector.connect() must be called before use.")
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
            logger.exception("DuckDBConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via DuckDB catalog functions.

        Row counts are sourced from ``duckdb_tables().estimated_size``.
        Primary keys are detected via ``duckdb_constraints()``.
        Foreign keys are skipped — DuckDB FK constraints exist but are rarely
        used in analytics workloads and their catalog format varies across
        DuckDB versions.
        """
        conn = self._require_conn()

        row_counts = self._fetch_row_counts(conn)
        cols_by_table = self._fetch_columns(conn)
        pks = self._fetch_primary_keys(conn)

        tables: list[TableSpec] = []
        for (schema, name), row_count in row_counts.items():
            key = (schema, name)
            pk_cols = pks.get(key, set())
            columns = [
                ColumnSpec(
                    name=col["column_name"],
                    type=col["data_type"],
                    nullable=col["is_nullable"],
                    is_primary_key=col["column_name"] in pk_cols,
                    is_foreign_key=False,
                    references=None,
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

    @staticmethod
    def _fetch_row_counts(conn: Any) -> dict[tuple[str, str], int | None]:
        """Return ``{(schema, table): estimated_size}`` from ``duckdb_tables()``."""
        try:
            result = conn.execute(
                """
                SELECT schema_name, table_name, estimated_size
                FROM duckdb_tables()
                WHERE internal = false
                ORDER BY schema_name, table_name
                """
            )
            return {
                (r[0], r[1]): (int(r[2]) if r[2] is not None else None) for r in result.fetchall()
            }
        except Exception:  # noqa: BLE001
            # Fallback for older DuckDB versions without duckdb_tables()
            result = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                  AND table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_schema, table_name
                """
            )
            return {(r[0], r[1]): None for r in result.fetchall()}

    @staticmethod
    def _fetch_columns(conn: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
        result = conn.execute(
            """
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position
            """
        )
        cols: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in result.fetchall():
            key = (r[0], r[1])
            cols.setdefault(key, []).append(
                {
                    "column_name": r[2],
                    "data_type": r[3],
                    "is_nullable": str(r[4]).upper() == "YES",
                }
            )
        return cols

    @staticmethod
    def _fetch_primary_keys(conn: Any) -> dict[tuple[str, str], set[str]]:
        """Return ``{(schema, table): {pk_column_names}}`` from ``duckdb_constraints()``."""
        try:
            result = conn.execute(
                """
                SELECT schema_name, table_name, constraint_column_names
                FROM duckdb_constraints()
                WHERE constraint_type = 'PRIMARY KEY'
                """
            )
            pks: dict[tuple[str, str], set[str]] = {}
            for r in result.fetchall():
                schema, table, col_names = r[0], r[1], r[2]
                if col_names:
                    pks.setdefault((schema, table), set()).update(col_names)
            return pks
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """DuckDB has no persistent query history — always returns an empty list."""
        return []

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

        DuckDB has no server-side statement timeout, so a runaway query is bounded
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
                result = conn.execute(sql)
                _truncated, _reason = False, None
                if result.description:
                    columns = [desc[0] for desc in result.description]
                    rows, _truncated, _reason = collect(iter(result.fetchall()), max_rows)
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
            logger.exception("DuckDBConnector.execute_query failed")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
