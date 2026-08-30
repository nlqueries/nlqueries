"""
nlqueries.connectors.mssql
~~~~~~~~~~~~~~~~~~~~~~~~~~
SQL Server / Azure SQL implementation of
:class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on SQLAlchemy with the ``pymssql`` driver.  Install the optional extra
before use::

    pip install "nlqueries-core[mssql]"

Both on-premises SQL Server and Azure SQL are supported — Azure SQL speaks the
same T-SQL dialect.  Schema is introspected via standard ``information_schema``
views (PKs and FKs work identically to PostgreSQL).  Row counts come from
``sys.partitions`` (no elevated permissions required).  Query history is sourced
from ``sys.dm_exec_query_stats`` + ``sys.dm_exec_sql_text``, which requires
``VIEW SERVER STATE`` (on-premises) or ``VIEW DATABASE STATE`` (Azure SQL).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

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

# SQL Server built-in schemas that contain no user data.
_SYSTEM_SCHEMAS = frozenset({"sys", "INFORMATION_SCHEMA", "guest", "db_owner"})


class MSSQLConnector(DatabaseConnector):
    """Connector for SQL Server and Azure SQL.

    Usage::

        connector = MSSQLConnector()
        connector.connect({
            "host": "my-server.database.windows.net",
            "port": 1433,
            "database": "mydb",
            "user": "alice",
            "password": "YOUR_PASSWORD",
        })
        if connector.test_connection():
            schema = connector.extract_schema()

    Azure SQL tip: use ``<user>@<server-shortname>`` as the ``user`` value when
    connecting with SQL authentication to Azure SQL.
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials``.

        Required keys: ``host``, ``database``, ``user``, ``password``.
        ``port`` defaults to ``1433``.

        Raises :class:`ImportError` when ``pymssql`` is not installed.
        """
        try:
            import pymssql  # noqa: PLC0415, F401
        except ImportError as exc:
            raise ImportError(
                "The 'pymssql' package is required for SQL Server connections.\n"
                "Install it with:  pip install 'nlqueries-core[mssql]'"
            ) from exc

        url = URL.create(
            drivername="mssql+pymssql",
            username=credentials.get("user"),
            password=credentials.get("password"),
            host=credentials.get("host", "localhost"),
            port=int(credentials.get("port") or 1433),
            database=credentials["database"],
        )
        # SQL Server has no SET-based statement timeout, so bound queries via
        # pymssql's connection-level query `timeout` (seconds). Applied from the
        # CONNECTOR_STATEMENT_TIMEOUT_SECONDS default so a runaway query can't hang
        # indefinitely; 0 disables it. (Per-call execute_query timeout_seconds is
        # not separately honored on this driver.)
        connect_args: dict[str, Any] = {}
        if config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS > 0:
            connect_args["timeout"] = max(1, int(config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS))
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("MSSQLConnector.connect() must be called before use.")
        return self._engine

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds against the connected database."""
        try:
            engine = self._require_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            logger.exception("MSSQLConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via ``information_schema`` and ``sys.partitions``.

        Returns a :class:`SchemaSpec` for every user base table.  Row counts
        come from ``sys.partitions`` (approximate but permission-free).
        Column descriptions are not available in SQL Server's standard catalog
        and are returned as ``None``; extended properties would require a
        separate lookup and are out of scope here.
        """
        engine = self._require_engine()

        with engine.connect() as conn:
            database = conn.execute(text("SELECT DB_NAME()")).scalar_one()
            tables_meta = self._fetch_tables(conn)
            cols_by_table = self._fetch_columns(conn)
            pks = self._fetch_primary_keys(conn)
            fks = self._fetch_foreign_keys(conn)

        tables: list[TableSpec] = []
        for (schema, name), row_count in tables_meta.items():
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
            database=str(database),
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    @staticmethod
    def _fetch_tables(conn: Any) -> dict[tuple[str, str], int | None]:
        """Return ``{(schema, table): row_count}`` from ``sys.partitions``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    SCHEMA_NAME(t.schema_id)  AS table_schema,
                    t.name                    AS table_name,
                    SUM(p.rows)               AS row_count
                FROM sys.tables t
                JOIN sys.partitions p
                    ON p.object_id = t.object_id
                   AND p.index_id IN (0, 1)
                GROUP BY t.schema_id, t.name
                ORDER BY table_schema, table_name
                """
            )
        )
        result: dict[tuple[str, str], int | None] = {}
        for row in rows.mappings():
            rc = row["row_count"]
            result[(row["table_schema"], row["table_name"])] = int(rc) if rc is not None else None
        return result

    @staticmethod
    def _fetch_columns(conn: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
        rows = conn.execute(
            text(
                """
                SELECT
                    TABLE_SCHEMA,
                    TABLE_NAME,
                    COLUMN_NAME,
                    DATA_TYPE,
                    (CASE WHEN IS_NULLABLE = 'YES' THEN 1 ELSE 0 END) AS is_nullable
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest')
                ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
                """
            )
        )
        result: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows.mappings():
            key = (row["TABLE_SCHEMA"], row["TABLE_NAME"])
            result.setdefault(key, []).append(
                {
                    "column_name": row["COLUMN_NAME"],
                    "data_type": row["DATA_TYPE"],
                    "is_nullable": bool(row["is_nullable"]),
                }
            )
        return result

    @staticmethod
    def _fetch_primary_keys(conn: Any) -> dict[tuple[str, str], set[str]]:
        rows = conn.execute(
            text(
                """
                SELECT tc.TABLE_SCHEMA, tc.TABLE_NAME, kcu.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    ON kcu.CONSTRAINT_NAME  = tc.CONSTRAINT_NAME
                   AND kcu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                  AND tc.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
                """
            )
        )
        result: dict[tuple[str, str], set[str]] = {}
        for row in rows.mappings():
            result.setdefault((row["TABLE_SCHEMA"], row["TABLE_NAME"]), set()).add(
                row["COLUMN_NAME"]
            )
        return result

    @staticmethod
    def _fetch_foreign_keys(conn: Any) -> dict[tuple[str, str], dict[str, str]]:
        rows = conn.execute(
            text(
                """
                SELECT
                    tc.TABLE_SCHEMA,
                    tc.TABLE_NAME,
                    kcu.COLUMN_NAME,
                    ccu.TABLE_NAME  AS ref_table,
                    ccu.COLUMN_NAME AS ref_col
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    ON kcu.CONSTRAINT_NAME  = tc.CONSTRAINT_NAME
                   AND kcu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                    ON ccu.CONSTRAINT_NAME  = tc.CONSTRAINT_NAME
                   AND ccu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND tc.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
                """
            )
        )
        result: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows.mappings():
            key = (row["TABLE_SCHEMA"], row["TABLE_NAME"])
            result.setdefault(key, {})[row["COLUMN_NAME"]] = f"{row['ref_table']}.{row['ref_col']}"
        return result

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return top queries from ``sys.dm_exec_query_stats``.

        Requires ``VIEW SERVER STATE`` (SQL Server) or ``VIEW DATABASE STATE``
        (Azure SQL).  Returns an empty list if the DMV is inaccessible.
        """
        engine = self._require_engine()
        with engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT TOP {limit}
                            SUBSTRING(
                                qt.text,
                                (qs.statement_start_offset / 2) + 1,
                                (CASE
                                     WHEN qs.statement_end_offset = -1
                                     THEN LEN(CONVERT(nvarchar(MAX), qt.text)) * 2
                                     ELSE qs.statement_end_offset
                                 END - qs.statement_start_offset) / 2 + 1
                            )                                   AS query,
                            qs.execution_count,
                            qs.total_elapsed_time * 1.0
                                / qs.execution_count / 1000.0   AS avg_duration_ms,
                            qs.last_execution_time
                        FROM sys.dm_exec_query_stats qs
                        CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) qt
                        WHERE qs.last_execution_time >= DATEADD(day, :neg_days, GETDATE())
                          AND qt.text NOT LIKE '%sys.dm_exec%'
                        ORDER BY qs.execution_count DESC
                        """
                    ),
                    {"neg_days": -days},
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "MSSQLConnector.extract_query_history: could not query "
                    "sys.dm_exec_query_stats (requires VIEW SERVER STATE or "
                    "VIEW DATABASE STATE). Returning empty history."
                )
                return []

        return [
            QueryRecord(
                sql=str(row["query"] or "").strip(),
                execution_count=int(row["execution_count"]),
                avg_duration_ms=(
                    float(row["avg_duration_ms"]) if row["avg_duration_ms"] is not None else None
                ),
                last_executed=(
                    str(row["last_execution_time"]) if row["last_execution_time"] else None
                ),
            )
            for row in rows.mappings()
            if (row["query"] or "").strip()
        ]

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

        *timeout_seconds* is accepted for interface parity with
        :class:`~nlqueries.connectors.base.DatabaseConnector` but not yet
        implemented for MSSQL (Task 26.5 — Sprint 26 only wired this up
        for Postgres).

        Exceptions are caught and surfaced via ``QueryResult.error``.
        """
        start = time.perf_counter()
        try:
            engine = self._require_engine()
            with engine.begin() as conn:
                cursor_result = conn.execute(text(sql))
                elapsed_ms = (time.perf_counter() - start) * 1000
                _truncated, _reason = False, None
                if cursor_result.returns_rows:
                    columns = list(cursor_result.keys())
                    rows, _truncated, _reason = collect(cursor_result, max_rows)
                else:
                    columns, rows = [], []
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
            logger.exception("MSSQLConnector.execute_query failed")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
