"""
nlqueries.connectors.postgres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PostgreSQL implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on SQLAlchemy with the ``psycopg2`` driver. This module is part of the
public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from nlqueries import config
from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)
from nlqueries.telemetry import get_tracer

logger = logging.getLogger(__name__)

# Schemas that are part of Postgres / the catalog itself, never user data.
_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")


class PostgresConnector(DatabaseConnector):
    """Connector for PostgreSQL databases.

    Usage::

        connector = PostgresConnector()
        connector.connect({
            "host": "localhost",
            "port": 5432,
            "database": "mydb",
            "user": "alice",
            "password": "s3cr3t",
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials``.

        Expected keys: ``host``, ``port``, ``database``, ``user``, ``password``.
        ``host`` defaults to ``"localhost"`` and ``port`` to ``5432`` when omitted.
        """
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=credentials.get("user"),
            password=credentials.get("password"),
            host=credentials.get("host", "localhost"),
            port=credentials.get("port", 5432),
            database=credentials["database"],
        )
        connect_args: dict[str, Any] = {"connect_timeout": 10}
        ssl_mode = credentials.get("ssl_mode", "prefer")
        connect_args["sslmode"] = ssl_mode
        if ssl_ca_cert := credentials.get("ssl_ca_cert"):
            connect_args["sslrootcert"] = ssl_ca_cert
        if ssl_client_cert := credentials.get("ssl_client_cert"):
            connect_args["sslcert"] = ssl_client_cert
        if ssl_client_key := credentials.get("ssl_client_key"):
            connect_args["sslkey"] = ssl_client_key
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("PostgresConnector.connect() must be called before use.")
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
        except Exception:  # noqa: BLE001 — any failure means "not connected"
            logger.exception("PostgresConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via ``information_schema`` (+ ``pg_class`` for row counts).

        Builds a full :class:`SchemaSpec` describing every base table outside the
        system schemas (``pg_catalog``, ``information_schema``): its columns,
        primary keys, foreign keys, and an estimated row count.
        """
        engine = self._require_engine()

        with engine.connect() as conn:
            database = conn.execute(text("SELECT current_database()")).scalar_one()

            tables_meta = self._fetch_tables(conn)
            columns_by_table = self._fetch_columns(conn)
            primary_keys = self._fetch_primary_keys(conn)
            foreign_keys = self._fetch_foreign_keys(conn)

            tables: list[TableSpec] = []
            for (schema_name, table_name), meta in tables_meta.items():
                key = (schema_name, table_name)
                pk_columns = primary_keys.get(key, set())
                fk_columns = foreign_keys.get(key, {})

                columns = [
                    ColumnSpec(
                        name=col["column_name"],
                        type=col["data_type"],
                        nullable=col["is_nullable"],
                        is_primary_key=col["column_name"] in pk_columns,
                        is_foreign_key=col["column_name"] in fk_columns,
                        references=fk_columns.get(col["column_name"]),
                        description=col["description"],
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
            database=str(database),
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    @staticmethod
    def _fetch_tables(conn: Any) -> dict[tuple[str, str], dict[str, Any]]:
        """Return ``{(schema, table): {row_count, description}}`` using ``pg_class.reltuples``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    t.table_schema,
                    t.table_name,
                    c.reltuples::bigint AS row_estimate,
                    obj_description(c.oid) AS table_description
                FROM information_schema.tables t
                JOIN pg_catalog.pg_class c
                    ON c.relname = t.table_name
                JOIN pg_catalog.pg_namespace n
                    ON n.oid = c.relnamespace AND n.nspname = t.table_schema
                WHERE t.table_type = 'BASE TABLE'
                  AND t.table_schema NOT IN :system_schemas
                ORDER BY t.table_schema, t.table_name
                """
            ),
            {"system_schemas": _SYSTEM_SCHEMAS},
        )
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows.mappings():
            row_estimate = row["row_estimate"]
            # reltuples is -1 for tables that have never been analysed/vacuumed.
            has_estimate = row_estimate is not None and row_estimate >= 0
            result[(row["table_schema"], row["table_name"])] = {
                "row_count": int(row_estimate) if has_estimate else None,
                "description": row["table_description"],
            }
        return result

    @staticmethod
    def _fetch_columns(conn: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return ``{(schema, table): [column dicts in ordinal order]}``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    (c.is_nullable = 'YES') AS is_nullable,
                    pgd.description
                FROM information_schema.columns c
                LEFT JOIN pg_catalog.pg_statio_all_tables st
                    ON st.schemaname = c.table_schema AND st.relname = c.table_name
                LEFT JOIN pg_catalog.pg_description pgd
                    ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
                WHERE c.table_schema NOT IN :system_schemas
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
                """
            ),
            {"system_schemas": _SYSTEM_SCHEMAS},
        )
        result: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows.mappings():
            key = (row["table_schema"], row["table_name"])
            result.setdefault(key, []).append(dict(row))
        return result

    @staticmethod
    def _fetch_primary_keys(conn: Any) -> dict[tuple[str, str], set[str]]:
        """Return ``{(schema, table): {primary key column names}}``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON kcu.constraint_name = tc.constraint_name
                   AND kcu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema NOT IN :system_schemas
                """
            ),
            {"system_schemas": _SYSTEM_SCHEMAS},
        )
        result: dict[tuple[str, str], set[str]] = {}
        for row in rows.mappings():
            key = (row["table_schema"], row["table_name"])
            result.setdefault(key, set()).add(row["column_name"])
        return result

    @staticmethod
    def _fetch_foreign_keys(conn: Any) -> dict[tuple[str, str], dict[str, str]]:
        """Return ``{(schema, table): {column_name: "ref_table.ref_column"}}``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON kcu.constraint_name = tc.constraint_name
                   AND kcu.constraint_schema = tc.constraint_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                   AND ccu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema NOT IN :system_schemas
                """
            ),
            {"system_schemas": _SYSTEM_SCHEMAS},
        )
        result: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows.mappings():
            key = (row["table_schema"], row["table_name"])
            reference = f"{row['foreign_table_name']}.{row['foreign_column_name']}"
            result.setdefault(key, {})[row["column_name"]] = reference
        return result

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return the top ``pg_stat_statements`` queries by execution count.

        ``pg_stat_statements`` accumulates statistics since the extension's
        last reset rather than tracking individual execution timestamps, so
        ``days`` cannot be used as a precise SQL filter; it is accepted for
        interface compatibility and to size logging/messaging. Up to
        ``limit`` queries are returned, ordered by execution count
        (``calls``) descending.

        If the ``pg_stat_statements`` extension is not installed, this logs
        a warning and returns an empty list rather than raising.
        """
        engine = self._require_engine()

        with engine.connect() as conn:
            installed = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'")
            ).scalar_one_or_none()

            if not installed:
                logger.warning(
                    "extract_query_history: the 'pg_stat_statements' extension is not "
                    "installed on this database — returning an empty query history. "
                    "Install it with `CREATE EXTENSION pg_stat_statements;` "
                    "(requires it to be listed in shared_preload_libraries) to enable "
                    "query history extraction."
                )
                return []

            try:
                rows = conn.execute(
                    text(
                        """
                        SELECT query, calls, mean_exec_time
                        FROM pg_stat_statements
                        ORDER BY calls DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
            except Exception:
                # Postgres < 13 named the column `mean_time` instead of `mean_exec_time`.
                rows = conn.execute(
                    text(
                        """
                        SELECT query, calls, mean_time AS mean_exec_time
                        FROM pg_stat_statements
                        ORDER BY calls DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )

            return [
                QueryRecord(
                    sql=row["query"],
                    execution_count=int(row["calls"]),
                    avg_duration_ms=(
                        float(row["mean_exec_time"]) if row["mean_exec_time"] is not None else None
                    ),
                    # pg_stat_statements does not record per-query timestamps.
                    last_executed=None,
                )
                for row in rows.mappings()
            ]

    # ------------------------------------------------------------------
    # execute_query
    # ------------------------------------------------------------------

    def execute_query(self, sql: str, timeout_seconds: float | None = None) -> QueryResult:
        """Execute ``sql`` and return a :class:`QueryResult`.

        Any exception raised during execution is caught and surfaced via
        ``QueryResult.error`` rather than propagating, so callers can treat
        query execution as always returning a result object.

        ``SET LOCAL statement_timeout`` is applied inside the same transaction so
        the server itself aborts the query once it's been running that long, rather
        than the query continuing to hold locks/connections after the caller (e.g.
        an API request that already returned 504) has given up waiting on it.

        The budget is *timeout_seconds* when given (Task 26.5 — Sprint 26),
        otherwise the ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS`` default — so a query
        without an explicit budget still can't hang indefinitely. A budget of 0
        (config default set to 0) disables the timeout.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
        )
        tracer = get_tracer()
        start = time.perf_counter()
        with tracer.start_as_current_span("postgres_connector.execute_query") as span:
            span.set_attribute("db.system", "postgresql")
            try:
                engine = self._require_engine()
                with engine.begin() as conn:
                    if effective_timeout is not None and effective_timeout > 0:
                        statement_timeout_ms = max(1, int(effective_timeout * 1000))
                        conn.execute(text(f"SET LOCAL statement_timeout = {statement_timeout_ms}"))
                    cursor_result = conn.execute(text(sql))
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    if cursor_result.returns_rows:
                        columns = list(cursor_result.keys())
                        rows = [list(row) for row in cursor_result.fetchall()]
                    else:
                        columns = []
                        rows = []

                    span.set_attribute("db.row_count", len(rows))
                    return QueryResult(
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        execution_time_ms=elapsed_ms,
                        error=None,
                    )
            except Exception as exc:  # noqa: BLE001 — surfaced via QueryResult.error, not raised
                elapsed_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("error", True)
                logger.exception("PostgresConnector.execute_query failed")
                return QueryResult(
                    columns=[],
                    rows=[],
                    row_count=0,
                    execution_time_ms=elapsed_ms,
                    error=str(exc),
                )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
