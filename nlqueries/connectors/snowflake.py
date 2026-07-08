"""
nlqueries.connectors.snowflake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Snowflake implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on the official ``snowflake-connector-python`` driver. This module is
part of the public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import snowflake.connector

from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)

# Schemas that are part of Snowflake / the catalog itself, never user data.
_SYSTEM_SCHEMAS = ("INFORMATION_SCHEMA",)


class SnowflakeConnector(DatabaseConnector):
    """Connector for Snowflake.

    Usage::

        connector = SnowflakeConnector()
        connector.connect({
            "account": "acme-prod",
            "user": "alice",
            "password": "s3cr3t",
            "warehouse": "COMPUTE_WH",
            "database": "ANALYTICS",
            "schema": "PUBLIC",       # optional
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._connection: Any = None
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

    def execute_query(self, sql: str, timeout_seconds: float | None = None) -> QueryResult:
        """Execute ``sql`` and return a :class:`QueryResult`.

        *timeout_seconds* is accepted for interface parity with
        :class:`~nlqueries.connectors.base.DatabaseConnector` but not yet
        implemented for Snowflake (Task 26.5 — Sprint 26 only wired this up
        for Postgres).

        Any exception raised during execution is caught and surfaced via
        ``QueryResult.error`` rather than propagating, so callers can treat
        query execution as always returning a result object.
        """
        start = time.perf_counter()
        try:
            connection = self._require_connection()
            cursor = connection.cursor()
            try:
                cursor.execute(sql)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = [list(row) for row in cursor.fetchall()]
                else:
                    columns = []
                    rows = []

                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=elapsed_ms,
                    error=None,
                )
            finally:
                cursor.close()
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


def _in_clause(values: tuple[str, ...]) -> str:
    """Render ``('A', 'B', ...)`` -> ``'A', 'B', ...`` for use inside a SQL ``IN (...)``."""
    return ", ".join(f"'{value}'" for value in values)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
