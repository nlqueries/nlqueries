"""
nlqueries.connectors.sqlalchemy_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A generic :class:`~nlqueries.connectors.base.DatabaseConnector` for **any**
database SQLAlchemy can reach, driven by a full SQLAlchemy URL.

Unlike the per-database connectors, this one issues no dialect-specific catalog
SQL: schema is introspected through SQLAlchemy's dialect-agnostic
:class:`~sqlalchemy.engine.reflection.Inspector`, so it works for PostgreSQL,
MySQL/MariaDB, SQLite, Oracle, SQL Server, and anything else with a SQLAlchemy
dialect installed. The only credential is the URL::

    connector = SQLAlchemyConnector()
    connector.connect({"url": "mysql+pymysql://user:pass@host:3306/analytics"})

The URL's driver (``pymysql``, ``cx_Oracle``, …) must be installed in the
environment; SQLAlchemy itself is a core dependency. Query history has no
portable source, so it is always empty. Schema reflection covers the
connection's default schema.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

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


def _apply_statement_timeout(conn: Connection, seconds: float) -> None:
    """Best-effort per-dialect statement timeout on *conn*.

    Statement timeouts are dialect-specific, so this covers the common cases and is
    a **no-op** on dialects without a simple session-level one (e.g. SQLite) — the
    query then runs unbounded there rather than erroring. Applied inside the query's
    transaction so it takes effect for the statement that follows.
    """
    dialect = conn.engine.dialect
    name = dialect.name
    ms = max(1, int(seconds * 1000))
    if name in ("postgresql", "redshift"):
        # SET LOCAL is transaction-scoped, so it resets when the tx ends — no leak
        # onto the pooled connection.
        conn.execute(text(f"SET LOCAL statement_timeout = {ms}"))
    elif name in ("mysql", "mariadb"):
        # MySQL 5.7.8+ uses max_execution_time (ms, SELECT only); MariaDB 10.1+ uses
        # max_statement_time (seconds). SQLAlchemy's `_is_mariadb` distinguishes them
        # even when both report dialect name "mysql" (e.g. via pymysql).
        if getattr(dialect, "_is_mariadb", False):
            conn.execute(text(f"SET SESSION max_statement_time = {float(seconds)}"))
        else:
            conn.execute(text(f"SET SESSION max_execution_time = {ms}"))
    else:
        logger.debug("No statement-timeout mechanism for dialect %r; running unbounded", name)


class SQLAlchemyConnector(DatabaseConnector):
    """Generic connector for any SQLAlchemy-reachable database (URL-driven)."""

    def __init__(self) -> None:
        self._engine: Engine | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials["url"]``.

        The URL is a standard SQLAlchemy URL, e.g.
        ``postgresql+psycopg://user:pass@host/db`` or ``sqlite:////data/db.sqlite``.
        Raises :class:`ValueError` when no URL is supplied.
        """
        url = credentials.get("url") or credentials.get("connection_string")
        if not url or not str(url).strip():
            raise ValueError("SQLAlchemyConnector requires a 'url' credential (a SQLAlchemy URL).")
        self._engine = create_engine(str(url).strip(), pool_pre_ping=True)

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("SQLAlchemyConnector.connect() must be called before use.")
        return self._engine

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds against the connected database."""
        try:
            with self._require_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            logger.exception("SQLAlchemyConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the default schema via SQLAlchemy's dialect-agnostic Inspector.

        Row counts and column descriptions are left ``None`` — neither is
        portably available across dialects without extra per-table queries.
        A table that fails to reflect is skipped rather than aborting the build.
        """
        engine = self._require_engine()
        inspector = inspect(engine)
        default_schema = inspector.default_schema_name or ""
        tables: list[TableSpec] = []

        for table_name in inspector.get_table_names():
            try:
                pk = inspector.get_pk_constraint(table_name).get("constrained_columns") or []
                pk_cols = set(pk)
                fk_cols: dict[str, str] = {}
                for fk in inspector.get_foreign_keys(table_name):
                    ref_table = fk.get("referred_table")
                    referred = fk.get("referred_columns") or []
                    constrained = fk.get("constrained_columns") or []
                    for i, col in enumerate(constrained):
                        ref_col = referred[i] if i < len(referred) else ""
                        fk_cols[col] = f"{ref_table}.{ref_col}"

                columns = [
                    ColumnSpec(
                        name=col["name"],
                        type=str(col.get("type")),
                        nullable=bool(col.get("nullable", True)),
                        is_primary_key=col["name"] in pk_cols,
                        is_foreign_key=col["name"] in fk_cols,
                        references=fk_cols.get(col["name"]),
                        description=col.get("comment"),
                    )
                    for col in inspector.get_columns(table_name)
                ]
                tables.append(
                    TableSpec(
                        name=table_name,
                        schema=default_schema,
                        row_count=None,
                        columns=columns,
                        description=None,
                    )
                )
            except Exception:  # noqa: BLE001 — one bad table never breaks the whole reflect
                logger.warning("SQLAlchemyConnector: could not reflect table %r", table_name)

        return SchemaSpec(
            database=str(engine.url.database or ""),
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """No portable cross-dialect query-history source — always empty."""
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

        A statement timeout is applied best-effort per dialect (see
        :func:`_apply_statement_timeout`): *timeout_seconds* when given, else the
        ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS`` default — so a runaway query can't
        hang indefinitely on a DB that supports it. Dialects without one (e.g.
        SQLite) run unbounded. Exceptions are surfaced via ``QueryResult.error``.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
        )
        start = time.perf_counter()
        try:
            with self._require_engine().begin() as conn:
                if effective_timeout is not None and effective_timeout > 0:
                    _apply_statement_timeout(conn, effective_timeout)
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
            logger.exception("SQLAlchemyConnector.execute_query failed")
            return QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=elapsed_ms, error=str(exc)
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
