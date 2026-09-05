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
from sqlalchemy.engine import Connection, Engine, make_url

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
from nlqueries.connectors.postgres_tls import TlsPosture, describe, resolve_ssl_mode

logger = logging.getLogger(__name__)


def _apply_read_only(conn: Connection) -> None:
    """Best-effort per-dialect read-only transaction on *conn*.

    The counterpart to :func:`_apply_statement_timeout`, and a **no-op** on
    dialects that cannot express it -- there the unconditional rollback in
    ``_execute_query`` is the whole of the guard.

    Postgres and Redshift take ``SET TRANSACTION READ ONLY``, which is
    transaction-scoped and must be the first statement in the transaction, which
    is where this is called from. It is broader than any validator can be,
    because it applies to what a statement *does* rather than to how it is
    spelled: DML and DDL anywhere in the call graph, and sequence functions,
    refused by name.

    MySQL and MariaDB are deliberately left to the rollback. Their form is
    ``SET SESSION TRANSACTION READ ONLY``, which sets the mode for *subsequent*
    transactions and is refused with error 1568 inside an open one -- and by the
    time anything here can run, SQLAlchemy has already emitted ``BEGIN`` on the
    first execute. ``START TRANSACTION READ ONLY`` would be the right statement
    and is not something a caller can issue when SQLAlchemy owns the transaction.
    Issuing the session form anyway would turn every MySQL query into an error,
    which is a worse outcome than the rollback it would be layered on top of.

    And the rollback does not cover MySQL's non-transactional storage engines at
    all: an ``INSERT`` into a MyISAM or MEMORY table survives, and the server
    reports warning 1196 rather than an error. Nothing here can prevent that; the
    grant is the control.

    Oracle is a gap rather than a limit, and the distinction is worth keeping.
    It has a transaction-scoped ``SET TRANSACTION READ ONLY`` that must be the
    first statement of the transaction -- exactly the position this function
    occupies -- and refuses DML and DDL thereafter with ORA-01456, so the DDL
    exposure recorded for Oracle elsewhere in the docs looks closable rather than
    inherent. It is left out because no test here can reach an Oracle instance,
    and adding a statement to the execution path of an engine nothing exercises
    is how every query against it becomes an error instead. Whoever has an
    instance to try it against should add ``oracle`` to the tuple below and move
    it out of the "commits implicitly and nothing more" column.
    """
    name = conn.engine.dialect.name
    if name in ("postgresql", "redshift"):
        conn.execute(text("SET TRANSACTION READ ONLY"))


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


#: Credential keys that configure TLS, mapped to the libpq connection
#: parameters of the same meaning. Every PostgreSQL driver SQLAlchemy ships a
#: dialect for -- psycopg2, psycopg 3 -- passes these straight to libpq.
_LIBPQ_TLS_PARAMS = {
    "ssl_mode": "sslmode",
    "ssl_ca_cert": "sslrootcert",
    "ssl_client_cert": "sslcert",
    "ssl_client_key": "sslkey",
}

#: Drivers known to accept the libpq parameter names above. `pg8000` and
#: `asyncpg` speak to PostgreSQL but configure TLS through an ``ssl_context``
#: object instead, so they are deliberately absent: guessing wrong here would
#: silently produce the unverified connection this exists to prevent. An empty
#: driver name is psycopg2, SQLAlchemy's default for `postgresql://`.
_LIBPQ_DRIVERS = frozenset({"", "psycopg2", "psycopg"})


#: The reverse of :data:`_LIBPQ_TLS_PARAMS`, for reading a URL's own parameters
#: back into the credential names :func:`describe` understands.
_LIBPQ_TO_CREDENTIAL = {param: key for key, param in _LIBPQ_TLS_PARAMS.items()}


def _merged_tls_view(url: str, credentials: dict[str, Any]) -> dict[str, Any]:
    """Every TLS setting in force, in credential names, from both sources.

    The clash check refuses any parameter set on both sides, so the two are
    disjoint and can be merged rather than reconciled.

    This exists so the mode this connector injects and the posture it reports
    are decided by the same view. They were decided separately, and disagreed:
    `_effective_posture` read `sslrootcert` from the URL while
    `resolve_ssl_mode` saw only the credentials, so a root certificate supplied
    in the URL did not select `verify-full` the way the connector's own rule --
    and `PostgresConnector` on identical material -- says it should. The
    operator's only way to clear the resulting concern was to duplicate the CA
    path into the credentials.
    """
    merged: dict[str, Any] = {}
    for param, value in make_url(url).query.items():
        if param in _LIBPQ_TO_CREDENTIAL:
            # A repeated query parameter arrives as a tuple; libpq takes the last.
            merged[_LIBPQ_TO_CREDENTIAL[param]] = value[-1] if isinstance(value, tuple) else value
    merged.update({key: credentials[key] for key in _LIBPQ_TLS_PARAMS if credentials.get(key)})
    return merged


def _posture_for(merged: dict[str, Any], injected: dict[str, Any]) -> TlsPosture | None:
    """The posture in force, or ``None`` when it is not ours to know.

    ``None`` when nothing on either side settles the mode: libpq then applies
    its own ``prefer`` default, which is not what :func:`resolve_ssl_mode` would
    have chosen, so describing these settings would name a posture the
    connection is not running under.
    """
    if "sslmode" in injected:
        return describe({**merged, "ssl_mode": injected["sslmode"]})
    if "ssl_mode" in merged:
        return describe(merged)
    return None


def _tls_connect_args(url: str, credentials: dict[str, Any]) -> dict[str, Any]:
    """Translate configured TLS settings into ``create_engine`` connect args.

    This connector takes a whole URL, so it has no per-field TLS handling of its
    own the way :class:`~nlqueries.connectors.postgres.PostgresConnector` does.
    That left it accepting ``ssl_mode`` and ``ssl_ca_cert``, storing them,
    delivering them here, and then connecting as though they had not been set --
    under libpq's ``prefer`` default, which falls back to plaintext without
    saying so. A setting that is accepted and then ignored is worse than one
    that is not supported: the operator has no way to discover the difference.

    So a setting we cannot apply is refused rather than dropped. An operator who
    wants the generic connector on a driver we do not know can still express any
    posture they like in the URL's own query string, which is this connector's
    documented interface.
    """
    configured = {key: credentials[key] for key in _LIBPQ_TLS_PARAMS if credentials.get(key)}
    if not configured:
        return {}

    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or parsed.get_driver_name() not in _LIBPQ_DRIVERS:
        raise ValueError(
            f"SQLAlchemyConnector cannot apply {sorted(configured)} to a "
            f"'{parsed.drivername}' URL: the TLS parameter names are driver-specific "
            f"and only the libpq-based PostgreSQL drivers are mapped. Put the "
            f"equivalent settings in the URL's query string instead."
        )

    args = {_LIBPQ_TLS_PARAMS[key]: value for key, value in configured.items()}
    # Resolve the mode rather than only renaming keys. `ssl_ca_cert` with no
    # `ssl_mode` is the case that matters: renaming alone yields `sslrootcert`
    # under libpq's `prefer` default, so the operator supplies a CA and gets a
    # session that falls back to plaintext and verifies nothing. That
    # combination is deliberate elsewhere -- the GCP Cloud SQL IAM provider
    # leaves `ssl_mode` unset when a root certificate is configured precisely
    # because it expects the connector to choose `verify-full`, which is what
    # `resolve_ssl_mode` does and what PostgresConnector already does with the
    # same credentials.
    # From the merged view, not the credentials alone: a root certificate given
    # in the URL must select `verify-full` exactly as one given in the
    # credentials does. Skipped when the URL already sets the mode -- injecting
    # the same value would trip the clash check below on a configuration where
    # nothing is in conflict.
    if "sslmode" not in parsed.query:
        args["sslmode"] = resolve_ssl_mode(_merged_tls_view(url, credentials))

    # `create_engine` unions the URL's own parameters with `connect_args` and
    # lets `connect_args` win, so a URL that already names one of these would be
    # overruled with no indication -- and the documented advice for an unmapped
    # driver is to put the posture in the query string, so both being present is
    # a reasonable thing for someone to do. Refusing is the same rule as above:
    # a configured setting is applied or refused, never quietly dropped.
    # Against `args`, not the whole TLS vocabulary. A URL naming a parameter
    # this connector would not set is not a conflict: `?sslrootcert=...` beside
    # a configured `ssl_mode` replaces nothing, and refusing it turned a valid
    # split configuration into an error whose message named two sets that did
    # not overlap.
    clash = sorted(set(parsed.query) & set(args))
    if clash:
        raise ValueError(
            f"SQLAlchemyConnector will not silently overrule the URL: both it and "
            f"this connector's TLS credentials set {clash}, and SQLAlchemy would "
            f"let the credentials win without saying so. Configure each setting in "
            f"one place -- either the URL's query string or the ssl_* credentials."
        )
    return args


class SQLAlchemyConnector(DatabaseConnector):
    """Generic connector for any SQLAlchemy-reachable database (URL-driven)."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._tls: TlsPosture | None = None

    @property
    def tls(self) -> TlsPosture | None:
        """The TLS posture this connector resolved, or None.

        Built from the URL's own TLS parameters and this connector's, which the
        clash check keeps disjoint. ``None`` means the posture is not ours to
        report: ``connect`` has not run, or nothing on either side set a mode,
        leaving libpq's ``prefer`` default -- and reporting a mode the
        connection is not running under would be worse than reporting nothing.
        """
        return self._tls

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials["url"]``.

        The URL is a standard SQLAlchemy URL, e.g.
        ``postgresql+psycopg://user:pass@host/db`` or ``sqlite:////data/db.sqlite``.
        Raises :class:`ValueError` when no URL is supplied.

        TLS settings configured alongside the URL are applied when the driver
        is one whose parameter names we know; see :func:`_tls_connect_args`. A
        setting we cannot apply raises rather than being ignored.
        """
        url = credentials.get("url") or credentials.get("connection_string")
        if not url or not str(url).strip():
            raise ValueError("SQLAlchemyConnector requires a 'url' credential (a SQLAlchemy URL).")
        url = str(url).strip()
        connect_args = _tls_connect_args(url, credentials)
        merged = _merged_tls_view(url, credentials)
        # Record what was resolved, the way PostgresConnector does. `nlqueries
        # health` reads `connector.tls` and stayed silent for this connector
        # type, so a `sqlalchemy` entry running at `require` with no certificate
        # -- encrypted, verifying nothing -- reported no concern at all, while
        # the identical `postgres` entry reported one. The posture is known here
        # now; discarding it was the last place this connector knew something
        # about its own TLS and did not say.
        #
        # From both sources: a split configuration is now permitted, so neither
        # half describes the connection on its own.
        self._tls = _posture_for(merged, connect_args)
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

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
            # `connect()` with an unconditional rollback, never `begin()`, which
            # commits on the way out. Every validator in front of this one asks
            # only whether the root node is a Select, and
            # `SELECT some_volatile_function(...)` satisfies that while still
            # writing -- reproduced through the Postgres connector twice by audit,
            # eight weeks apart, and committed by `begin()` both times. This
            # connector serves MySQL and any URL nobody wrote a dedicated class
            # for, so it had the same ending and no read-only layer at all.
            with self._require_engine().connect() as conn:
                try:
                    # First statement in the transaction, which is where a
                    # read-only mode has to be set if the dialect has one.
                    _apply_read_only(conn)
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
                finally:
                    # In `finally`, so it runs on the success path too: the point
                    # is that a statement which succeeded is still not committed.
                    # Rows have already been collected by here.
                    #
                    # The failure is logged and swallowed rather than raised. A
                    # connection dropped after a cancelled or timed-out statement
                    # is the realistic case, and letting it out of the `finally`
                    # would report a successful query as an error, or replace the
                    # driver's message -- the one the caller needs -- with the
                    # rollback's. Nothing is committed either way: the connection
                    # is reset when the pool takes it back, and an unreachable
                    # connection has no transaction left to commit.
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "%s: rollback after the query failed. Nothing was "
                            "committed; the connection is reset on return to the "
                            "pool.",
                            "SQLAlchemyConnector",
                            exc_info=True,
                        )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("SQLAlchemyConnector.execute_query failed")
            return QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=elapsed_ms, error=str(exc)
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
