"""
nlqueries.connectors.sqlalchemy_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A generic :class:`~nlqueries.connectors.base.DatabaseConnector` for **any**
database SQLAlchemy can reach, driven by a full SQLAlchemy URL.

Unlike the per-database connectors, this one issues no dialect-specific catalog
SQL: schema is introspected through SQLAlchemy's dialect-agnostic
:class:`~sqlalchemy.engine.reflection.Inspector`, so it works for PostgreSQL,
MySQL/MariaDB, Oracle, SQL Server, and anything else with a SQLAlchemy dialect
installed. SQLite and DuckDB URLs are refused: each has a dedicated connector
that opens the file read-only and sandboxed, which this one does not. The only
credential is the URL::

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
from sqlalchemy.engine import URL, Connection, Engine, make_url
from sqlalchemy.exc import DatabaseError

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
    short_cause,
)
from nlqueries.connectors.login_identity import LoginIdentity, watch_engine
from nlqueries.connectors.mysql_identity import inspect_identity as inspect_mysql_identity
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

    MySQL and MariaDB take ``START TRANSACTION READ ONLY``, which opens the
    transaction itself. SQLAlchemy's MySQL dialects send no ``BEGIN`` of their
    own -- ``do_begin`` is the default no-op, and the server opens a transaction
    implicitly on the first statement -- so this is that first statement, and
    the rollback in ``_execute_query`` ends it. The statement timeout is set
    after it: ``START TRANSACTION`` commits implicitly, and the
    ``SET SESSION`` timeout is then in force for the query it precedes.

    Measured through this connector with PyMySQL 1.2.3 and SQLAlchemy 2.0
    against MySQL 8.0.46, MySQL 8.4.11 and MariaDB 10.11.14: a ``SELECT``
    runs; an ``INSERT`` into an InnoDB table and into a MyISAM table are both
    refused with error 1792, where before the MyISAM row survived the rollback;
    the timeout reads back as set and cancels a query over it; and the next
    use of the same pooled connection is not read-only, because the mode
    belongs to the transaction. (``SET SESSION TRANSACTION READ ONLY`` would
    outlive the query on the pooled connection, which is why it is not the
    form used.) DDL still commits implicitly, ending the transaction first, so
    a ``CREATE TABLE`` stands.

    Oracle takes the same ``SET TRANSACTION READ ONLY`` as Postgres, also
    transaction-scoped and also first. Measured through this connector with
    python-oracledb 26.1.0 and SQLAlchemy 2.0 against Oracle Database Free
    23.26.3: a ``SELECT`` runs; ``INSERT``, ``UPDATE``, ``DELETE`` and
    ``SELECT ... FOR UPDATE`` are refused with ORA-01456; and the next use of
    the same pooled connection is not read-only. DDL is not refused: Oracle
    commits implicitly before it, which ends the read-only transaction, so a
    ``CREATE TABLE`` stands, as on MySQL. One cost: a read-only transaction
    cannot read a table newer than its snapshot, so a query against a table
    truncated or created a moment before is refused with ORA-01466 ("table
    definition has changed"). Measured eight times each: refused every time
    directly after a ``TRUNCATE``, and for about two seconds after a
    ``CREATE``; never a second after the ``TRUNCATE`` or two after the
    ``CREATE``. ``ALTER TABLE ... ADD`` and ordinary writes did not cause it.
    So ``_execute_query`` retries a query refused with ORA-01466 once, after
    ``ORACLE_SNAPSHOT_RETRY_SECONDS`` (3 by default; see that setting).
    """
    name = conn.engine.dialect.name
    if name in ("postgresql", "redshift", "oracle"):
        conn.execute(text("SET TRANSACTION READ ONLY"))
    elif name in _MYSQL_BACKENDS:
        conn.execute(text("START TRANSACTION READ ONLY"))


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


#: Backends that have a dedicated connector, mapped to its db type. A URL for
#: one of these is refused here rather than opened: the dedicated connectors open
#: the file read-only and, for DuckDB, with external file access switched off.
#: Through this connector the same URL opens the file read-write, with
#: ``ATTACH`` and DuckDB's file-reading functions available to any query.
_DEDICATED_BACKENDS = {"sqlite": "sqlite", "duckdb": "duckdb"}

#: Backend names SQLAlchemy uses for MySQL and MariaDB.
_MYSQL_BACKENDS = frozenset({"mysql", "mariadb"})

#: ``CLIENT_MULTI_STATEMENTS`` in the MySQL protocol: lets one call carry
#: several ``;``-separated statements, which the server then runs in turn.
_CLIENT_MULTI_STATEMENTS = 1 << 16

#: URL parameters that carry the MySQL client capability flags: ``client_flag``
#: for mysqlclient, PyMySQL and MariaDB Connector/Python, ``client_flags`` for
#: MySQL Connector/Python.
_CLIENT_FLAG_PARAMS = ("client_flag", "client_flags")

#: URL parameters that would switch ``LOAD DATA LOCAL INFILE`` back on.
_LOCAL_INFILE_PARAMS = ("local_infile", "allow_local_infile")

#: For each MySQL driver whose spelling is known, the connect argument that
#: turns ``LOAD DATA LOCAL INFILE`` off. The empty name is ``mysqldb``
#: (mysqlclient), SQLAlchemy's default for ``mysql://``. Another driver -- one
#: reached through ODBC, say -- takes neither keyword and failed with a
#: ``TypeError`` at its first connection, so it is refused by name instead.
_MYSQL_LOCAL_INFILE_OFF: dict[str, dict[str, Any]] = {
    "": {"local_infile": 0},
    "mysqldb": {"local_infile": 0},
    "pymysql": {"local_infile": 0},
    "mariadbconnector": {"local_infile": 0},
    "mysqlconnector": {"allow_local_infile": False},
}


def _refuse_dedicated_backend(parsed: URL) -> None:
    """Raise if *parsed* names an engine that has a connector of its own."""
    backend = parsed.get_backend_name()
    if backend in _DEDICATED_BACKENDS:
        db_type = _DEDICATED_BACKENDS[backend]
        raise ValueError(
            f"SQLAlchemyConnector does not open {backend} URLs. Use the dedicated "
            f"'{db_type}' connector (nlqueries connect {db_type}), which opens the "
            f"database read-only and sandboxed."
        )


def _harden_mysql(parsed: URL) -> tuple[URL, dict[str, Any]]:
    """*parsed* with the MySQL settings this connector does not allow taken out.

    Returns the URL to build the engine from and the connect args to add. A
    URL for any other backend comes back unchanged with no connect args.

    ``LOAD DATA LOCAL INFILE`` asks the *client* to send a file from its own
    disk, and the server chooses the file name -- so with it enabled, the host
    running this process will upload whatever readable file a statement or the
    server names. It is disabled in the connect args, which SQLAlchemy lets
    win over the URL, and any URL parameter enabling it is dropped so the URL
    does not claim otherwise.

    The multi-statements capability is stripped from any client flag the URL
    sets. The SQL policy already refuses more than one statement; this makes
    the driver refuse it too, so a second statement is not one the server will
    run whatever reaches it. ``init_command`` and the other parameters are left
    alone.
    """
    if parsed.get_backend_name() not in _MYSQL_BACKENDS:
        return parsed, {}

    query = {k: v for k, v in parsed.query.items() if k not in _LOCAL_INFILE_PARAMS}
    for param in _CLIENT_FLAG_PARAMS:
        if param not in query:
            continue
        raw = query[param]
        # A repeated query parameter arrives as a tuple; the driver takes the last.
        value = raw[-1] if isinstance(raw, tuple) else raw
        try:
            flags = int(value)
        except ValueError:
            raise ValueError(
                f"SQLAlchemyConnector: the URL's {param} must be an integer, not {value!r}."
            ) from None
        query[param] = str(flags & ~_CLIENT_MULTI_STATEMENTS)

    connect_args = _MYSQL_LOCAL_INFILE_OFF.get(parsed.get_driver_name())
    if connect_args is None:
        known = ", ".join(sorted(d for d in _MYSQL_LOCAL_INFILE_OFF if d))
        raise ValueError(
            f"SQLAlchemyConnector cannot turn LOAD DATA LOCAL INFILE off for the "
            f"{parsed.get_driver_name()!r} MySQL driver, so it does not open it. "
            f"Use one of: {known}."
        )
    return parsed.set(query=query), dict(connect_args)


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
        self._identity: LoginIdentity | None = None

    @property
    def identity(self) -> LoginIdentity | None:
        """What a MySQL or MariaDB login holds beyond reading, or None.

        Read from the most recent connection the pool opened; None before one
        has opened, and for every other backend, which this does not inspect.
        """
        return self._identity

    def _set_identity(self, identity: LoginIdentity) -> None:
        self._identity = identity

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

    @property
    def dialect_name(self) -> str | None:
        """The backend SQLAlchemy reports for this connection, or None.

        The engine's ``dialect.name`` (``postgresql``, ``mysql``, ``mssql``):
        this connector's ``db_type`` is ``sqlalchemy`` whatever it reaches, so
        the backend is the only thing that says which grammar it speaks.
        ``None`` until ``connect`` has run.
        """
        if self._engine is None:
            return None
        return str(self._engine.dialect.name)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials["url"]``.

        The URL is a standard SQLAlchemy URL, e.g.
        ``postgresql+psycopg://user:pass@host/db`` or
        ``mysql+pymysql://user:pass@host/db``. Raises :class:`ValueError` when
        no URL is supplied, and for a SQLite or DuckDB URL, which have
        dedicated connectors.

        MySQL and MariaDB URLs are opened with ``LOAD DATA LOCAL INFILE``
        disabled and the multi-statements client flag removed; see
        :func:`_harden_mysql`.

        TLS settings configured alongside the URL are applied when the driver
        is one whose parameter names we know; see :func:`_tls_connect_args`. A
        setting we cannot apply raises rather than being ignored.
        """
        url = credentials.get("url") or credentials.get("connection_string")
        if not url or not str(url).strip():
            raise ValueError("SQLAlchemyConnector requires a 'url' credential (a SQLAlchemy URL).")
        url = str(url).strip()
        parsed = make_url(url)
        _refuse_dedicated_backend(parsed)
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
        hardened, mysql_args = _harden_mysql(parsed)
        connect_args.update(mysql_args)
        # Only a MySQL URL is rebuilt; any other reaches the engine exactly as
        # it was given.
        engine_url = hardened if mysql_args else url
        self._engine = create_engine(engine_url, pool_pre_ping=True, connect_args=connect_args)
        if parsed.get_backend_name() in _MYSQL_BACKENDS:
            watch_engine(self._engine, inspect_mysql_identity, self._set_identity)

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("SQLAlchemyConnector.connect() must be called before use.")
        return self._engine

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds against the connected database."""
        return self.connection_problem() is None

    def connection_problem(self) -> str | None:
        """The driver's error if the connection test fails, else ``None``."""
        try:
            with self._require_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("SQLAlchemyConnector.test_connection failed")
            return short_cause(exc)

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the default schema via SQLAlchemy's dialect-agnostic Inspector.

        Row counts and column descriptions are left ``None`` — neither is
        portably available across dialects without extra per-table queries.
        A table whose columns cannot be read is skipped rather than aborting the
        build; a table whose *keys* cannot be read is kept without them.
        """
        engine = self._require_engine()
        inspector = inspect(engine)
        # Several dialects (PostgreSQL, Oracle, Snowflake) return identifiers as
        # `quoted_name`, a `str` subclass. They are made plain `str` here, as
        # the SQLite connector does, so that nothing downstream -- above all
        # `yaml.safe_dump` of the knowledge base -- meets a type it refuses.
        default_schema = str(inspector.default_schema_name or "")
        tables: list[TableSpec] = []
        seen = 0
        keyless: set[str] = set()
        # Which kind of key was refused, and the first reason given for that kind.
        refused: dict[str, str] = {}

        for table_name in inspector.get_table_names():
            seen += 1
            try:
                reflected_columns = inspector.get_columns(table_name)
            except Exception as exc:  # noqa: BLE001 - one bad table never breaks the reflect
                # This is the only path that still drops a table, so it is the
                # one an operator most needs a reason from.
                logger.warning(
                    "SQLAlchemyConnector: could not reflect table %r, skipping it. "
                    "The driver said: %s",
                    table_name,
                    str(exc) or exc.__class__.__name__,
                )
                continue

            # Keys are an enrichment; the columns are the schema. Reflecting them
            # used to sit inside the same `try` as the columns, so a catalogue
            # that refuses constraint metadata -- a Snowflake share, a role
            # without the grant -- cost the whole table, and every table failing
            # the same way returned an empty schema the caller cannot tell from a
            # database with no tables.
            #
            # Primary and foreign keys get a `try` each because they are separate
            # catalogue reads on most dialects and a role can hold one grant and
            # not the other. Sharing one would discard a primary key that had
            # already been read because the foreign keys were refused after it.
            # `DatabaseError` rather than `Exception`, for the reason
            # `MSSQLConnector._enrichment` gives: it covers the permission and
            # missing-object cases a restricted catalogue raises, and leaves a
            # programming error in this module or a connection that has gone to
            # raise as before. Catching everything here would report a genuine
            # fault as "the catalogue refused foreign keys" and then state, as
            # fact, that the table has no keys.
            table_refused: dict[str, str] = {}
            try:
                pk_cols = set(
                    inspector.get_pk_constraint(table_name).get("constrained_columns") or []
                )
            except DatabaseError as exc:
                pk_cols = set()
                table_refused["primary keys"] = str(exc) or exc.__class__.__name__

            try:
                fk_cols = self._foreign_key_map(inspector, table_name)
            except DatabaseError as exc:
                fk_cols = {}
                table_refused["foreign keys"] = str(exc) or exc.__class__.__name__

            try:
                columns = [
                    ColumnSpec(
                        name=str(col["name"]),
                        type=str(col.get("type")),
                        nullable=bool(col.get("nullable", True)),
                        is_primary_key=col["name"] in pk_cols,
                        is_foreign_key=col["name"] in fk_cols,
                        references=fk_cols.get(col["name"]),
                        description=col.get("comment"),
                    )
                    for col in reflected_columns
                ]
                tables.append(
                    TableSpec(
                        name=str(table_name),
                        schema=default_schema,
                        row_count=None,
                        columns=columns,
                        description=None,
                    )
                )
                # Recorded only once the table is known to be kept -- a table
                # that was refused its keys and then dropped below is not one
                # the operator got back.
                if table_refused:
                    keyless.add(table_name)
                    for kind, reason in table_refused.items():
                        refused.setdefault(kind, reason)
            except Exception as exc:  # noqa: BLE001 — one bad table never breaks the reflect
                logger.warning(
                    "SQLAlchemyConnector: could not reflect table %r, skipping it. "
                    "The driver said: %s",
                    table_name,
                    str(exc) or exc.__class__.__name__,
                )

        if keyless:
            # `seen`, not `len(tables)`: how widespread the refusal was is the
            # figure worth having, and a run that also dropped tables would
            # otherwise report "2 of 2" out of five. Each kind carries its own
            # reason, since the two can be refused for different ones.
            logger.warning(
                "SQLAlchemyConnector: kept %d of %d table(s) without complete key metadata -- "
                "the catalogue refused %s",
                len(keyless),
                seen,
                "; ".join(f"{kind} ({reason})" for kind, reason in sorted(refused.items())),
            )

        return SchemaSpec(
            database=str(engine.url.database or ""),
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    @staticmethod
    def _foreign_key_map(inspector: Any, table_name: str) -> dict[str, str]:
        """Return ``{column: "ref_table.ref_column"}`` for *table_name*."""
        fk_cols: dict[str, str] = {}
        for fk in inspector.get_foreign_keys(table_name):
            ref_table = fk.get("referred_table")
            referred = fk.get("referred_columns") or []
            constrained = fk.get("constrained_columns") or []
            for i, col in enumerate(constrained):
                ref_col = referred[i] if i < len(referred) else ""
                fk_cols[col] = f"{ref_table}.{ref_col}"
        return fk_cols

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
            try:
                return self._run_once(sql, effective_timeout, max_rows, start)
            except Exception as exc:
                delay = config.ORACLE_SNAPSHOT_RETRY_SECONDS
                retry = _is_snapshot_refusal(exc) and self._dialect_name() == "oracle"
                if not retry or delay <= 0:
                    raise
                # A table truncated or created just before this read-only
                # transaction began (see _apply_read_only). A fresh one straight
                # away is refused too, so wait, then try once more.
                logger.info(
                    "SQLAlchemyConnector: ORA-01466 (a table changed just before the "
                    "query); retrying once in %.1f s.",
                    delay,
                )
                time.sleep(delay)
                return self._run_once(sql, effective_timeout, max_rows, start)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("SQLAlchemyConnector.execute_query failed")
            return QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=elapsed_ms, error=str(exc)
            )

    def _dialect_name(self) -> str:
        return self._require_engine().dialect.name

    def _run_once(
        self,
        sql: str,
        effective_timeout: float | None,
        max_rows: int | None,
        start: float,
    ) -> QueryResult:
        """One attempt at *sql* in its own read-only, rolled-back transaction.

        Raises on failure; ``_execute_query`` turns that into ``QueryResult.error``.
        """
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


def _is_snapshot_refusal(exc: BaseException) -> bool:
    """ORA-01466: "unable to read data - table definition has changed"."""
    return "ORA-01466" in str(exc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
