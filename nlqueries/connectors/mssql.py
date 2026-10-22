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
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import DatabaseError

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryHistoryUnavailable,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
    short_cause,
)
from nlqueries.connectors.login_identity import LoginIdentity, watch_engine
from nlqueries.connectors.mssql_identity import inspect_identity as inspect_mssql_identity

logger = logging.getLogger(__name__)

# SQL Server built-in schemas that contain no user data.
_SYSTEM_SCHEMAS = frozenset({"sys", "INFORMATION_SCHEMA", "guest", "db_owner"})

#: The values a connector entry's ``encryption`` key may take, each passed to
#: pymssql's ``encryption`` argument unchanged. pymssql also accepts
#: ``default``, which defers to FreeTDS's own configuration; it is left out so
#: that the posture `nlqueries health` reports is one this connector chose
#: rather than one it cannot see.
ENCRYPTION_LEVELS = ("require", "request", "off")

#: Applied when the entry sets no ``encryption``. It changes nothing for an
#: existing deployment: measured against SQL Server 2022, which offers TLS
#: without forcing it, ``request`` connects with the session unencrypted
#: (``sys.dm_exec_connections.encrypt_option`` reads FALSE), as no setting did.
#: Only ``require`` encrypted the session, so it is the setting to choose once
#: the server is known to offer TLS; it refuses a server that does not.
DEFAULT_ENCRYPTION = "request"

#: The first pymssql release that applies ``encryption``. Measured against SQL
#: Server 2022: 2.3.0 to 2.4.2 accept the argument and connect unencrypted
#: under ``require``; 2.4.3 to 2.4.5 encrypt.
PYMSSQL_APPLIES_ENCRYPTION = (2, 4, 3)


def _pymssql_applies_encryption(version: object) -> bool:
    """Whether the pymssql *version* string is one that applies ``encryption``.

    A version that cannot be read is taken as one that does not, so the posture
    reported is never stronger than the one in force.
    """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return False
    return tuple(int(part) for part in match.groups()) >= PYMSSQL_APPLIES_ENCRYPTION


@dataclass(frozen=True)
class EncryptionPosture:
    """The transport encryption a SQL Server connection was opened with.

    Duck-types the two members `nlqueries health` reads from a connector's
    ``tls`` -- :attr:`is_fully_verified` and :meth:`summary` -- so SQL Server
    gets a transport line the way PostgreSQL does.

    Never fully verified. pymssql's ``encryption`` argument decides whether the
    channel is encrypted; whether the server's certificate is checked is FreeTDS
    configuration (``ca file`` in ``freetds.conf``), which this connector does
    not set and cannot read back. Under ``require`` it connected to a server
    presenting a self-signed certificate. Reporting it as verified would be a
    claim nothing here has checked.
    """

    encryption: str
    #: False when the installed pymssql predates :data:`PYMSSQL_APPLIES_ENCRYPTION`
    #: and so ignores the setting.
    applied: bool = True
    driver_version: str = ""

    @property
    def concerns(self) -> tuple[str, ...]:
        if not self.applied:
            return (
                f"pymssql {self.driver_version or '(unknown version)'} ignores the "
                f"encryption setting, so the session is unencrypted unless the server "
                f"forces encryption; install pymssql 2.4.3 or later",
            )
        if self.encryption == "off":
            return ("encryption is 'off', so the session is not encrypted",)
        if self.encryption == "request":
            return (
                "encryption is 'request', which leaves the session unencrypted unless "
                "the server forces encryption; set 'require' to encrypt it",
            )
        return (
            "encryption is 'require'; the server's certificate is verified only if "
            "FreeTDS is configured with a CA file",
        )

    @property
    def is_fully_verified(self) -> bool:
        return False

    def summary(self) -> str:
        """A single line for logs and the health report."""
        return "; ".join(self.concerns)


def resolve_encryption(credentials: dict[str, Any]) -> str:
    """The ``encryption`` level for *credentials*, or :data:`DEFAULT_ENCRYPTION`.

    An unknown value is refused rather than passed on: pymssql would refuse it
    too, but only at connect time and naming its own vocabulary, which includes
    a ``default`` this connector does not offer.
    """
    raw = credentials.get("encryption")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_ENCRYPTION
    # YAML reads a bare `off` as the boolean False.
    if raw is False:
        return "off"
    value = str(raw).strip().lower()
    if value not in ENCRYPTION_LEVELS:
        raise ValueError(
            f"MSSQLConnector: encryption must be one of {', '.join(ENCRYPTION_LEVELS)}, "
            f"not {raw!r}."
        )
    return value


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
        self._tls: EncryptionPosture | None = None
        self._identity: LoginIdentity | None = None

    @property
    def tls(self) -> EncryptionPosture | None:
        """The encryption this connector opened with, or None before ``connect``."""
        return self._tls

    @property
    def identity(self) -> LoginIdentity | None:
        """What the login holds beyond reading, from the most recent connection.

        None until the pool has opened a connection, which it does lazily: no
        check has run, rather than a check that found nothing.
        """
        return self._identity

    def _set_identity(self, identity: LoginIdentity) -> None:
        self._identity = identity

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials``.

        Required keys: ``host``, ``database``, ``user``, ``password``.
        ``port`` defaults to ``1433``. ``encryption`` is one of
        :data:`ENCRYPTION_LEVELS` and defaults to :data:`DEFAULT_ENCRYPTION`.

        Raises :class:`ImportError` when ``pymssql`` is not installed.
        """
        try:
            import pymssql  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'pymssql' package is required for SQL Server connections.\n"
                "Install it with:  pip install 'nlqueries-core[mssql]'"
            ) from exc

        encryption = resolve_encryption(credentials)
        driver_version = str(getattr(pymssql, "__version__", ""))
        applied = _pymssql_applies_encryption(driver_version)
        if encryption == "require" and not applied:
            # Refused rather than opened: this release accepts the argument and
            # connects unencrypted, so the connection would not be what the
            # entry asks for, and health would have to report the opposite.
            raise RuntimeError(
                f"MSSQLConnector: encryption is 'require', but pymssql "
                f"{driver_version or '(unknown version)'} ignores that setting and "
                f"would connect unencrypted. Install pymssql 2.4.3 or later: "
                f"pip install 'nlqueries-core[mssql]'"
            )
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
        # Passed on every connection, including the default. Left to FreeTDS,
        # the level depended on whichever freetds.conf the driver happened to
        # find, so the same entry could connect in plaintext on one host and
        # encrypted on another, and nothing here could say which.
        connect_args: dict[str, Any] = {"encryption": encryption}
        if config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS > 0:
            connect_args["timeout"] = max(1, int(config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS))
        self._tls = EncryptionPosture(encryption, applied, driver_version)
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        watch_engine(self._engine, inspect_mssql_identity, self._set_identity)

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("MSSQLConnector.connect() must be called before use.")
        return self._engine

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds against the connected database."""
        return self.connection_problem() is None

    def connection_problem(self) -> str | None:
        """The driver's error if the connection test fails, else ``None``."""
        try:
            engine = self._require_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("MSSQLConnector.test_connection failed")
            return short_cause(exc)

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
            database: str = conn.execute(text("SELECT DB_NAME()")).scalar_one()
            tables_meta = self._fetch_tables(conn)
            cols_by_table = self._fetch_columns(conn)
            # Keys are an enrichment; tables and columns are the schema.
            # SQL Server's INFORMATION_SCHEMA views usually filter by
            # permission rather than erroring, so this is the quieter of the
            # engines -- but a login without VIEW DEFINITION on a locked-down
            # Azure SQL database can still fail the query outright, and losing
            # the whole schema over the key flags is the wrong trade.
            pks = self._enrichment("primary keys", lambda: self._fetch_primary_keys(conn), {})
            fks = self._enrichment("foreign keys", lambda: self._fetch_foreign_keys(conn), {})

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
    def _enrichment(label: str, fetch: Any, default: Any) -> Any:
        """Run an optional schema enrichment, or log and return *default*.

        Narrower than the Redshift equivalent because SQLAlchemy gives a shared
        base for driver faults: ``DatabaseError`` covers the permission and
        missing-object cases and leaves a programming error in this module, or
        a connection that has gone, to raise as before.
        """
        try:
            return fetch()
        except DatabaseError as exc:
            logger.warning(
                "MSSQLConnector: could not read %s (%s). The schema will omit them; "
                "this usually means the login lacks VIEW DEFINITION.",
                label,
                exc,
            )
            return default

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
        (Azure SQL).  Raises :class:`QueryHistoryUnavailable` if the DMV is
        inaccessible.

        Limited to plans compiled in the connected database. The DMV is
        server-wide, so without the filter a login with ``VIEW SERVER STATE``
        mined every database on the instance into this connector's knowledge
        base. The database is read from the plan's ``dbid`` attribute rather
        than ``dm_exec_sql_text.dbid``, which is NULL for ad hoc and prepared
        statements -- the bulk of what an application sends.
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
                        CROSS APPLY (
                            SELECT CONVERT(int, pa.value) AS dbid
                            FROM sys.dm_exec_plan_attributes(qs.plan_handle) pa
                            WHERE pa.attribute = 'dbid'
                        ) plan_db
                        WHERE qs.last_execution_time >= DATEADD(day, :neg_days, GETDATE())
                          AND plan_db.dbid = DB_ID()
                          AND qt.text NOT LIKE '%sys.dm_exec%'
                        ORDER BY qs.execution_count DESC
                        """
                    ),
                    {"neg_days": -days},
                )
                # Read here, while the connection is still checked out. Read
                # after the `with`, the connection had gone back to the pool
                # and the result came back empty on a real server, every time.
                records = list(rows.mappings())
            except Exception as exc:  # noqa: BLE001
                raise QueryHistoryUnavailable(
                    "Could not read sys.dm_exec_query_stats "
                    f"({short_cause(exc)}). The login needs VIEW SERVER STATE, "
                    "or VIEW DATABASE STATE on Azure SQL."
                ) from exc

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
            for row in records
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
            # `connect()` with an unconditional rollback, never `begin()`.
            #
            # `begin()` commits when its block exits without an exception, and
            # every validator in front of this one asks only whether the root
            # node is a Select. `SELECT some_volatile_function(...)` satisfies
            # that and can still write: an audit reproduced exactly that through
            # the Postgres connector twice, eight weeks apart, and `begin()`
            # committed it. The same statement reached the same ending here.
            #
            # SQL Server offers no read-only transaction to ask for instead --
            # `ApplicationIntent=ReadOnly` is availability-group routing, not a
            # permission. What it does offer is transactional DDL, so a `DROP`
            # inside this block is undone by the rollback along with any DML.
            #
            # This is the second layer and not the boundary. A transaction that
            # is never committed still permits `WAITFOR DELAY`, extended
            # procedures and anything the login may already read; refusing those
            # needs a least-privilege login, which only the operator can grant.
            # See docs/database-hardening.md.
            with engine.connect() as conn:
                try:
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
                            "MSSQLConnector",
                            exc_info=True,
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
