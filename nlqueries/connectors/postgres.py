"""
nlqueries.connectors.postgres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PostgreSQL implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on SQLAlchemy with the ``psycopg2`` driver. This module is part of the
public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, Engine

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    POLICY_ROW,
    ColumnSpec,
    DatabaseConnector,
    QueryHistoryUnavailable,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    SecurityPolicy,
    SecurityPolicyReport,
    TableSpec,
    short_cause,
)
from nlqueries.connectors.postgres_identity import PostgresIdentity, inspect_identity
from nlqueries.connectors.postgres_tls import TlsPosture, describe, resolve_ssl_mode
from nlqueries.telemetry import get_tracer

logger = logging.getLogger(__name__)

# Rows per network round trip while streaming. Small enough that a wide
# result cannot blow the budget between checks, large enough that a narrow
# one is not paying a round trip per handful of rows.
_YIELD_PER = 1000

# Statements that can be read through a server-side cursor. Deliberately a
# check of the first word rather than a parse: it runs on every query, and the
# cost of being wrong is only that a read is not streamed.
_STREAMABLE_KEYWORDS = frozenset({"select", "with", "table", "values"})

_FIRST_WORD = re.compile(r"\w+")


def _skip_leading_noise(sql: str) -> str:
    """*sql* from its first word: past whitespace, ``(`` and comments.

    Block comments are skipped by counting, because Postgres nests them: the
    first ``*/`` in ``/* a /* b */ c */`` does not end the comment. An
    unterminated comment runs to the end, leaving nothing.
    """
    i, n = 0, len(sql)
    while i < n:
        if sql[i].isspace() or sql[i] == "(":
            i += 1
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end < 0 else end + 1
        elif sql.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        else:
            break
    return sql[i:]


def _streamed_rows(conn: Any, result: Any, deadline: float | None) -> Iterator[Any]:
    """The rows of a streamed *result*, :data:`_YIELD_PER` to a FETCH.

    ``statement_timeout`` bounds each statement, and a streamed read is a
    DECLARE and then a FETCH per batch. Set once, it let every batch run the
    whole budget again: on PostgreSQL 16 a 2 s timeout let a 6 s read finish.
    Before each batch it is set to what is left of the read's budget
    (*deadline*, on :func:`time.monotonic`), so the server ends the read as a
    whole once the budget is spent, with the error a read that is not
    streamed gets. With no budget the batches are fetched as they are.

    ``set_config(..., true)`` is ``SET LOCAL`` as a function call.
    ``pg_stat_statements`` keeps a row for each value a SET is given
    (PostgreSQL 15 and 16 alike), so a SET per batch would leave a row per
    batch in a fixed-size view, crowding out the database's own statements; a
    SELECT is normalised to one row. ``query_filter`` drops it from history.
    """
    # iter(): partitions() is a generator, so this changes nothing for a real
    # result, but next() on anything else need not ever stop.
    batches = iter(result.partitions(_YIELD_PER))
    while True:
        if deadline is not None:
            left_ms = max(1, int((deadline - time.monotonic()) * 1000))
            conn.execute(
                text("SELECT set_config('statement_timeout', :ms, true)"), {"ms": str(left_ms)}
            )
        batch = next(batches, None)
        if batch is None:
            return
        yield from batch


def _returns_rows(sql: str) -> bool:
    """True when *sql* is a read that psycopg2 can stream.

    Leading comments say nothing about what a statement does, and the SQL
    console opens every statement with one, its cancellation tag
    (``/* nlq-console:<id> */``). Checking the first word of the raw text
    streamed none of its queries, so each one was loaded whole before the
    result budget saw a row.
    """
    word = _FIRST_WORD.match(_skip_leading_noise(sql))
    return word is not None and word.group().lower() in _STREAMABLE_KEYWORDS


# Schemas that are part of Postgres / the catalog itself, never user data.
_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")

#: Leading whitespace, block comments and line comments, as one prefix run.
#:
#: The block-comment branch is the classic non-nesting C-comment pattern, not
#: ``/\*.*?\*/``. Postgres fixes the length of the whole match from the overall
#: RE's greediness and only then lets each subexpression divide that length up,
#: so a non-greedy ``.*?`` inside a greedy total still ran from the first ``/*``
#: to the **last** ``*/`` in the statement: ``/* a */ SELECT x FROM t /* b */``
#: stripped to nothing and was rejected as not a read. ``([^*]|\*+[^*/])*``
#: cannot pass a ``*/``, so the match ends where the comment does.
_LEADING_NOISE = r"^(\s+|/\*([^*]|\*+[^*/])*\*+/|--[^\n]*)+"

#: The head of a server-side cursor's declaration, up to its statement. A
#: streamed read is recorded in ``pg_stat_statements`` only as this, with the
#: statement after it, plus a FETCH and a CLOSE.
_CURSOR_DECLARATION = (
    r'^DECLARE\s+("[^"]*"|[^\s"]+)\s+((NO\s+)?SCROLL\s+|BINARY\s+|(A|IN)SENSITIVE\s+)*'
    r"CURSOR\s+(WITH(OUT)?\s+HOLD\s+)?FOR\s+"
)

#: The tag nlqueries' own callers open a statement with to identify a run:
#: ``/* nlq-console:<id> */``, ``/* nlq-pinboard:<id> */`` and the like.
_RUN_TAG = r"/\*\s*nlq-[a-z-]+:"

#: A ``pg_stat_statements`` row's statement, as an expression over a column
#: named ``query``. A cursor declared for one of our tagged runs reads as the
#: run's statement: those are queries someone ran deliberately, in the SQL
#: console or on a pinboard, and once they stream the declaration is all that
#: is recorded of them. Every other row, every other cursor included, is left
#: as recorded.
STATEMENT_TEXT = (
    f"CASE WHEN query ~* '{_CURSOR_DECLARATION}{_RUN_TAG}' "
    f"THEN regexp_replace(query, '{_CURSOR_DECLARATION}', '', 'i') ELSE query END"
)

#: Whether a ``pg_stat_statements`` row is a read, as an expression over a
#: column named ``query`` -- so a test can bind statements to it and let
#: Postgres answer. Spelled with ``[^a-z]`` rather than a word boundary because
#: ``\b`` is a *backspace* in Postgres' regex flavour, not ``\y``.
#:
#: Loose on purpose. It exists to keep DDL, DML and transaction control out of a
#: bounded result; whatever survives is still parsed and filtered downstream. The
#: cost of being slightly generous is a parse. The cost of being slightly strict
#: is a query nobody ever sees again.
STATEMENT_IS_A_READ = (
    f"regexp_replace({STATEMENT_TEXT}, '{_LEADING_NOISE}', '') ~* '^(select|with)[^a-z]'"
)


class PostgresConnector(DatabaseConnector):
    """Connector for PostgreSQL databases.

    Usage::

        connector = PostgresConnector()
        connector.connect({
            "host": "localhost",
            "port": 5432,
            "database": "mydb",
            "user": "alice",
            "password": "YOUR_PASSWORD",
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._identity: PostgresIdentity | None = None
        self._tls: TlsPosture | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a SQLAlchemy engine from ``credentials``.

        Expected keys: ``host``, ``port``, ``database``, ``user``, ``password``.
        ``host`` defaults to ``"localhost"`` and ``port`` to ``5432`` when omitted.

        Optional ``application_name`` names every session this connector's
        engine opens, in ``pg_stat_activity``. A caller that cancels a running
        statement by finding it there needs it: while a streamed read runs, the
        session's text is the cursor's ``FETCH``, not the statement. The name
        belongs to the engine, not to one statement, so a caller that cancels by
        it builds a connector for the run and names it after that run, as the
        enterprise SQL console does. A connector shared between runs shares the
        name, and a cancel by it reaches every run in flight. Postgres keeps
        the first 63 bytes of a name.
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
        # `require`, not psycopg2's `prefer`. Under `prefer` a server that
        # offers no TLS gets a plaintext session instead, silently — an audit
        # confirmed exactly that against a TLS-less server, and there is no
        # signal anywhere that the credentials and every row of every result
        # crossed the network in the clear.
        #
        # `require` is the smallest change that removes the silent downgrade,
        # not the end of this work: it insists on encryption but verifies no
        # certificate, so it stops passive eavesdropping and not an active
        # man-in-the-middle. `verify-full` is the production setting, and
        # making it mandatory is its own piece of work.
        #
        # A database with no TLS now fails to connect rather than
        # downgrading silently. `ssl_mode: disable` expresses the previous
        # default explicitly in the connector's configuration.
        ssl_mode = resolve_ssl_mode(credentials)
        connect_args["sslmode"] = ssl_mode
        self._tls = describe(credentials)
        if ssl_ca_cert := credentials.get("ssl_ca_cert"):
            connect_args["sslrootcert"] = ssl_ca_cert
        if ssl_client_cert := credentials.get("ssl_client_cert"):
            connect_args["sslcert"] = ssl_client_cert
        if ssl_client_key := credentials.get("ssl_client_key"):
            connect_args["sslkey"] = ssl_client_key
        if application_name := credentials.get("application_name"):
            connect_args["application_name"] = str(application_name)
        # Pooling only pays off now that connectors are reused across
        # requests (see loader's cache). Each cached connector holds up to
        # pool_size + max_overflow connections, so the ceiling per API worker is
        # connectors_in_cache * (CONNECTOR_POOL_SIZE + CONNECTOR_MAX_OVERFLOW).
        # Consider that ceiling before raising either value.
        # pool_recycle keeps a connection from outliving a firewall or proxy idle
        # timeout, which otherwise surfaces as a random dead connection.
        self._engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=config.CONNECTOR_POOL_SIZE,
            max_overflow=config.CONNECTOR_MAX_OVERFLOW,
            pool_recycle=1800,
            connect_args=connect_args,
        )
        self._watch_identity(self._engine)

    def _watch_identity(self, engine: Engine) -> None:
        """Read the connected role's privileges once per physical connection.

        Attached to the pool's ``connect`` event rather than executed per query.
        The event fires when a backend connection is opened, which is when the
        result can change. Cost measured against PostgreSQL 16: 2 ms on a cold
        connection, 0.3 ms warm.

        The result is recorded and reported. It is not enforced here: a
        read-only transaction restricts the operations a statement performs but
        not the role it executes as, and ``pg_read_file()`` is refused by
        privilege rather than by the transaction.
        """

        @event.listens_for(engine, "connect")
        def _inspect(dbapi_connection: Any, _record: Any) -> None:
            identity = inspect_identity(dbapi_connection)
            self._identity = identity
            if identity.undetermined_reason is not None:
                logger.warning("PostgreSQL identity check: %s", identity.summary())
            elif identity.concerns:
                logger.warning(
                    "PostgreSQL identity is more privileged than this connector needs: %s. "
                    "See docs/database-hardening.md for the role to create instead.",
                    identity.summary(),
                )
            else:
                logger.debug("PostgreSQL identity check: %s", identity.summary())

    @property
    def tls(self) -> TlsPosture | None:
        """The TLS posture of this connection, or None before ``connect``."""
        return self._tls

    @property
    def identity(self) -> PostgresIdentity | None:
        """The most recent identity read, or None if no connection has opened.

        The pool opens connections lazily, so None indicates that no check has
        run rather than that a check found no concerns.
        """
        return self._identity

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("PostgresConnector.connect() must be called before use.")
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
        except Exception as exc:  # noqa: BLE001 — any failure means "not connected"
            logger.exception("PostgresConnector.test_connection failed")
            return short_cause(exc)

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
            database: str = conn.execute(text("SELECT current_database()")).scalar_one()

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

    # Keys are read from pg_catalog, not information_schema. PostgreSQL shows
    # information_schema.table_constraints and key_column_usage only for tables
    # the login owns or holds a privilege other than SELECT on, and
    # constraint_column_usage only for tables it owns. The login
    # docs/database-hardening.md recommends holds SELECT alone, so it saw no
    # keys at all: every column read as no key, and no join was known.
    # pg_constraint is readable by every login, so each query keeps to tables
    # the login may use, as information_schema.columns does for the columns.
    @staticmethod
    def _fetch_primary_keys(conn: Any) -> dict[tuple[str, str], set[str]]:
        """Return ``{(schema, table): {primary key column names}}``."""
        rows = conn.execute(
            text(
                """
                SELECT
                    n.nspname AS table_schema,
                    c.relname AS table_name,
                    a.attname AS column_name
                FROM pg_catalog.pg_constraint con
                JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN LATERAL unnest(con.conkey) AS k(attnum) ON TRUE
                JOIN pg_catalog.pg_attribute a
                    ON a.attrelid = c.oid AND a.attnum = k.attnum
                WHERE con.contype = 'p'
                  AND n.nspname NOT IN :system_schemas
                  AND has_any_column_privilege(c.oid, 'SELECT, INSERT, UPDATE, REFERENCES')
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
                    n.nspname AS table_schema,
                    c.relname AS table_name,
                    a.attname AS column_name,
                    fc.relname AS foreign_table_name,
                    fa.attname AS foreign_column_name
                FROM pg_catalog.pg_constraint con
                JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_class fc ON fc.oid = con.confrelid
                -- The two arrays pair each column with the one it references,
                -- by position, so a composite key maps column to column.
                JOIN LATERAL unnest(con.conkey, con.confkey) AS k(attnum, fattnum) ON TRUE
                JOIN pg_catalog.pg_attribute a
                    ON a.attrelid = c.oid AND a.attnum = k.attnum
                JOIN pg_catalog.pg_attribute fa
                    ON fa.attrelid = fc.oid AND fa.attnum = k.fattnum
                WHERE con.contype = 'f'
                  AND n.nspname NOT IN :system_schemas
                  AND has_any_column_privilege(c.oid, 'SELECT, INSERT, UPDATE, REFERENCES')
                  -- Nor to a table it may not use: the knowledge base would then
                  -- offer a join to a table that is not in the schema.
                  AND has_any_column_privilege(fc.oid, 'SELECT, INSERT, UPDATE, REFERENCES')
                  -- A key into a partitioned table gets a copy per partition of
                  -- that table (same referencing table, conparentid set), which
                  -- would name a partition in place of the table. A copy a
                  -- partition of the referencing table inherits has its own
                  -- conrelid and is kept.
                  AND NOT EXISTS (
                      SELECT 1 FROM pg_catalog.pg_constraint parent
                      WHERE parent.oid = con.conparentid
                        AND parent.conrelid = con.conrelid
                  )
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
        """Return the top ``pg_stat_statements`` reads for *this* database.

        ``pg_stat_statements`` accumulates statistics since the extension's
        last reset rather than tracking individual execution timestamps, so
        ``days`` cannot be used as a precise SQL filter; it is accepted for
        interface compatibility and to size logging/messaging. Up to
        ``limit`` queries are returned, ordered by execution count
        (``calls``) descending.

        Two filters decide what that budget is spent on, and both were added
        after measuring a live deployment where it was spent on neither.

        **This database only.** ``pg_stat_statements`` is cluster-wide. Without
        the ``dbid`` filter, extracting one connector's history returns every
        other database's SQL on the same server — 6,076 of 9,880 entries, in the
        case that prompted this — which is both noise and somebody else's
        queries. ``LIMIT`` is applied before any caller can tell them apart.

        **Reads only.** The view records utility statements too, and each
        streamed read declares a server-side cursor under a name SQLAlchemy
        generates fresh every time (``c_7f1449704a10_76``), so ``DECLARE``,
        ``FETCH`` and ``CLOSE`` each take a permanent entry that never
        normalises. On that same deployment 3,750 of this database's 3,782
        entries were exactly that, and only **75** entries cluster-wide had
        eight calls or more — so 425 of a 500-row budget went on cursor
        bookkeeping and transaction control before any real query was reached.
        Operators who want those out of the view entirely can set
        ``pg_stat_statements.track_utility = off``; this makes the extraction
        immune either way.

        Except our own tagged runs. A run the SQL console or a pinboard tags
        (``/* nlq-console:<id> */``) streams like any read, so a cursor
        declared for one is returned as its statement (:data:`STATEMENT_TEXT`).
        Each such run is a row of its own, with one call: the cursor's name
        changes every time, so they are never counted together. Its time is the
        declaration's, not the read's. Rows are still ordered by calls and cut
        at *limit*, so where the database's reads outnumber the limit, these
        one-call rows are the first left out: the history holds the tagged runs
        that fit, not every one the console ran.

        The prefix test skips leading whitespace and comments rather than
        anchoring on the first character. Our own SQL Console tags every
        statement it runs with ``/* nlq-console:<uuid> */``, and any tool that
        annotates its SQL does the same — anchoring would discard precisely the
        queries someone ran deliberately.

        If the ``pg_stat_statements`` extension is not installed, or the login
        may not read it, this raises :class:`QueryHistoryUnavailable` saying
        so -- an empty list would read as "no history" when the history simply
        cannot be read. The usual permission failure is a ``REVOKE EXECUTE ON
        ALL FUNCTIONS IN SCHEMA ... FROM PUBLIC`` on the schema the extension
        lives in, which takes ``pg_stat_statements()`` away from every login.
        """
        engine = self._require_engine()

        with engine.connect() as conn:
            installed = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'")
            ).scalar_one_or_none()

            if not installed:
                raise QueryHistoryUnavailable(
                    "The pg_stat_statements extension is not installed on this "
                    "database, so its query history cannot be read. Install it with "
                    "`CREATE EXTENSION pg_stat_statements;` (it must also be listed "
                    "in shared_preload_libraries)."
                )

            # Postgres < 13 named the column `mean_time`. Chosen from the server's
            # version: retrying on the error instead ran the second statement in
            # the transaction the first had aborted, so every failure read as
            # "current transaction is aborted" and hid its cause.
            server_version = conn.dialect.server_version_info or (13,)
            mean_time = "mean_exec_time" if server_version >= (13,) else "mean_time"
            try:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT {STATEMENT_TEXT} AS query, calls, {mean_time} AS mean_exec_time
                        FROM pg_stat_statements
                        WHERE dbid = (
                            SELECT oid FROM pg_database WHERE datname = current_database()
                        )
                          AND {STATEMENT_IS_A_READ}
                        ORDER BY calls DESC
                        LIMIT :limit
                        """  # noqa: S608 - module constants; the limit is bound
                    ),
                    {"limit": limit},
                )
                records = list(rows.mappings())
            except Exception as exc:  # noqa: BLE001
                raise QueryHistoryUnavailable(
                    f"Could not read pg_stat_statements ({short_cause(exc)}). If that is "
                    "a permission error, the login needs EXECUTE on the "
                    "pg_stat_statements() function, which REVOKE EXECUTE ON ALL FUNCTIONS "
                    "... FROM PUBLIC on the extension's schema removes: GRANT EXECUTE ON "
                    "FUNCTION pg_stat_statements(boolean) TO <login>."
                ) from exc

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
                for row in records
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
                    # First statement in the transaction, and it has to be: this
                    # is what stops a SELECT from writing.
                    #
                    # Generated SQL is an LLM's output, and every validator in
                    # front of this one asks only whether the root node is a
                    # Select. `SELECT some_volatile_function(...)` satisfies
                    # that and can still INSERT — which is exactly what a
                    # security audit reproduced through this connector, twice,
                    # eight weeks apart. `engine.begin()` then committed it.
                    #
                    # PostgreSQL's own guard is broader than the validator's
                    # could be, because it applies to whatever the statement
                    # eventually does rather than to how it is spelled: DML and
                    # DDL anywhere in the call graph, and sequence functions,
                    # which it refuses by name.
                    #
                    # It is not the whole boundary, and should not be read as
                    # one. A read-only transaction still permits advisory locks,
                    # sleeps and privileged file reads; refusing those needs a
                    # least-privilege role, which only the operator can grant.
                    conn.execute(text("SET TRANSACTION READ ONLY"))

                    deadline: float | None = None
                    if effective_timeout is not None and effective_timeout > 0:
                        statement_timeout_ms = max(1, int(effective_timeout * 1000))
                        conn.execute(text(f"SET LOCAL statement_timeout = {statement_timeout_ms}"))
                        deadline = time.monotonic() + effective_timeout

                    # Server-side cursor for reads: rows arrive in batches
                    # instead of the driver building the entire result before
                    # returning. Safe here because the connection is opened per
                    # query and the transaction stays open for the whole read,
                    # which is what psycopg2's streaming mode requires.
                    #
                    # Only for reads. psycopg2 cannot run DDL or DML through a
                    # named cursor at all — it fails outright — so streaming is
                    # requested only for statements that plainly return rows.
                    # Misjudging one costs streaming, never correctness: the
                    # budget below still bounds what is turned into Python
                    # objects either way.
                    # For this statement only: Connection.execution_options()
                    # changes the connection, and the SETs below must not be
                    # sent through a cursor of their own.
                    streamed = _returns_rows(sql)
                    cursor_result = conn.execute(
                        text(sql), execution_options={"stream_results": streamed}
                    )

                    truncated = False
                    reason: str | None = None
                    if cursor_result.returns_rows:
                        columns = list(cursor_result.keys())
                        source = (
                            _streamed_rows(conn, cursor_result, deadline)
                            if streamed
                            else iter(cursor_result)
                        )
                        rows, truncated, reason = collect(source, max_rows)
                        if truncated:
                            # Stop the server sending the rest of a result we
                            # have already decided not to read.
                            cursor_result.close()
                            logger.warning("Result truncated (%s) after %d rows", reason, len(rows))
                    else:
                        columns = []
                        rows = []
                    # After the rows: a streamed read's execute returns once the
                    # cursor is declared, and the read happens in the batches.
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    span.set_attribute("db.row_count", len(rows))
                    span.set_attribute("db.result_truncated", truncated)
                    return QueryResult(
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        execution_time_ms=elapsed_ms,
                        error=None,
                        truncated=truncated,
                        truncation_reason=reason,
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

    def list_security_policies(self) -> SecurityPolicyReport:
        """Introspect Postgres Row-Level Security policies via ``pg_catalog.pg_policies``.

        Each policy's ``USING`` predicate (``qual``) becomes a :data:`POLICY_ROW`
        policy the caller can translate to a row filter. ``pg_policies`` shows only
        policies on tables the connection role can see, and is world-readable, so
        this degrades to an empty (but supported) report on any read error rather
        than raising. Policies with no ``USING`` clause (``WITH CHECK`` only, i.e.
        write-time policies) are skipped — they don't restrict reads.
        """
        policies: list[SecurityPolicy] = []
        try:
            engine = self._require_engine()
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT schemaname, tablename, policyname, roles, qual
                        FROM pg_catalog.pg_policies
                        WHERE schemaname NOT IN :system_schemas
                        ORDER BY schemaname, tablename, policyname
                        """
                    ),
                    {"system_schemas": _SYSTEM_SCHEMAS},
                )
                for row in rows.mappings():
                    qual = row["qual"]
                    if not qual:  # WITH CHECK-only policy — doesn't restrict reads
                        continue
                    policies.append(
                        SecurityPolicy(
                            name=row["policyname"],
                            kind=POLICY_ROW,
                            table_schema=row["schemaname"],
                            table=row["tablename"],
                            columns=[],
                            expression=str(qual),
                            roles=_pg_roles(row["roles"]),
                        )
                    )
        except Exception:  # noqa: BLE001 — introspection is best-effort; degrade to empty
            logger.warning("PostgresConnector.list_security_policies failed", exc_info=True)
            return SecurityPolicyReport(supported=True, policies=[])
        return SecurityPolicyReport(supported=True, policies=policies)


def _pg_roles(raw: Any) -> list[str]:
    """Normalise the ``pg_policies.roles`` value (a Postgres text[]) to a list.

    The driver may hand it back as a Python list already, or as the raw array
    literal ``{role_a,role_b}`` / ``{public}``; handle both.
    """
    if isinstance(raw, list):
        return [str(r) for r in raw]
    if isinstance(raw, str):
        return [r for r in raw.strip("{}").split(",") if r]
    return []


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
