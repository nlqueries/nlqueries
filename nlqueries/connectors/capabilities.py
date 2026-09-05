# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What each connector enforces, and what it leaves to the operator.

The read-only transaction and sandbox settings applied by the PostgreSQL,
SQLite and DuckDB connectors are not available on every engine. Where they are
not, a connector applies the most restrictive execution its engine offers --
usually a transaction that is never committed and is rolled back either way.
Only BigQuery records no mechanism at all, because a query job has no
transaction to roll back. Each entry states the mechanism used, or states that
there is none.

Two of the recorded mechanisms stop short of covering DDL, and say so: DDL is
not transactional on Snowflake, and behind the generic SQLAlchemy connector
MySQL, MariaDB and Oracle commit implicitly around it while SQLite runs it
outside the transaction.

``verified_here`` distinguishes a mechanism exercised against a real engine by
this repository's tests from one asserted against a fake driver or only
documented by the vendor. Snowflake, BigQuery, Redshift, SQL Server and the
generic SQLAlchemy connector all require engines a test run cannot provision and
are recorded as unverified.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialectCapabilities:
    """The controls a connector applies, and those it does not."""

    dialect: str

    #: How the connector prevents a statement from writing, or None if it
    #: applies nothing and the restriction must come from database privileges.
    read_only_mechanism: str | None

    #: How the connector bounds a query's runtime, or None if it applies
    #: nothing.
    statement_timeout_mechanism: str | None

    #: True when the mechanisms above are exercised by tests in this
    #: repository against a real engine.
    verified_here: bool

    #: What the operator must do for this dialect, stated even where the
    #: connector applies a mechanism of its own.
    operator_requirement: str

    @property
    def enforces_read_only(self) -> bool:
        return self.read_only_mechanism is not None

    @property
    def enforces_statement_timeout(self) -> bool:
        return self.statement_timeout_mechanism is not None

    @property
    def concerns(self) -> tuple[str, ...]:
        found: list[str] = []
        if not self.enforces_read_only:
            found.append(
                "the connector applies no read-only mechanism, so a write is "
                "prevented only by the privileges granted to the login"
            )
        if not self.enforces_statement_timeout:
            found.append("the connector applies no statement timeout")
        if self.enforces_read_only and not self.verified_here:
            found.append(
                f"the read-only mechanism ({self.read_only_mechanism}) is not "
                f"exercised by any test in this repository"
            )
        return tuple(found)

    def summary(self) -> str:
        if not self.concerns:
            return f"{self.dialect}: {self.read_only_mechanism}, verified"
        return f"{self.dialect}: " + "; ".join(self.concerns)


#: Keyed by the name used in ``CONNECTOR_REGISTRY``.
CAPABILITIES: dict[str, DialectCapabilities] = {
    "postgres": DialectCapabilities(
        dialect="postgres",
        read_only_mechanism=(
            "SET TRANSACTION READ ONLY, as the first statement of every transaction"
        ),
        statement_timeout_mechanism="SET LOCAL statement_timeout",
        verified_here=True,
        operator_requirement=(
            "A login that owns nothing, holds SELECT on an explicit list of schemas, and "
            "belongs to none of the pg_read_server_files family. The read-only transaction "
            "does not restrict privileges; see docs/database-hardening.md."
        ),
    ),
    "sqlite": DialectCapabilities(
        dialect="sqlite",
        read_only_mechanism="URI mode=ro, plus an authorizer refusing ATTACH and unlisted pragmas",
        statement_timeout_mechanism="a watchdog calling sqlite3.Connection.interrupt",
        verified_here=True,
        operator_requirement=(
            "Filesystem permissions on the database file and its directory. A read-only "
            "handle is an application-level promise; file permissions are not."
        ),
    ),
    "duckdb": DialectCapabilities(
        dialect="duckdb",
        read_only_mechanism="read_only=True, external access disabled, configuration locked",
        statement_timeout_mechanism="a watchdog cancelling the query",
        verified_here=True,
        operator_requirement=(
            "Run it where there is nothing else to read: a container with only the database "
            "file mounted, no host secrets and no egress."
        ),
    ),
    "mssql": DialectCapabilities(
        dialect="mssql",
        read_only_mechanism=(
            "the query runs on a connection that is never committed and is rolled back "
            "whether it succeeded or failed; T-SQL DDL is transactional, so a DROP is "
            "undone with it"
        ),
        statement_timeout_mechanism=None,
        verified_here=False,
        operator_requirement=(
            "SQL Server has no read-only transaction mode to ask for -- "
            "ApplicationIntent=ReadOnly is availability-group routing, not a permission -- so "
            "the rollback is the whole of what the connector can do. Grant db_datareader and "
            "nothing else: WAITFOR DELAY, extended procedures and table access are all "
            "privilege questions. The pymssql connection-level 'timeout' is applied at connect "
            "from CONNECTOR_STATEMENT_TIMEOUT_SECONDS; a per-query timeout is not implemented."
        ),
    ),
    "redshift": DialectCapabilities(
        dialect="redshift",
        read_only_mechanism=(
            "SET TRANSACTION READ ONLY, as the first statement of every transaction"
        ),
        statement_timeout_mechanism="SET statement_timeout, applied per query",
        # No test in this repository reaches a Redshift cluster; CI cannot
        # provision one. Both mechanisms were measured by hand against Redshift
        # Serverless on 2026-08-27: a write is refused with SQLSTATE 25006, and
        # a query exceeding the budget is cancelled with SQLSTATE 57014.
        verified_here=False,
        operator_requirement=(
            "A read-only user and a WLM query-monitoring rule remain worth having. The "
            "connector's read-only transaction restricts what a statement may do, not what "
            "the login may reach."
        ),
    ),
    "snowflake": DialectCapabilities(
        dialect="snowflake",
        read_only_mechanism=(
            "the query runs inside BEGIN ... ROLLBACK, so DML is undone; DDL is not "
            "transactional on Snowflake and still stands"
        ),
        statement_timeout_mechanism="cursor.execute(timeout=…), cancelled server-side",
        verified_here=False,
        operator_requirement=(
            "The rollback undoes an INSERT and does not undo a CREATE or DROP, so the grant "
            "carries more of the boundary here than on any other engine. Use a role with "
            "SELECT on explicit objects, no CREATE on any schema, and a resource monitor on "
            "the warehouse."
        ),
    ),
    "bigquery": DialectCapabilities(
        dialect="bigquery",
        # Deliberately still None. The job is pinned to standard SQL with no
        # session, and a statement type other than SELECT is logged at warning --
        # but a query job cannot be rolled back, and the type is only readable
        # after the job has run. Recording that as a read-only mechanism would
        # tell an operator they are protected when nothing was prevented.
        read_only_mechanism=None,
        statement_timeout_mechanism="job_timeout_ms, cancelled server-side",
        verified_here=False,
        operator_requirement=(
            "BigQuery has no transaction to roll back, so IAM is the whole boundary. Grant "
            "roles/bigquery.dataViewer on explicit datasets, withhold jobUser where possible, "
            "and set a maximum bytes billed. The connector logs a non-SELECT statement type at "
            "warning after the job has run; that is an audit signal, not a control."
        ),
    ),
    "sqlalchemy": DialectCapabilities(
        dialect="sqlalchemy",
        read_only_mechanism=(
            "DML on a transactional table is never committed and is rolled back either "
            "way; plus SET TRANSACTION READ ONLY where the dialect is postgresql or "
            "redshift. Two things are NOT covered: DDL, since MySQL, MariaDB and Oracle "
            "commit implicitly around it and SQLite runs it outside the transaction; and "
            "MySQL's non-transactional storage engines, where an INSERT into a MyISAM or "
            "MEMORY table survives the rollback with only warning 1196"
        ),
        statement_timeout_mechanism="a best-effort per-dialect SET, where the dialect is known",
        verified_here=False,
        operator_requirement=(
            "This connector reaches any engine SQLAlchemy supports, so the rollback is the "
            "only thing that holds everywhere. MySQL and MariaDB get no read-only transaction: "
            "their form configures subsequent transactions and is refused inside an open one, "
            "and SQLAlchemy has already opened one by the time the connector can send "
            "anything. The restriction must come from the privileges granted to the login."
        ),
    ),
}


def for_dialect(dialect: str) -> DialectCapabilities | None:
    """The capabilities recorded for *dialect*, or None if it has no entry.

    None indicates the dialect is not described here. It does not indicate that
    the dialect is safe.
    """
    return CAPABILITIES.get(dialect.lower())
