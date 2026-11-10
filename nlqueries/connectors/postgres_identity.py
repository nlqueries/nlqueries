# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.postgres_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inspection of the PostgreSQL login a connector is using.

The read-only transaction opened for every query constrains the operations a
statement may perform. It does not constrain the role the statement executes as.
Several relevant restrictions are privileges rather than transaction state:
``docs/database-hardening.md`` records that ``pg_read_file()`` is refused by
privilege, so a role holding ``pg_read_server_files`` retains access that the
application cannot withdraw.

This module reads the connected role's privileges and returns them as a report.
It does not refuse a connection. The report is surfaced through ``nlqueries
health``, where an over-privileged role is visible to the operator. Enforcement
is out of scope for this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Predefined roles whose membership grants access the application cannot
#: withdraw. ``pg_database_owner`` is excluded: it is held by the owner of any
#: database and confers no additional privilege.
DANGEROUS_PREDEFINED_ROLES = frozenset(
    {
        "pg_read_server_files",
        "pg_write_server_files",
        "pg_execute_server_program",
        "pg_read_all_data",
        "pg_write_all_data",
    }
)

#: Read in a single round trip. Predefined roles are enumerated from
#: ``pg_roles`` rather than named individually, so that a server version which
#: lacks or adds one produces neither an error nor an unreported membership.
#:
#: The last three columns are the table grants. Role attributes alone said
#: nothing about them, so a login with SELECT on a whole schema, or INSERT on a
#: table, was reported as if it held nothing. ``reachable`` is every table and
#: view in a schema the login holds USAGE on, outside the catalog's own: a table
#: privilege in a schema without USAGE cannot be exercised, so it is neither a
#: read nor a write here. Column-level grants count on both sides: `UPDATE
#: (salary)` writes as surely as `UPDATE` does, and `SELECT (id)` reads.
#:
#: A partition or inheritance child is counted through its parent when the
#: login holds the same kind of privilege there, so a schema-wide grant does not
#: count every partition, and on its own otherwise: grants are not inherited,
#: and a statement against a partition checks only the partition's, so an
#: INSERT held on one partition alone is a write.
IDENTITY_SQL = r"""
WITH reachable AS (
    SELECT
        c.oid,
        n.nspname,
        c.relname,
        (SELECT i.inhparent FROM pg_inherits i
         WHERE i.inhrelid = c.oid ORDER BY i.inhseqno LIMIT 1) AS parent
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema')
      AND n.nspname NOT LIKE 'pg\_toast%'
      AND n.nspname NOT LIKE 'pg\_temp\_%'
      AND has_schema_privilege(n.oid, 'USAGE')
),
writable AS (
    SELECT nspname, relname FROM reachable
    WHERE (has_table_privilege(oid, 'INSERT, UPDATE, DELETE, TRUNCATE')
           OR has_any_column_privilege(oid, 'INSERT, UPDATE'))
      AND NOT (parent IS NOT NULL
               AND (has_table_privilege(parent, 'INSERT, UPDATE, DELETE, TRUNCATE')
                    OR has_any_column_privilege(parent, 'INSERT, UPDATE')))
),
readable AS (
    SELECT nspname FROM reachable
    WHERE (has_table_privilege(oid, 'SELECT') OR has_any_column_privilege(oid, 'SELECT'))
      AND NOT (parent IS NOT NULL
               AND (has_table_privilege(parent, 'SELECT')
                    OR has_any_column_privilege(parent, 'SELECT')))
)
SELECT
    current_user                                          AS user_name,
    session_user                                          AS session_user_name,
    current_database()                                    AS database_name,
    current_setting('search_path')                        AS search_path,
    current_setting('default_transaction_read_only')      AS default_read_only,
    r.rolsuper                                            AS is_superuser,
    r.rolcreatedb                                         AS may_create_database,
    r.rolcreaterole                                       AS may_create_role,
    r.rolbypassrls                                        AS bypasses_rls,
    r.rolreplication                                      AS may_replicate,
    ARRAY(
        SELECT g.rolname FROM pg_roles g
        WHERE left(g.rolname, 3) = 'pg_'
          AND pg_has_role(current_user, g.oid, 'USAGE')
    )                                                     AS predefined_roles,
    ARRAY(
        SELECT quote_ident(nspname) || '.' || quote_ident(relname) FROM writable
        ORDER BY 1
    )                                                     AS writable_tables,
    (SELECT count(*) FROM readable)                       AS readable_tables,
    ARRAY(SELECT DISTINCT nspname FROM readable ORDER BY 1) AS readable_schemas
FROM pg_roles r
WHERE r.rolname = current_user
"""

#: How many writable tables a concern names before it counts the rest.
_NAMED_TABLES = 5


@dataclass(frozen=True)
class PostgresIdentity:
    """What the connected role is, and what it is able to do."""

    user: str
    session_user: str
    database: str
    search_path: str
    default_transaction_read_only: bool
    is_superuser: bool
    may_create_database: bool
    may_create_role: bool
    bypasses_row_level_security: bool
    may_replicate: bool
    predefined_roles: tuple[str, ...] = ()
    #: Tables and views the login may INSERT, UPDATE, DELETE or TRUNCATE,
    #: schema-qualified. Each is a concern: the read-only transaction refuses
    #: the write, and the grant is what holds if anything gets past it.
    writable_tables: tuple[str, ...] = ()
    #: How many tables and views the login may read, and in which schemas.
    #: ``None`` when not read. Reported, not judged: whether a table should be
    #: readable is not something grants can say.
    readable_tables: int | None = None
    readable_schemas: tuple[str, ...] = ()
    #: Set when the identity could not be read. Held separately from
    #: :attr:`concerns` so that a check which did not complete is distinguishable
    #: from one that completed and found nothing.
    undetermined_reason: str | None = None

    @property
    def concerns(self) -> tuple[str, ...]:
        """One entry per privilege the application cannot withdraw.

        Derived from the flags rather than stored alongside them, so the two
        cannot disagree.
        """
        found: list[str] = []
        if self.is_superuser:
            found.append("connects as a superuser, which bypasses every privilege check")
        if self.bypasses_row_level_security:
            found.append("holds BYPASSRLS, so row-level security does not apply")
        if self.may_create_database:
            found.append("holds CREATEDB")
        if self.may_create_role:
            found.append("holds CREATEROLE, and can grant itself further privileges")
        if self.may_replicate:
            found.append("holds REPLICATION, which permits streaming the cluster")

        dangerous = sorted(set(self.predefined_roles) & DANGEROUS_PREDEFINED_ROLES)
        if dangerous:
            found.append(f"is a member of {', '.join(dangerous)}")

        if self.writable_tables:
            count = len(self.writable_tables)
            named = ", ".join(self.writable_tables[:_NAMED_TABLES])
            more = count - _NAMED_TABLES
            noun = "table" if count == 1 else "tables"
            found.append(
                f"may write to {count} {noun}: {named}" + (f", and {more} more" if more > 0 else "")
            )
        return tuple(found)

    @property
    def is_least_privilege(self) -> bool:
        """True when the identity was read and no concerns were found.

        The name is kept for callers; what it means is narrower than it says.
        No administrative right, dangerous predefined role or table write
        was found. Whether everything the login may read should be readable
        is not something grants can answer, so :meth:`summary` never claims it.
        """
        return self.undetermined_reason is None and not self.concerns

    def _reads(self) -> str:
        if self.readable_tables is None:
            return ""
        if self.readable_tables == 0:
            return "; reads no tables"
        tables = "table" if self.readable_tables == 1 else "tables"
        schemas = "schema" if len(self.readable_schemas) == 1 else "schemas"
        where = f" in {schemas} {', '.join(self.readable_schemas)}" if self.readable_schemas else ""
        return f"; reads {self.readable_tables} {tables}{where}"

    def summary(self) -> str:
        """A single line for logs and the health report.

        A clean result says what the login holds, not that it is the least it
        needs: "least privilege" was printed for a login that could read every
        salary row in a schema, because nothing here looked at table grants.
        """
        if self.undetermined_reason is not None:
            return f"identity could not be determined: {self.undetermined_reason}"
        if not self.concerns:
            return (
                f"{self.user}@{self.database}: no administrative or write privileges"
                + self._reads()
            )
        return f"{self.user}@{self.database}: {'; '.join(self.concerns)}{self._reads()}"

    @classmethod
    def undetermined(cls, reason: str) -> PostgresIdentity:
        return cls(
            user="",
            session_user="",
            database="",
            search_path="",
            default_transaction_read_only=False,
            is_superuser=False,
            may_create_database=False,
            may_create_role=False,
            bypasses_row_level_security=False,
            may_replicate=False,
            undetermined_reason=reason,
        )


def inspect_identity(connection: Any) -> PostgresIdentity:
    """Read the connected role's privileges.

    *connection* is any object exposing the DBAPI ``cursor()`` method. This
    function does not raise. A check that cannot complete returns an
    undetermined report, so that a failure of the check does not prevent the
    connector from opening.
    """
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(IDENTITY_SQL)
            row = cursor.fetchone()
        finally:
            cursor.close()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Could not read the PostgreSQL identity: %s", exc)
        return PostgresIdentity.undetermined(str(exc))

    if row is None:
        # current_user has no pg_roles row. Not expected on a conforming
        # server; reported rather than treated as an absence of findings.
        return PostgresIdentity.undetermined("no pg_roles row for current_user")

    (
        user,
        session_user,
        database,
        search_path,
        default_read_only,
        is_superuser,
        may_create_database,
        may_create_role,
        bypasses_rls,
        may_replicate,
        predefined,
        writable,
        readable,
        readable_schemas,
    ) = row

    return PostgresIdentity(
        user=str(user),
        session_user=str(session_user),
        database=str(database),
        search_path=str(search_path),
        default_transaction_read_only=str(default_read_only).lower() == "on",
        is_superuser=bool(is_superuser),
        may_create_database=bool(may_create_database),
        may_create_role=bool(may_create_role),
        bypasses_row_level_security=bool(bypasses_rls),
        may_replicate=bool(may_replicate),
        predefined_roles=tuple(predefined or ()),
        writable_tables=tuple(writable or ()),
        readable_tables=int(readable) if readable is not None else None,
        readable_schemas=tuple(readable_schemas or ()),
    )
