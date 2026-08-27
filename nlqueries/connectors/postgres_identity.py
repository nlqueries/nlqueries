# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.postgres_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inspection of the PostgreSQL login a connector is using.

The read-only transaction opened for every query constrains what a statement may
do. It does not constrain who the statement runs as, and several of the controls
that matter are privileges rather than transaction state. ``docs/database-
hardening.md`` records the measured result: ``pg_read_file()`` is refused by
privilege, not by the read-only transaction, so a role holding
``pg_read_server_files`` retains access that no application-level control can
withdraw.

This module reads the connected role's privileges and reports what it finds. It
does not refuse a connection: the report is surfaced through ``nlqueries
health``, so that a deployment pointed at an over-privileged role is visible
rather than assumed correct. Enforcement is a separate decision, and refusing
here would take it away from the operator at the point where they can least
afford a surprise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Predefined roles whose membership grants access the application cannot
#: withdraw. ``pg_database_owner`` is deliberately absent: every owner of a
#: database holds it, and it confers nothing beyond that ownership.
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
#: ``pg_roles`` rather than named individually, so a server version that lacks
#: one -- or adds one -- does not cause an error or a silent gap.
IDENTITY_SQL = """
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
    )                                                     AS predefined_roles
FROM pg_roles r
WHERE r.rolname = current_user
"""


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
    #: Set when the identity could not be read. Distinct from an identity that
    #: was read and found acceptable, and reported as its own state so that a
    #: failed check is not mistaken for a passed one.
    undetermined_reason: str | None = None

    @property
    def concerns(self) -> tuple[str, ...]:
        """One entry per privilege the application cannot withdraw.

        Derived rather than stored, so that an identity cannot exist in a state
        where the flags say one thing and the findings say another.
        """
        found: list[str] = []
        if self.is_superuser:
            found.append("connects as a superuser, which bypasses every privilege check")
        if self.bypasses_row_level_security:
            found.append("holds BYPASSRLS, so row-level security does not apply")
        if self.may_create_database:
            found.append("holds CREATEDB")
        if self.may_create_role:
            found.append("holds CREATEROLE, and can therefore grant itself more")
        if self.may_replicate:
            found.append("holds REPLICATION, which can stream the whole cluster")

        dangerous = sorted(set(self.predefined_roles) & DANGEROUS_PREDEFINED_ROLES)
        if dangerous:
            found.append(f"is a member of {', '.join(dangerous)}")
        return tuple(found)

    @property
    def is_least_privilege(self) -> bool:
        """True only when the identity was read and nothing of concern found."""
        return self.undetermined_reason is None and not self.concerns

    def summary(self) -> str:
        """A single line for logs and the health report."""
        if self.undetermined_reason is not None:
            return f"identity could not be determined: {self.undetermined_reason}"
        if not self.concerns:
            return f"{self.user}@{self.database}: least privilege"
        return f"{self.user}@{self.database}: {'; '.join(self.concerns)}"

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

    *connection* is anything exposing DBAPI ``cursor()``. Never raises: a check
    that cannot run is reported as undetermined, because a connector that
    refused to open because its self-check failed would convert a diagnostic
    into an outage.
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
        # current_user has no pg_roles row. Not reachable on a normal server,
        # and reported rather than assumed benign.
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
    )
