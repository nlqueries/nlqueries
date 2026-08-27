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
        return tuple(found)

    @property
    def is_least_privilege(self) -> bool:
        """True when the identity was read and no concerns were found."""
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
