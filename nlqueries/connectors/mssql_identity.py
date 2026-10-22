# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.mssql_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The roles and effective permissions of the SQL Server login the ``mssql``
connector uses. See :mod:`nlqueries.connectors.login_identity`.

Role membership names the obvious cases (``sysadmin``, ``db_owner``,
``db_datawriter``); the effective permissions from ``fn_my_permissions``
catch the rest, including what a custom role or a direct grant gives, which a
role list alone cannot show.

What this does not see: grants on a schema or an object. ``fn_my_permissions``
is read for the database and the server securables, and a permission granted
``ON SCHEMA::dbo`` or on a table is held on that securable, not on the
database, so a login granted ``INSERT, UPDATE, DELETE ON SCHEMA::dbo`` and no
role reads as least privilege here. On SQL Server the line therefore means
nothing beyond reading at database and server scope; schema and object grants
are checked in ``sys.database_permissions``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from nlqueries.connectors.login_identity import LoginIdentity

logger = logging.getLogger(__name__)

ENGINE = "mssql"

#: Database roles a read-only login may belong to. ``public`` is every user's.
READ_DATABASE_ROLES = frozenset({"public", "db_datareader"})

#: Database permissions that read or describe and do nothing else.
#:
#: ``fn_my_permissions`` reports effective permissions, which includes what
#: ``public`` holds and what a listed permission implies, so both are here too.
#: SQL Server 2016 and later grant the two column-key ``VIEW`` permissions to
#: ``public`` in every database; SQL Server 2022 splits ``VIEW DEFINITION`` and
#: ``VIEW DATABASE STATE`` into the narrower ``VIEW`` permissions below them.
#: Only those whose place under a listed permission is certain are here: one
#: wrongly left off shows as a warning that can be explained, one wrongly added
#: hides a right the login holds.
READ_DATABASE_PERMISSIONS = frozenset(
    {
        "CONNECT",
        "SELECT",
        "VIEW DEFINITION",
        "VIEW DATABASE STATE",
        "VIEW DATABASE PERFORMANCE STATE",
        "SHOWPLAN",
        "REFERENCES",
        # public's, since SQL Server 2016
        "VIEW ANY COLUMN ENCRYPTION KEY DEFINITION",
        "VIEW ANY COLUMN MASTER KEY DEFINITION",
        # implied by VIEW DEFINITION and VIEW DATABASE STATE, SQL Server 2022
        "VIEW SECURITY DEFINITION",
        "VIEW PERFORMANCE DEFINITION",
        "VIEW CRYPTOGRAPHICALLY SECURED DEFINITION",
        "VIEW DATABASE SECURITY STATE",
    }
)

#: Server permissions every login, or a monitoring one, holds, and the
#: narrower ``VIEW`` permissions SQL Server 2022 splits those into, on the same
#: terms as the database's. ``VIEW ANY ERROR LOG`` and the two ``... SECURITY
#: AUDIT`` permissions are left off for that reason.
READ_SERVER_PERMISSIONS = frozenset(
    {
        "CONNECT SQL",
        "VIEW ANY DATABASE",
        "VIEW ANY DEFINITION",
        "VIEW SERVER STATE",
        "VIEW SERVER PERFORMANCE STATE",
        "CONNECT ANY DATABASE",
        # implied by VIEW ANY DEFINITION and VIEW SERVER STATE, SQL Server 2022
        "VIEW ANY SECURITY DEFINITION",
        "VIEW ANY PERFORMANCE DEFINITION",
        "VIEW ANY CRYPTOGRAPHICALLY SECURED DEFINITION",
        "VIEW SERVER SECURITY STATE",
    }
)

#: ``USER_NAME()`` when there is no server login to name: a contained database
#: user, the usual kind on Azure SQL, has none, and ``SUSER_SNAME()`` is NULL.
LOGIN_SQL = "SELECT COALESCE(SUSER_SNAME(), USER_NAME()), DB_NAME()"
SERVER_ROLES_SQL = (
    "SELECT name FROM sys.server_principals WHERE type = 'R' AND IS_SRVROLEMEMBER(name) = 1"
)
DATABASE_ROLES_SQL = (
    "SELECT name FROM sys.database_principals WHERE type = 'R' AND IS_ROLEMEMBER(name) = 1"
)
DATABASE_PERMISSIONS_SQL = "SELECT permission_name FROM fn_my_permissions(NULL, 'DATABASE')"
SERVER_PERMISSIONS_SQL = "SELECT permission_name FROM fn_my_permissions(NULL, 'SERVER')"


#: How many permissions one concern names before it counts the rest. A
#: ``db_owner`` login's effective permissions run past a hundred.
NAMED_PERMISSIONS = 8


def _listed(permissions: Iterable[str], control: str) -> str:
    """*permissions* for a concern, short enough to read.

    ``CONTROL`` (``CONTROL SERVER`` for the server) implies every other
    permission in its scope, so when it is there it is the one named.
    """
    names = sorted(set(permissions))
    if control in names:
        names.remove(control)
        return f"{control} and {len(names)} more" if names else control
    if len(names) <= NAMED_PERMISSIONS:
        return ", ".join(names)
    return f"{', '.join(names[:NAMED_PERMISSIONS])} and {len(names) - NAMED_PERMISSIONS} more"


def concerns_from(
    server_roles: Iterable[str],
    database_roles: Iterable[str],
    database_permissions: Iterable[str],
    server_permissions: Iterable[str],
) -> tuple[str, ...]:
    """The roles and permissions in the login's reach beyond reading."""
    found: list[str] = []
    roles = sorted({r for r in server_roles if r.lower() != "public"})
    if roles:
        found.append(f"is a member of server role {', '.join(roles)}")
    roles = sorted({r for r in database_roles if r.lower() not in READ_DATABASE_ROLES})
    if roles:
        found.append(f"is a member of database role {', '.join(roles)}")
    beyond = {p.upper() for p in database_permissions} - READ_DATABASE_PERMISSIONS
    if beyond:
        found.append(f"holds {_listed(beyond, 'CONTROL')} on the database")
    beyond = {p.upper() for p in server_permissions} - READ_SERVER_PERMISSIONS
    if beyond:
        found.append(f"holds {_listed(beyond, 'CONTROL SERVER')} on the server")
    return tuple(found)


def _column(connection: Any, sql: str) -> list[str]:
    """The first column of *sql*'s rows, on a cursor of its own."""
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        cursor.close()


def inspect_identity(connection: Any) -> LoginIdentity:
    """Read the connected login's roles and permissions. Never raises.

    *connection* is any object with a DBAPI ``cursor()``.

    The database scope has to be read for the check to mean anything. The
    server scope is read separately, one statement at a time, and may fail on
    its own: Azure SQL does not reliably offer it from a user database, and a
    hardened login there should not fail ``health`` because of that. What could
    not be read is reported on the identity as unread. A ``sysadmin`` login is
    caught without the server read, since it is the database owner and holds
    ``CONTROL`` on the database; other server roles show only when it succeeds.
    """
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(LOGIN_SQL)
            rows = cursor.fetchall()
        finally:
            cursor.close()
        user, database = rows[0] if rows else ("", "")
        database_roles = _column(connection, DATABASE_ROLES_SQL)
        database_permissions = _column(connection, DATABASE_PERMISSIONS_SQL)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Could not read the SQL Server login's permissions: %s", exc)
        return LoginIdentity.undetermined(ENGINE, str(exc))
    if not database_permissions:
        # Every user holds at least CONNECT; nothing back means the read failed.
        return LoginIdentity.undetermined(ENGINE, "fn_my_permissions returned nothing")

    server: dict[str, list[str]] = {}
    unread: list[str] = []
    for what, sql in (
        ("server roles", SERVER_ROLES_SQL),
        ("server permissions", SERVER_PERMISSIONS_SQL),
    ):
        try:
            server[what] = _column(connection, sql)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.info("Could not read the SQL Server login's %s: %s", what, exc)
            server[what] = []
            unread.append(f"{what} ({exc})")
    return LoginIdentity(
        engine=ENGINE,
        user=str(user or ""),
        database=str(database or ""),
        concerns=concerns_from(
            server["server roles"],
            database_roles,
            database_permissions,
            server["server permissions"],
        ),
        unread=tuple(unread),
        # Said with a clean result: schema and object grants are not read.
        scope="database and server scope",
    )
