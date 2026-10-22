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
READ_DATABASE_PERMISSIONS = frozenset(
    {
        "CONNECT",
        "SELECT",
        "VIEW DEFINITION",
        "VIEW DATABASE STATE",
        "VIEW DATABASE PERFORMANCE STATE",
        "SHOWPLAN",
        "REFERENCES",
    }
)

#: Server permissions every login, or a monitoring one, holds.
READ_SERVER_PERMISSIONS = frozenset(
    {
        "CONNECT SQL",
        "VIEW ANY DATABASE",
        "VIEW ANY DEFINITION",
        "VIEW SERVER STATE",
        "VIEW SERVER PERFORMANCE STATE",
        "CONNECT ANY DATABASE",
    }
)

LOGIN_SQL = "SELECT SUSER_SNAME(), DB_NAME()"
SERVER_ROLES_SQL = (
    "SELECT name FROM sys.server_principals WHERE type = 'R' AND IS_SRVROLEMEMBER(name) = 1"
)
DATABASE_ROLES_SQL = (
    "SELECT name FROM sys.database_principals WHERE type = 'R' AND IS_ROLEMEMBER(name) = 1"
)
DATABASE_PERMISSIONS_SQL = "SELECT permission_name FROM fn_my_permissions(NULL, 'DATABASE')"
SERVER_PERMISSIONS_SQL = "SELECT permission_name FROM fn_my_permissions(NULL, 'SERVER')"


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
    beyond = sorted({p.upper() for p in database_permissions} - READ_DATABASE_PERMISSIONS)
    if beyond:
        found.append(f"holds {', '.join(beyond)} on the database")
    beyond = sorted({p.upper() for p in server_permissions} - READ_SERVER_PERMISSIONS)
    if beyond:
        found.append(f"holds {', '.join(beyond)} on the server")
    return tuple(found)


def inspect_identity(connection: Any) -> LoginIdentity:
    """Read the connected login's roles and permissions. Never raises.

    *connection* is any object with a DBAPI ``cursor()``.
    """
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(LOGIN_SQL)
            user, database = cursor.fetchone() or ("", "")
            columns: list[list[str]] = []
            for sql in (
                SERVER_ROLES_SQL,
                DATABASE_ROLES_SQL,
                DATABASE_PERMISSIONS_SQL,
                SERVER_PERMISSIONS_SQL,
            ):
                cursor.execute(sql)
                columns.append([str(row[0]) for row in cursor.fetchall()])
        finally:
            cursor.close()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Could not read the SQL Server login's permissions: %s", exc)
        return LoginIdentity.undetermined(ENGINE, str(exc))
    server_roles, database_roles, database_permissions, server_permissions = columns
    if not database_permissions:
        # Every user holds at least CONNECT; nothing back means the read failed.
        return LoginIdentity.undetermined(ENGINE, "fn_my_permissions returned nothing")
    return LoginIdentity(
        engine=ENGINE,
        user=str(user),
        database=str(database or ""),
        concerns=concerns_from(
            server_roles, database_roles, database_permissions, server_permissions
        ),
    )
