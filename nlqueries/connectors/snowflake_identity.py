# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.snowflake_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What the role a Snowflake connection uses can do, through every role it
inherits. See :mod:`nlqueries.connectors.login_identity`.

A Snowflake session acts with its current role and every role granted to it,
so the grants of the whole hierarchy are read: ``SHOW GRANTS TO ROLE`` for the
current role, then for each role it has ``USAGE`` on, and so on. Secondary
roles are not followed.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from nlqueries.connectors.login_identity import LoginIdentity

logger = logging.getLogger(__name__)

ENGINE = "snowflake"

#: System roles that administer the account.
ADMIN_ROLES = frozenset({"ACCOUNTADMIN", "ORGADMIN", "SECURITYADMIN", "SYSADMIN", "USERADMIN"})

#: Privileges that read or use and do nothing else. ``OPERATE`` resumes and
#: suspends a warehouse; ``READ`` reads a stage.
READ_PRIVILEGES = frozenset(
    {"USAGE", "SELECT", "REFERENCES", "MONITOR", "OPERATE", "READ", "IMPORTED PRIVILEGES"}
)

#: The most roles read before the walk stops and says so.
MAX_ROLES = 50

#: One row of ``SHOW GRANTS TO ROLE``: (privilege, granted_on, name).
Grant = tuple[str, str, str]


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def walk_roles(
    current_role: str, grants_of: Callable[[str], Iterable[Grant]]
) -> tuple[list[str], list[tuple[str, Grant]], bool]:
    """Every role reachable from *current_role*, and every grant they hold.

    *grants_of* returns a role's grants. Returns the roles in the order read,
    each grant with the role that holds it, and whether :data:`MAX_ROLES` cut
    the walk short.
    """
    seen: list[str] = []
    grants: list[tuple[str, Grant]] = []
    queue = [current_role]
    while queue:
        role = queue.pop(0)
        if role in seen:
            continue
        if len(seen) == MAX_ROLES:
            return seen, grants, True
        seen.append(role)
        for grant in grants_of(role):
            grants.append((role, grant))
            privilege, granted_on, name = grant
            if granted_on.upper() == "ROLE" and privilege.upper() == "USAGE":
                queue.append(name)
    return seen, grants, False


def unexpanded_roles(grants: Iterable[tuple[str, Grant]]) -> tuple[str, ...]:
    """Roles of another kind in the hierarchy, whose own grants the walk does not read.

    A database role granted to an account role is listed by ``SHOW GRANTS TO
    ROLE`` as ``USAGE`` on it, and ``USAGE`` alone reads as harmless; what the
    database role holds -- ``INSERT`` on a schema's tables, say -- takes ``SHOW
    GRANTS TO DATABASE ROLE``, which this does not run. Such a grant is reported
    as unread, so the line is a caveat rather than a pass. Matched on any kind of
    role but the account role the walk follows, rather than one spelling of it.
    """
    found: set[str] = set()
    for _role, (privilege, granted_on, name) in grants:
        kind = granted_on.upper()
        if privilege.upper() == "USAGE" and kind != "ROLE" and kind.endswith("ROLE"):
            found.add(f"the grants of {kind.lower().replace('_', ' ')} {name}")
    return tuple(sorted(found))


def concerns_from(
    roles: Iterable[str], grants: Iterable[tuple[str, Grant]], truncated: bool
) -> tuple[str, ...]:
    """Administrative roles in reach, and privileges beyond reading by kind."""
    found: list[str] = []
    admin = sorted({r.upper() for r in roles} & ADMIN_ROLES)
    if admin:
        found.append(f"can use {', '.join(admin)}")
    beyond = Counter(
        (privilege.upper(), granted_on.upper())
        for _role, (privilege, granted_on, _name) in grants
        if privilege.upper() not in READ_PRIVILEGES
    )
    for (privilege, granted_on), count in sorted(beyond.items()):
        found.append(f"holds {privilege} on {count} {granted_on.lower()}(s)")
    if truncated:
        found.append(f"inherits more than {MAX_ROLES} roles; the rest were not read")
    return tuple(found)


def inspect_identity(connection: Any) -> LoginIdentity:
    """Read the session's role hierarchy and its grants. Never raises.

    *connection* is a Snowflake DBAPI connection.
    """
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE()")
            user, role, database = cursor.fetchone() or ("", "", "")
            if not role:
                return LoginIdentity.undetermined(ENGINE, "the session has no current role")

            def grants_of(name: str) -> list[Grant]:
                cursor.execute(f"SHOW GRANTS TO ROLE {_quoted(name)}")
                columns = [d[0].lower() for d in cursor.description]
                at = {c: columns.index(c) for c in ("privilege", "granted_on", "name")}
                return [
                    (str(row[at["privilege"]]), str(row[at["granted_on"]]), str(row[at["name"]]))
                    for row in cursor.fetchall()
                ]

            roles, grants, truncated = walk_roles(str(role), grants_of)
        finally:
            cursor.close()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Could not read the Snowflake role's grants: %s", exc)
        return LoginIdentity.undetermined(ENGINE, str(exc))
    return LoginIdentity(
        engine=ENGINE,
        user=f"{user} (role {role})",
        database=str(database or ""),
        concerns=concerns_from(roles, grants, truncated),
        unread=unexpanded_roles(grants),
    )
