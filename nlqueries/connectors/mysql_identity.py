# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.mysql_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The privileges of the MySQL or MariaDB login the generic SQLAlchemy connector
uses, read from ``SHOW GRANTS``. See :mod:`nlqueries.connectors.login_identity`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from nlqueries.connectors.login_identity import LoginIdentity

logger = logging.getLogger(__name__)

ENGINE = "mysql"

#: The privileges a read-only login needs. ``USAGE`` is "no privilege", which
#: every account shows on ``*.*``; ``SHOW VIEW`` reads a view's definition.
READ_PRIVILEGES = frozenset({"SELECT", "SHOW VIEW", "USAGE"})

_GRANT_ON = re.compile(r"^GRANT\s+(?P<privileges>.+?)\s+ON\s+(?P<scope>.+?)\s+TO\s", re.IGNORECASE)
_GRANT_ROLES = re.compile(r"^GRANT\s+(?P<roles>.+?)\s+TO\s", re.IGNORECASE)
#: A column list, as in ``GRANT SELECT (id, name) ON ...``.
_COLUMN_LIST = re.compile(r"\([^)]*\)")


def concerns_from_grants(grants: Iterable[str]) -> tuple[str, ...]:
    """One concern per ``SHOW GRANTS`` line that gives more than reading.

    A privilege outside :data:`READ_PRIVILEGES` is reported with the scope it
    applies to, and so is ``WITH GRANT OPTION``. A line granting roles is
    reported too: a role's privileges are not expanded here, so the check
    cannot say the login stops at reading.
    """
    found: list[str] = []
    for line in grants:
        text = " ".join(str(line).split())
        on = _GRANT_ON.match(text)
        if on is None:
            roles = _GRANT_ROLES.match(text)
            if roles is not None:
                found.append(
                    f"is granted {roles['roles']}, whose privileges this check does not read"
                )
            continue
        privileges = [
            p.strip().upper()
            for p in _COLUMN_LIST.sub("", on["privileges"]).split(",")
            if p.strip()
        ]
        beyond = [p for p in privileges if p not in READ_PRIVILEGES]
        if beyond:
            found.append(f"holds {', '.join(beyond)} on {on['scope']}")
        if re.search(r"\bWITH\s+GRANT\s+OPTION\b", text, re.IGNORECASE):
            found.append(f"can grant its privileges on {on['scope']} to others")
    return tuple(found)


def inspect_identity(connection: Any) -> LoginIdentity:
    """Read the connected login's grants. Never raises.

    *connection* is any object with a DBAPI ``cursor()``. A check that cannot
    complete returns an undetermined identity, so it never stops a connection
    from opening.
    """
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CURRENT_USER(), DATABASE()")
            user, database = cursor.fetchone() or ("", "")
            cursor.execute("SHOW GRANTS")
            grants = [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Could not read the MySQL login's grants: %s", exc)
        return LoginIdentity.undetermined(ENGINE, str(exc))
    if not grants:
        return LoginIdentity.undetermined(ENGINE, "SHOW GRANTS returned nothing")
    return LoginIdentity(
        engine=ENGINE,
        user=str(user),
        database=str(database or ""),
        concerns=concerns_from_grants(grants),
    )
