"""
The PostgreSQL identity check, against a real server (W3-1).

``tests/test_postgres_identity.py`` covers the classification rules without a
database. This file covers the part that cannot be asserted from a unit test:
that the query runs on a real PostgreSQL, that it distinguishes the
least-privilege role the hardening guide describes from the privileged one the
container ships with, and that the connector populates the report by itself.

The lab's ``sec_reader`` is built from ``docs/database-hardening.md``, so a
change to the guide that leaves the role over-privileged is visible here.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from nlqueries.connectors.postgres import PostgresConnector
from nlqueries.connectors.postgres_identity import inspect_identity

from tests.conftest import granted

pytestmark = pytest.mark.security


def _raw_connection(credentials: dict[str, Any]) -> Any:
    """A DBAPI connection, which is what the pool's connect event hands over."""
    url = sa.URL.create(
        "postgresql+psycopg2",
        username=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        database=credentials["database"],
    )
    return sa.create_engine(url).raw_connection()


def test_the_least_privilege_role_reports_least_privilege(
    least_privilege_credentials: dict[str, Any],
) -> None:
    """The control for this whole check.

    If the role the guide tells operators to create were still reported as
    over-privileged, the check would be noise and would be turned off.
    """
    connection = _raw_connection(least_privilege_credentials)
    try:
        identity = inspect_identity(connection)
    finally:
        connection.close()

    assert identity.is_least_privilege, identity.summary()
    assert identity.concerns == ()
    assert not identity.is_superuser


def test_the_privileged_role_is_reported_as_privileged(
    privileged_credentials: dict[str, Any],
) -> None:
    """The container's own login, which is what an unconfigured deployment
    would use."""
    connection = _raw_connection(privileged_credentials)
    try:
        identity = inspect_identity(connection)
    finally:
        connection.close()

    assert not identity.is_least_privilege
    assert identity.concerns, "a superuser login produced no findings"
    assert any("superuser" in concern for concern in identity.concerns)


def test_the_search_path_the_guide_sets_is_reported(
    least_privilege_credentials: dict[str, Any],
) -> None:
    """The lab sets `search_path = pg_catalog, lab` on the role, following the
    guide. Reporting it is how an operator sees a path they did not intend."""
    connection = _raw_connection(least_privilege_credentials)
    try:
        identity = inspect_identity(connection)
    finally:
        connection.close()

    assert "lab" in identity.search_path
    assert "pg_catalog" in identity.search_path


def test_the_connector_reports_its_identity_without_being_asked(
    least_privilege_credentials: dict[str, Any],
) -> None:
    """The report is produced by the pool's connect event, so an operator does
    not have to know the check exists to benefit from it."""
    connector = granted(PostgresConnector())
    connector.connect({**least_privilege_credentials, "ssl_mode": "disable"})
    try:
        # None before the first connection: the pool opens them lazily, and
        # "nothing reported yet" is not the same as "nothing to report".
        assert connector.identity is None

        result = connector.execute_query("SELECT 1")

        assert result.error is None
        assert connector.identity is not None
        assert connector.identity.is_least_privilege, connector.identity.summary()
    finally:
        connector.close()


def test_a_privileged_connector_reports_its_concerns(
    privileged_credentials: dict[str, Any],
) -> None:
    connector = granted(PostgresConnector())
    connector.connect({**privileged_credentials, "ssl_mode": "disable"})
    try:
        connector.execute_query("SELECT 1")

        assert connector.identity is not None
        assert not connector.identity.is_least_privilege
    finally:
        connector.close()
