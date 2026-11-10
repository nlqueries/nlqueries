"""
The PostgreSQL identity check, against a real server (W3-1).

``tests/test_postgres_identity.py`` covers the classification rules without a
database. This file covers what a unit test cannot assert: that the query
executes on a real PostgreSQL, that it distinguishes the least-privilege role
described in the hardening guide from the privileged role the container
provides, and that the connector populates the report without an explicit call.

The lab's ``sec_reader`` role is built from ``docs/database-hardening.md``, so a
change to the guide that leaves the role over-privileged is detected here.
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
    """The control for the check: the role the guide instructs operators to
    create must be reported as least privilege."""
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
    """The container's default login, as used by an unconfigured
    deployment."""
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
    guide. The report includes it so an unintended path is visible."""
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
    """The report is produced by the pool's connect event and requires no
    explicit call."""
    connector = granted(PostgresConnector())
    connector.connect({**least_privilege_credentials, "ssl_mode": "disable"})
    try:
        # None before the first connection: the pool opens connections
        # lazily, so no check has run at this point.
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


# ---------------------------------------------------------------------------
# Table grants (CI-8): what health may say about what a login reads and writes
# ---------------------------------------------------------------------------


@pytest.fixture()
def login_with(privileged_credentials: dict[str, Any]) -> Any:
    """A throwaway login with the grants a test names, dropped afterwards."""
    engine = sa.create_engine(
        sa.URL.create(
            "postgresql+psycopg2",
            username=privileged_credentials["user"],
            password=privileged_credentials["password"],
            host=privileged_credentials["host"],
            port=privileged_credentials["port"],
            database=privileged_credentials["database"],
        )
    )
    made: list[str] = []

    def _make(name: str, *grants: str) -> dict[str, Any]:
        with engine.begin() as conn:
            conn.execute(sa.text(f"CREATE ROLE {name} LOGIN PASSWORD 'lab-only'"))
            for grant in grants:
                conn.execute(sa.text(grant.format(role=name)))
        made.append(name)
        return {**privileged_credentials, "user": name, "password": "lab-only"}

    yield _make
    with engine.begin() as conn:
        for name in made:
            conn.execute(sa.text(f"DROP OWNED BY {name}"))
            conn.execute(sa.text(f"DROP ROLE {name}"))
    engine.dispose()


def _identity_of(credentials: dict[str, Any]) -> Any:
    connection = _raw_connection(credentials)
    try:
        return inspect_identity(connection)
    finally:
        connection.close()


def test_the_guide_s_login_reports_what_it_reads(
    least_privilege_credentials: dict[str, Any],
) -> None:
    identity = _identity_of(least_privilege_credentials)

    assert identity.concerns == ()
    assert identity.summary().endswith(
        "no administrative or write privileges; reads 1 table in schema lab"
    )


def test_schema_wide_select_is_counted_not_called_least_privilege(login_with: Any) -> None:
    """The report that started this: a login with SELECT on a whole schema,
    every salary row included, was printed as "least privilege"."""
    credentials = login_with(
        "ci8_wide",
        "GRANT USAGE ON SCHEMA lab TO {role}",
        "GRANT SELECT ON ALL TABLES IN SCHEMA lab TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.concerns == ()
    assert "least privilege" not in identity.summary()
    assert identity.readable_tables == 2
    assert identity.readable_schemas == ("lab",)


def test_a_write_grant_is_a_concern(login_with: Any) -> None:
    credentials = login_with(
        "ci8_writer",
        "GRANT USAGE ON SCHEMA lab TO {role}",
        "GRANT SELECT ON lab.orders TO {role}",
        "GRANT INSERT ON lab.marker TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.writable_tables == ("lab.marker",)
    assert "may write to 1 table: lab.marker" in identity.concerns


def test_a_grant_in_a_schema_the_login_cannot_use_is_not_counted(login_with: Any) -> None:
    """Without USAGE on the schema the table cannot be reached at all."""
    credentials = login_with("ci8_nousage", "GRANT SELECT, INSERT ON lab.marker TO {role}")

    identity = _identity_of(credentials)

    assert identity.writable_tables == ()
    assert identity.readable_tables == 0


def test_a_column_grant_counts_as_reading_the_table(login_with: Any) -> None:
    """`GRANT SELECT (id)` reads part of a table, which is still a read."""
    credentials = login_with(
        "ci8_columns",
        "GRANT USAGE ON SCHEMA lab TO {role}",
        "GRANT SELECT (id) ON lab.orders TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.readable_tables == 1
    assert identity.readable_schemas == ("lab",)


def test_a_column_write_grant_is_a_concern(login_with: Any) -> None:
    """`UPDATE (note)` writes as surely as `UPDATE` does (from #399's review)."""
    credentials = login_with(
        "ci8_colwriter",
        "GRANT USAGE ON SCHEMA lab TO {role}",
        "GRANT SELECT ON lab.orders TO {role}",
        "GRANT UPDATE (note) ON lab.marker TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.writable_tables == ("lab.marker",)
    assert "no administrative or write privileges" not in identity.summary()


def test_partitions_are_counted_with_their_table(login_with: Any, privileged_credentials) -> None:
    """A partitioned table with three partitions is one table to read."""
    engine = sa.create_engine(
        sa.URL.create(
            "postgresql+psycopg2",
            username=privileged_credentials["user"],
            password=privileged_credentials["password"],
            host=privileged_credentials["host"],
            port=privileged_credentials["port"],
            database=privileged_credentials["database"],
        )
    )
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE SCHEMA ci8_parts"))
        conn.execute(sa.text("CREATE TABLE ci8_parts.events (id int) PARTITION BY RANGE (id)"))
        for i in range(3):
            conn.execute(
                sa.text(
                    f"CREATE TABLE ci8_parts.events_{i} PARTITION OF ci8_parts.events "
                    f"FOR VALUES FROM ({i * 10}) TO ({i * 10 + 10})"
                )
            )
    try:
        credentials = login_with(
            "ci8_partreader",
            "GRANT USAGE ON SCHEMA ci8_parts TO {role}",
            "GRANT SELECT ON ALL TABLES IN SCHEMA ci8_parts TO {role}",
        )
        identity = _identity_of(credentials)
        assert identity.readable_tables == 1
        assert identity.readable_schemas == ("ci8_parts",)
    finally:
        with engine.begin() as conn:
            conn.execute(sa.text("DROP SCHEMA ci8_parts CASCADE"))
        engine.dispose()


@pytest.fixture()
def partitioned(privileged_credentials: dict[str, Any]) -> Any:
    """ci8_parts.events with three partitions, dropped afterwards."""
    engine = sa.create_engine(
        sa.URL.create(
            "postgresql+psycopg2",
            username=privileged_credentials["user"],
            password=privileged_credentials["password"],
            host=privileged_credentials["host"],
            port=privileged_credentials["port"],
            database=privileged_credentials["database"],
        )
    )
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE SCHEMA ci8_parts2"))
        conn.execute(sa.text("CREATE TABLE ci8_parts2.events (id int) PARTITION BY RANGE (id)"))
        for i in range(3):
            conn.execute(
                sa.text(
                    f"CREATE TABLE ci8_parts2.events_{i} PARTITION OF ci8_parts2.events "
                    f"FOR VALUES FROM ({i * 10}) TO ({i * 10 + 10})"
                )
            )
    yield
    with engine.begin() as conn:
        conn.execute(sa.text("DROP SCHEMA ci8_parts2 CASCADE"))
    engine.dispose()


def test_a_write_held_on_one_partition_alone_is_a_concern(
    login_with: Any, partitioned: Any
) -> None:
    """Grants are not inherited, and a statement against a partition checks the
    partition's own: INSERT on one partition is a write, though the parent
    grants nothing (from #399's review)."""
    credentials = login_with(
        "ci8_partwriter",
        "GRANT USAGE ON SCHEMA ci8_parts2 TO {role}",
        "GRANT INSERT ON ci8_parts2.events_0 TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.writable_tables == ("ci8_parts2.events_0",)


def test_a_read_held_on_one_partition_alone_is_counted(login_with: Any, partitioned: Any) -> None:
    credentials = login_with(
        "ci8_partonly",
        "GRANT USAGE ON SCHEMA ci8_parts2 TO {role}",
        "GRANT SELECT ON ci8_parts2.events_1 TO {role}",
    )

    identity = _identity_of(credentials)

    assert identity.readable_tables == 1
    assert identity.readable_schemas == ("ci8_parts2",)


def test_a_partition_is_not_folded_into_a_parent_the_login_cannot_reach(
    login_with: Any, privileged_credentials: dict[str, Any]
) -> None:
    """A partition can live in another schema from its parent. With USAGE on
    the partition's schema only, the parent is out of reach, so a write held
    on both is a write through the partition (from #399's review)."""
    engine = sa.create_engine(
        sa.URL.create(
            "postgresql+psycopg2",
            username=privileged_credentials["user"],
            password=privileged_credentials["password"],
            host=privileged_credentials["host"],
            port=privileged_credentials["port"],
            database=privileged_credentials["database"],
        )
    )
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE SCHEMA ci8_parent"))
        conn.execute(sa.text("CREATE SCHEMA ci8_child"))
        conn.execute(sa.text("CREATE TABLE ci8_parent.log (id int) PARTITION BY RANGE (id)"))
        conn.execute(
            sa.text(
                "CREATE TABLE ci8_child.log_0 PARTITION OF ci8_parent.log "
                "FOR VALUES FROM (0) TO (10)"
            )
        )
    try:
        credentials = login_with(
            "ci8_crossschema",
            "GRANT USAGE ON SCHEMA ci8_child TO {role}",
            "GRANT INSERT ON ci8_parent.log TO {role}",
            "GRANT INSERT ON ci8_child.log_0 TO {role}",
        )
        identity = _identity_of(credentials)
        assert identity.writable_tables == ("ci8_child.log_0",)
    finally:
        with engine.begin() as conn:
            conn.execute(sa.text("DROP SCHEMA ci8_child CASCADE"))
            conn.execute(sa.text("DROP SCHEMA ci8_parent CASCADE"))
        engine.dispose()
