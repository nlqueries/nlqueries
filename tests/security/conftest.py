"""
The disposable lab the security corpus runs against.

Two independent reviews reproduced attack chains here — a `SELECT` that
committed a write, a poisoned cache entry that reached the database, a
`--no-execute` run that executed — and then destroyed the environment they used.
What survived was prose. The cost of that showed up immediately: one finding was
recorded as open by one review, missing from another, and looked fixed to anyone
reading the code, and only running it settled the question.

So the lab lives in the repository. Everything here is synthetic, disposable and
loopback-only: a scratch PostgreSQL with a marker table, a function that writes
through a `SELECT`, a sequence, and both a privileged and a least-privilege
login built from `docs/database-hardening.md`.

The point of a marker table is that these tests assert the **absence of an
effect** rather than the presence of an error string. A denial that returns the
right message while the row still lands is not a fix.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy as sa

testcontainers_postgres = pytest.importorskip(
    "testcontainers.postgres", reason="testcontainers[postgres] is not installed"
)
from testcontainers.postgres import PostgresContainer  # noqa: E402


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker is not available in this environment"
)

#: The least-privilege login, built from docs/database-hardening.md. Every
#: statement here is one the guide tells an operator to run, so the lab and the
#: documentation cannot drift apart without a test noticing.
_READONLY_ROLE = "sec_reader"
_READONLY_PASSWORD = "lab-only-not-a-secret"

_LAB_SETUP = (
    "CREATE SCHEMA IF NOT EXISTS lab",
    "CREATE TABLE lab.marker (note text, at timestamptz DEFAULT now())",
    "CREATE TABLE lab.orders (id int, customer_id int, total numeric)",
    "INSERT INTO lab.orders VALUES (1, 10, 9.99), (2, 11, 25.00)",
    "CREATE SEQUENCE lab.counter",
    # A volatile function that writes, reached through a plain SELECT. This is
    # the shape every validator in this codebase accepts and the shape the audit
    # used to commit a row.
    """
    CREATE FUNCTION lab.mark(note text) RETURNS text AS $$
    BEGIN
        INSERT INTO lab.marker (note) VALUES (note);
        RETURN note;
    END;
    $$ LANGUAGE plpgsql VOLATILE
    """,
    f"CREATE ROLE {_READONLY_ROLE} LOGIN PASSWORD '{_READONLY_PASSWORD}'",
    f"ALTER ROLE {_READONLY_ROLE} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS",
    f"GRANT USAGE ON SCHEMA lab TO {_READONLY_ROLE}",
    f"GRANT SELECT ON lab.orders TO {_READONLY_ROLE}",
    # Deliberately no SELECT on lab.marker and no EXECUTE on lab.mark: the
    # least-privilege login should not be able to reach either.
    f"REVOKE ALL ON ALL FUNCTIONS IN SCHEMA lab FROM PUBLIC, {_READONLY_ROLE}",
    f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA lab FROM PUBLIC, {_READONLY_ROLE}",
    f"ALTER ROLE {_READONLY_ROLE} SET search_path = pg_catalog, lab",
)


@pytest.fixture(scope="session")
def lab_postgres() -> Iterator[PostgresContainer]:
    """A scratch PostgreSQL, torn down with the session."""
    container = PostgresContainer("postgres:16-alpine")
    try:
        container.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not start the lab database: {exc}")
    try:
        engine = sa.create_engine(container.get_connection_url())
        with engine.connect() as conn:
            conn.execute(sa.text("COMMIT"))
            for statement in _LAB_SETUP:
                conn.execute(sa.text(statement))
                conn.execute(sa.text("COMMIT"))
        engine.dispose()
        yield container
    finally:
        container.stop()


@pytest.fixture()
def privileged_credentials(lab_postgres: PostgresContainer) -> dict[str, Any]:
    """The database owner, as used by an unhardened deployment.

    Present so the corpus can establish which payloads the application refuses
    on its own, without assistance from database privileges.
    """
    return {
        "host": lab_postgres.get_container_host_ip(),
        "port": int(lab_postgres.get_exposed_port(5432)),
        "database": lab_postgres.dbname,
        "user": lab_postgres.username,
        "password": lab_postgres.password,
        # The lab container serves no TLS; say so rather than silently
        # downgrading, which is the behaviour the connector now refuses.
        "ssl_mode": "disable",
    }


@pytest.fixture()
def least_privilege_credentials(lab_postgres: PostgresContainer) -> dict[str, Any]:
    """The login `docs/database-hardening.md` tells operators to create."""
    return {
        "host": lab_postgres.get_container_host_ip(),
        "port": int(lab_postgres.get_exposed_port(5432)),
        "database": lab_postgres.dbname,
        "user": _READONLY_ROLE,
        "password": _READONLY_PASSWORD,
        "ssl_mode": "disable",
    }


@pytest.fixture()
def lab_admin(lab_postgres: PostgresContainer) -> Iterator[sa.Engine]:
    """A privileged engine for asserting effects — never for running payloads."""
    engine = sa.create_engine(lab_postgres.get_connection_url())
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def marker(lab_admin: sa.Engine) -> Iterator[Any]:
    """Reads the marker table and the sequence, so a test can prove nothing happened.

    Truncates before each test rather than after: a leftover row from a failing
    test should be visible while debugging it, not swept away.
    """

    class Marker:
        def rows(self) -> list[str]:
            with lab_admin.connect() as conn:
                return [r[0] for r in conn.execute(sa.text("SELECT note FROM lab.marker"))]

        def sequence_value(self) -> int | None:
            with lab_admin.connect() as conn:
                return conn.execute(
                    sa.text("SELECT last_value FROM lab.counter WHERE is_called")
                ).scalar()

    with lab_admin.connect() as conn:
        conn.execute(sa.text("TRUNCATE lab.marker"))
        conn.execute(sa.text("COMMIT"))

    yield Marker()
