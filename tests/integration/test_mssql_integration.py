"""Integration tests for MSSQLConnector against a real SQL Server container.

Requires Docker. The whole module is skipped when ``testcontainers`` or
``pymssql`` are not installed, and individual container startup failures
(e.g. no Docker daemon reachable) are treated as a skip rather than a
failure, since these tests may not be runnable in every environment.
"""

from __future__ import annotations

import pytest
from tests.conftest import granted

pytest.importorskip("testcontainers")
pytest.importorskip("pymssql")

from nlqueries.connectors.mssql import MSSQLConnector  # noqa: E402
from testcontainers.mssql import SqlServerContainer  # noqa: E402


@pytest.fixture(scope="module")
def mssql_container() -> object:
    try:
        container = SqlServerContainer("mcr.microsoft.com/mssql/server:2022-latest")
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker / SQL Server container unavailable: {exc}")
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def connector(mssql_container: SqlServerContainer) -> MSSQLConnector:
    conn = granted(MSSQLConnector())
    conn.connect(
        {
            "host": mssql_container.get_container_host_ip(),
            "port": mssql_container.get_exposed_port(mssql_container.port),
            "database": mssql_container.dbname,
            "user": mssql_container.username,
            "password": mssql_container.password,
        }
    )
    return conn


def test_connect_and_test_connection(connector: MSSQLConnector) -> None:
    assert connector.test_connection() is True


def test_execute_real_query(connector: MSSQLConnector) -> None:
    result = connector.execute_query("SELECT 1 AS n")

    assert result.error is None
    assert result.rows == [[1]]


def _seed(connector: MSSQLConnector, *statements: str) -> None:
    """Set up fixture data outside the answer path.

    `execute_query` is the path an answer takes and never commits, and T-SQL DDL
    is transactional -- so a `CREATE TABLE` sent through it is rolled back with
    everything else. Fixtures have to commit for themselves.
    """
    from sqlalchemy import text

    engine = connector._engine  # noqa: SLF001
    assert engine is not None
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def test_a_write_through_the_answer_path_does_not_survive(connector: MSSQLConnector) -> None:
    """The rollback, against a real SQL Server rather than a fake driver.

    This is the only place in the suite that can make that assertion: every other
    test of it uses a stub engine, which is why `capabilities.py` records
    `verified_here=False` for this dialect. T-SQL DDL is transactional, so a
    `CREATE TABLE` sent through the answer path should be observably absent
    afterwards -- and unlike the unit tests, nothing here is pretending.
    """
    result = connector.execute_query("CREATE TABLE integration_rollback_probe (id INT PRIMARY KEY)")
    assert result.error is None, f"the statement did not run at all: {result.error}"

    tables = {t.name for t in connector.extract_schema().tables}
    assert "integration_rollback_probe" not in tables, (
        "a CREATE TABLE sent through execute_query survived, so the connection was committed"
    )


def test_extract_schema_from_real_table(connector: MSSQLConnector) -> None:
    _seed(
        connector,
        "CREATE TABLE integration_test_products "
        "(id INT PRIMARY KEY, name VARCHAR(100), price FLOAT)",
    )

    spec = connector.extract_schema()

    tables = [t for t in spec.tables if t.name == "integration_test_products"]
    assert len(tables) == 1
    column_names = {c.name for c in tables[0].columns}
    assert column_names == {"id", "name", "price"}


def test_extract_query_history_returns_list(connector: MSSQLConnector) -> None:
    history = connector.extract_query_history()

    assert isinstance(history, list)


def test_execute_query_surfaces_error(connector: MSSQLConnector) -> None:
    result = connector.execute_query("SELECT * FROM nonexistent_table_xyz")

    assert result.error is not None


def _connect(container: SqlServerContainer, database: str, **entry: object) -> MSSQLConnector:
    conn = granted(MSSQLConnector())
    conn.connect(
        {
            "host": container.get_container_host_ip(),
            "port": container.get_exposed_port(container.port),
            "database": database,
            "user": container.username,
            "password": container.password,
            **entry,
        }
    )
    return conn


def _session_encrypted(connector: MSSQLConnector) -> str:
    result = connector.execute_query(
        "SELECT encrypt_option FROM sys.dm_exec_connections WHERE session_id = @@SPID"
    )
    assert result.error is None, result.error
    return str(result.rows[0][0])


def test_require_encrypts_the_session(mssql_container: SqlServerContainer) -> None:
    connector = _connect(mssql_container, mssql_container.dbname, encryption="require")

    assert _session_encrypted(connector) == "TRUE"


@pytest.mark.parametrize("entry", [{}, {"encryption": "request"}, {"encryption": "off"}])
def test_request_and_off_leave_the_session_unencrypted(
    mssql_container: SqlServerContainer, entry: dict[str, str]
) -> None:
    """The server offers TLS without forcing it. What the docs and the health
    line say about `request` rests on this."""
    connector = _connect(mssql_container, mssql_container.dbname, **entry)

    assert _session_encrypted(connector) == "FALSE"


def test_history_holds_only_the_connected_database(mssql_container: SqlServerContainer) -> None:
    """Run a marked query in two databases; each connector's history must
    hold its own marker and not the other's. Before the rows were read inside
    the connection's block, this history came back empty."""
    admin = _connect(mssql_container, "master")
    engine = admin._engine  # noqa: SLF001
    assert engine is not None
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        from sqlalchemy import text

        for name in ("nlq_hist_a", "nlq_hist_b"):
            conn.execute(text(f"IF DB_ID('{name}') IS NULL CREATE DATABASE {name}"))

    for name in ("nlq_hist_a", "nlq_hist_b"):
        connector = _connect(mssql_container, name)
        for _ in range(2):
            marker = f"marker_{name}"
            # Against a catalog view, not a bare `SELECT 1`, whose trivial plan
            # SQL Server does not keep in the plan cache.
            sql = f"SELECT name AS {marker} FROM sys.objects WHERE object_id = 3"
            assert connector.execute_query(sql).error is None

    for name, other in (("nlq_hist_a", "nlq_hist_b"), ("nlq_hist_b", "nlq_hist_a")):
        texts = [r.sql for r in _connect(mssql_container, name).extract_query_history(days=1)]
        assert any(f"marker_{name}" in t for t in texts), texts
        assert not any(f"marker_{other}" in t for t in texts), texts
