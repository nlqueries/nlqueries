"""Integration tests for MSSQLConnector against a real SQL Server container.

Requires Docker. The whole module is skipped when ``testcontainers`` or
``pymssql`` are not installed, and individual container startup failures
(e.g. no Docker daemon reachable) are treated as a skip rather than a
failure, since these tests may not be runnable in every environment.
"""

from __future__ import annotations

import pytest

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
    conn = MSSQLConnector()
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


def test_extract_schema_from_real_table(connector: MSSQLConnector) -> None:
    connector.execute_query(
        "CREATE TABLE integration_test_products "
        "(id INT PRIMARY KEY, name VARCHAR(100), price FLOAT)"
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
