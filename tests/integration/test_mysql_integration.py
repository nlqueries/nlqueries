"""The generic connector against a real MySQL or MariaDB server.

CI provisions neither, so this runs only when ``NLQ_TEST_MYSQL_URL`` names a
scratch database the login may create tables in. Run by hand against MySQL
8.0.46, MySQL 8.4.11 and MariaDB 10.11.14 with PyMySQL 1.2.3, for example::

    NLQ_TEST_MYSQL_URL="mysql+pymysql://nlq:pw@127.0.0.1:3306/scratch" pytest \\
        tests/integration/test_mysql_integration.py

It creates and drops its own tables. Skipped when the variable is unset or
``pymysql`` is not installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import granted

pytest.importorskip("pymysql")

_URL = os.environ.get("NLQ_TEST_MYSQL_URL", "")
pytestmark = pytest.mark.skipif(not _URL, reason="NLQ_TEST_MYSQL_URL is not set")

from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

_TABLES = {"nlq_it_innodb": "InnoDB", "nlq_it_myisam": "MyISAM"}


@pytest.fixture()
def connector() -> Iterator[SQLAlchemyConnector]:
    seed = create_engine(_URL)
    with seed.begin() as conn:
        for name, engine in _TABLES.items():
            conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
            conn.execute(text(f"CREATE TABLE {name} (a INT) ENGINE={engine}"))
    c = granted(SQLAlchemyConnector())
    # The flag and the parameter this connector takes out, set on purpose.
    url = make_url(_URL).update_query_dict({"client_flag": "65536", "local_infile": "1"})
    c.connect({"url": url.render_as_string(hide_password=False)})
    try:
        yield c
    finally:
        c.close()
        with seed.begin() as conn:
            for name in _TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
        seed.dispose()


def _count(table: str) -> int:
    engine = create_engine(_URL)
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
    finally:
        engine.dispose()


def test_reads_still_work(connector: SQLAlchemyConnector) -> None:
    result = connector.execute_query("SELECT COUNT(*) FROM nlq_it_innodb")

    assert result.error is None
    assert result.rows == [[0]]


@pytest.mark.parametrize("table", sorted(_TABLES))
def test_a_write_is_refused_inside_the_read_only_transaction(
    connector: SQLAlchemyConnector, table: str
) -> None:
    """MyISAM included: it has no transaction, so the rollback alone left an
    INSERT into it standing."""
    result = connector.execute_query(f"INSERT INTO {table} VALUES (1)")

    assert result.error is not None and "1792" in result.error, result.error
    assert _count(table) == 0


def test_the_read_only_mode_does_not_outlive_the_query(connector: SQLAlchemyConnector) -> None:
    """Transaction-scoped, not session-scoped, so the pooled connection is not
    handed on read-only: a plain write on the same connection afterwards works."""
    query_connection = connector.execute_query("SELECT CONNECTION_ID()").rows[0][0]

    engine = connector._engine  # noqa: SLF001
    assert engine is not None
    with engine.connect() as conn:
        assert conn.execute(text("SELECT CONNECTION_ID()")).scalar() == query_connection
        conn.execute(text("INSERT INTO nlq_it_innodb VALUES (42)"))
        conn.commit()

    assert _count("nlq_it_innodb") == 1


def _is_mariadb(connector: SQLAlchemyConnector) -> bool:
    return "mariadb" in str(connector.execute_query("SELECT VERSION()").rows[0][0]).lower()


def test_the_statement_timeout_is_in_force_inside_the_transaction(
    connector: SQLAlchemyConnector,
) -> None:
    """Set after `START TRANSACTION READ ONLY`, which commits implicitly, and
    session-scoped, so it is the value the query runs under."""
    variable = "max_statement_time" if _is_mariadb(connector) else "max_execution_time"

    result = connector.execute_query(f"SELECT @@session.{variable}", timeout_seconds=2)

    assert result.error is None
    assert float(result.rows[0][0]) == (2.0 if variable == "max_statement_time" else 2000)


def test_a_query_over_its_budget_is_cancelled(connector: SQLAlchemyConnector) -> None:
    result = connector.execute_query(
        "SELECT COUNT(*) FROM information_schema.columns a, information_schema.columns b, "
        "information_schema.columns c",
        timeout_seconds=1,
    )

    assert result.error is not None
    assert "interrupted" in result.error


def test_a_second_statement_is_refused(connector: SQLAlchemyConnector) -> None:
    result = connector.execute_query("SELECT 1; SELECT 2")

    assert result.error is not None


def test_load_data_local_infile_is_refused(connector: SQLAlchemyConnector, tmp_path: Path) -> None:
    canary = tmp_path / "canary.csv"
    canary.write_text("7\n", encoding="utf-8")

    result = connector.execute_query(
        f"LOAD DATA LOCAL INFILE '{canary.as_posix()}' INTO TABLE nlq_it_myisam"
    )

    assert result.error is not None
    assert _count("nlq_it_myisam") == 0
