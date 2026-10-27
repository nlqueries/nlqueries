"""The generic connector against a real Oracle database.

CI provisions none, so this runs only when ``NLQ_TEST_ORACLE_URL`` names a
schema the login may create tables in. Run by hand against Oracle Database Free
23.26.3 (``gvenzl/oracle-free:23.26.3-slim-faststart``) with python-oracledb
26.1.0, for example::

    NLQ_TEST_ORACLE_URL="oracle+oracledb://nlq:pw@127.0.0.1:1521/?service_name=FREEPDB1" \\
        pytest tests/integration/test_oracle_integration.py

It creates and drops its own tables. Skipped when the variable is unset or
``oracledb`` is not installed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest
from tests.conftest import granted

pytest.importorskip("oracledb")

_URL = os.environ.get("NLQ_TEST_ORACLE_URL", "")
pytestmark = pytest.mark.skipif(not _URL, reason="NLQ_TEST_ORACLE_URL is not set")

from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

_TABLE = "nlq_it_t"
_DDL_TABLE = "nlq_it_ddl"


def _drop(conn: object, table: str) -> None:
    conn.execute(  # type: ignore[attr-defined]
        text(f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    )


@pytest.fixture()
def connector() -> Iterator[SQLAlchemyConnector]:
    seed = create_engine(_URL)
    with seed.begin() as conn:
        for table in (_TABLE, _DDL_TABLE):
            _drop(conn, table)
        conn.execute(text(f"CREATE TABLE {_TABLE} (a NUMBER)"))
        conn.execute(text(f"INSERT INTO {_TABLE} VALUES (7)"))
    # A read-only transaction cannot read a table created a moment before
    # (ORA-01466 for about two seconds; see _apply_read_only), so it settles first.
    time.sleep(3)
    c = granted(SQLAlchemyConnector())
    c.connect({"url": _URL})
    try:
        yield c
    finally:
        c.close()
        with seed.begin() as conn:
            for table in (_TABLE, _DDL_TABLE):
                _drop(conn, table)
        seed.dispose()


def _rows() -> list[int]:
    engine = create_engine(_URL)
    try:
        with engine.connect() as conn:
            return [
                int(a) for a in conn.execute(text(f"SELECT a FROM {_TABLE} ORDER BY a")).scalars()
            ]
    finally:
        engine.dispose()


def test_reads_still_work(connector: SQLAlchemyConnector) -> None:
    result = connector.execute_query(f"SELECT COUNT(*) FROM {_TABLE}")

    assert result.error is None, result.error
    assert result.rows == [[1]]


@pytest.mark.parametrize(
    "statement",
    [
        f"INSERT INTO {_TABLE} VALUES (1)",
        f"UPDATE {_TABLE} SET a = 2",
        f"DELETE FROM {_TABLE}",
        f"SELECT a FROM {_TABLE} FOR UPDATE",
    ],
    ids=["insert", "update", "delete", "for update"],
)
def test_a_write_is_refused_inside_the_read_only_transaction(
    connector: SQLAlchemyConnector, statement: str
) -> None:
    result = connector.execute_query(statement)

    assert result.error is not None and "ORA-01456" in result.error, result.error
    assert _rows() == [7]


def test_the_read_only_mode_does_not_outlive_the_query(connector: SQLAlchemyConnector) -> None:
    """Transaction-scoped, so the pooled connection is not handed on read-only:
    a plain write on the same session afterwards works."""
    session = connector.execute_query("SELECT SYS_CONTEXT('USERENV', 'SID') FROM dual").rows[0][0]

    engine = connector._engine  # noqa: SLF001
    assert engine is not None
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT SYS_CONTEXT('USERENV', 'SID') FROM dual")).scalar() == session
        )
        conn.execute(text(f"INSERT INTO {_TABLE} VALUES (42)"))
        conn.commit()

    assert _rows() == [7, 42]


def test_ddl_still_stands(connector: SQLAlchemyConnector) -> None:
    """The limit, measured rather than assumed: Oracle commits implicitly before
    DDL, which ends the read-only transaction, so the grant is what stops it."""
    result = connector.execute_query(f"CREATE TABLE {_DDL_TABLE} (a NUMBER)")

    engine = create_engine(_URL)
    try:
        with engine.connect() as conn:
            made = conn.execute(
                text("SELECT COUNT(*) FROM user_tables WHERE table_name = :t"),
                {"t": _DDL_TABLE.upper()},
            ).scalar()
    finally:
        engine.dispose()
    assert made == 1, result.error
