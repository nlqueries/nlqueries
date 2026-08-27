"""
The statements RedshiftConnector issues around a query.

CI cannot provision a Redshift cluster, so these assert the sequence sent to the
driver rather than the database's response to it. The responses were measured by
hand against Redshift Serverless on 2026-08-27:

- `SET TRANSACTION READ ONLY` then INSERT, UPDATE, DELETE or CREATE is refused
  with SQLSTATE 25006, `transaction is read-only`, while SELECT still runs.
- `SET statement_timeout TO n` cancels a longer query with SQLSTATE 57014.
  Through the connector: a 3s budget cancelled after 3.5s, an 8s budget after
  8.5s, and an ordinary query under a 30s budget returned normally.
- Four consecutive queries on one connection each read successfully and each
  refused a write, which is what the rollback between them is for.
"""

from __future__ import annotations

from typing import Any

import pytest
from nlqueries import config
from nlqueries.connectors.redshift import RedshiftConnector

from tests.conftest import granted


class _Cursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.description = [("count",)]

    def execute(self, sql: str) -> None:
        self._log.append(sql)

    def fetchmany(self, size: int) -> list[list[Any]]:
        return []

    def fetchall(self) -> list[list[Any]]:
        return []

    def close(self) -> None:
        return None


class _Connection:
    """Records the statements sent, and whether the transaction was closed."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self.statements)

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.fixture()
def connector() -> tuple[RedshiftConnector, _Connection]:
    c = granted(RedshiftConnector())
    conn = _Connection()
    c._conn = conn
    return c, conn


def test_the_transaction_is_made_read_only_before_the_query(connector) -> None:
    c, conn = connector

    c.execute_query("SELECT 1")

    assert conn.statements[0] == "SET TRANSACTION READ ONLY"
    assert conn.statements[-1] == "SELECT 1"


def test_read_only_is_the_first_statement_in_the_transaction(connector) -> None:
    """`SET TRANSACTION` applies to the transaction it opens and must precede
    any query in it, so nothing may be sent before it."""
    c, conn = connector

    c.execute_query("SELECT 1", timeout_seconds=5)

    assert conn.statements.index("SET TRANSACTION READ ONLY") == 0


def test_the_timeout_is_applied_in_milliseconds(connector) -> None:
    c, conn = connector

    c.execute_query("SELECT 1", timeout_seconds=7)

    assert "SET statement_timeout TO 7000" in conn.statements


def test_the_configured_default_is_used_when_no_budget_is_given(connector) -> None:
    c, conn = connector

    c.execute_query("SELECT 1")

    expected = int(config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS * 1000)
    assert f"SET statement_timeout TO {expected}" in conn.statements


def test_a_zero_budget_sets_no_timeout(connector) -> None:
    """Zero means unbounded, and `SET statement_timeout TO 0` would mean the
    same thing to Redshift but says it less clearly."""
    c, conn = connector

    c.execute_query("SELECT 1", timeout_seconds=0)

    assert not any(s.startswith("SET statement_timeout") for s in conn.statements)


def test_a_fractional_budget_rounds_to_at_least_one_millisecond(connector) -> None:
    c, conn = connector

    c.execute_query("SELECT 1", timeout_seconds=0.0001)

    assert "SET statement_timeout TO 1" in conn.statements


def test_the_transaction_is_closed_after_a_successful_query(connector) -> None:
    """Without this the next query cannot open a read-only transaction, and
    would run unguarded."""
    c, conn = connector

    c.execute_query("SELECT 1")

    assert conn.rollbacks == 1


def test_the_transaction_is_closed_after_a_failed_query(connector) -> None:
    c, conn = connector

    def boom(_sql: str) -> None:
        raise RuntimeError("query failed")

    cur = _Cursor(conn.statements)
    cur.execute = boom  # type: ignore[method-assign]
    conn.cursor = lambda: cur  # type: ignore[method-assign]

    result = c.execute_query("SELECT 1")

    assert result.error is not None
    assert conn.rollbacks == 1


def test_each_query_re_establishes_the_guards(connector) -> None:
    """The guards belong to a transaction, not to the connection, so they are
    issued again for every query."""
    c, conn = connector

    for _ in range(3):
        c.execute_query("SELECT 1")

    assert conn.statements.count("SET TRANSACTION READ ONLY") == 3
    assert conn.rollbacks == 3
