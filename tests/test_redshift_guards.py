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

import importlib
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


class TestTheSocketBudget:
    """How long the driver is given, which is not only for the handshake.

    ``redshift_connector`` calls ``settimeout`` once on the socket before
    connecting and never clears it, so the value bounds every later read as
    well: a query returning nothing for longer than this dies with a socket
    timeout regardless of what the server was told. Verified against the
    installed driver — one ``settimeout`` call in ``core.py``, no reset — and
    reproduced on a plain socket, where a read three seconds after a successful
    connect still expired on a one-second budget.

    The hardcoded fifteen was therefore capping every query at fifteen seconds,
    well under CONNECTOR_STATEMENT_TIMEOUT_SECONDS.
    """

    def _connect(self, monkeypatch, **overrides: Any) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class _Driver:
            @staticmethod
            def connect(**kwargs: Any) -> Any:
                captured.update(kwargs)
                return _Connection()

        monkeypatch.setitem(__import__("sys").modules, "redshift_connector", _Driver)
        connector = RedshiftConnector()
        connector.connect({"database": "dev", "host": "h", "user": "u", "password": "p"})
        return captured

    def test_the_configured_budget_is_passed_to_the_driver(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "REDSHIFT_SOCKET_TIMEOUT_SECONDS", 45)

        assert self._connect(monkeypatch)["timeout"] == 45

    def test_the_default_leaves_room_above_the_statement_timeout(self, monkeypatch) -> None:
        """Reads the source default rather than whatever this machine is set to.

        The first version asserted on `config.REDSHIFT_SOCKET_TIMEOUT_SECONDS`,
        which resolves from the environment and from any .env in the working
        directory at import time — so a developer who set the variable locally
        would see it fail with the default untouched.
        """
        monkeypatch.delenv("REDSHIFT_SOCKET_TIMEOUT_SECONDS", raising=False)
        module = importlib.reload(config)
        try:
            assert (
                module.REDSHIFT_SOCKET_TIMEOUT_SECONDS > module.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
            )
        finally:
            importlib.reload(config)

    @pytest.mark.parametrize("statement_timeout", ["30", "120", "600"])
    def test_the_default_tracks_the_statement_timeout(
        self, monkeypatch, statement_timeout: str
    ) -> None:
        """The derivation, not one value of it. A deployment that raises its
        statement timeout should not have to know this exists to keep queries
        being cancelled by the server rather than killed by the socket."""
        monkeypatch.delenv("REDSHIFT_SOCKET_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv("CONNECTOR_STATEMENT_TIMEOUT_SECONDS", statement_timeout)
        module = importlib.reload(config)
        try:
            assert float(statement_timeout) < module.REDSHIFT_SOCKET_TIMEOUT_SECONDS
        finally:
            monkeypatch.undo()
            importlib.reload(config)

    def test_tls_is_still_required(self, monkeypatch) -> None:
        """The control: the connect call carries more than the timeout, and this
        is the part of it that matters most."""
        assert self._connect(monkeypatch)["ssl"] is True
