"""
tests.test_mssql_connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for MSSQLConnector wiring that don't need a live SQL Server (the
end-to-end path is covered by tests/integration/test_mssql_integration.py). SQL
Server has no SET-based statement timeout, so the connector bounds queries via
pymssql's connection-level query ``timeout`` set at connect from the config
default — asserted here by mocking ``create_engine``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from nlqueries import config
from nlqueries.connectors.mssql import MSSQLConnector

_CREDS = {"host": "sql.example.com", "database": "analytics", "user": "u", "password": "p"}


def _connect_args_from_mocked_engine(monkeypatch, timeout_default) -> dict:
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", timeout_default)
    # pymssql is an optional extra (not installed in CI); connect() imports it
    # first. Stub it so the import succeeds and we can assert the engine wiring.
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock())
    with patch("nlqueries.connectors.mssql.create_engine") as mock_create_engine:
        MSSQLConnector().connect(dict(_CREDS))
    return mock_create_engine.call_args.kwargs["connect_args"]


def test_connect_sets_query_timeout_from_config_default(monkeypatch):
    connect_args = _connect_args_from_mocked_engine(monkeypatch, 90)
    assert connect_args["timeout"] == 90


def test_connect_omits_timeout_when_disabled(monkeypatch):
    connect_args = _connect_args_from_mocked_engine(monkeypatch, 0)
    assert "timeout" not in connect_args
