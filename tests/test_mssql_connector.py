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

import nlqueries.connectors.mssql as mssql_module
import pytest
from nlqueries import config
from nlqueries.connectors.mssql import MSSQLConnector

from tests.conftest import granted


@pytest.fixture(autouse=True)
def _no_identity_watch(monkeypatch):
    """These tests stand a mock in for `create_engine`, which SQLAlchemy cannot
    attach the identity check's pool event to; that check has its own tests
    in test_login_identity.py."""
    monkeypatch.setattr(mssql_module, "watch_engine", lambda *args: None)


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


# ---------------------------------------------------------------------------
# Transport encryption
# ---------------------------------------------------------------------------


def _connect(monkeypatch, **entry) -> tuple[MSSQLConnector, dict]:
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock(__version__="2.4.5"))
    connector = MSSQLConnector()
    with patch("nlqueries.connectors.mssql.create_engine") as mock_create_engine:
        connector.connect({**_CREDS, **entry})
    return connector, mock_create_engine.call_args.kwargs["connect_args"]


def test_encryption_defaults_to_request(monkeypatch):
    """Passed even when the entry says nothing. Left to FreeTDS, the level was
    whatever the freetds.conf the driver found said, so one entry could connect
    in plaintext on one host and encrypted on another."""
    _, connect_args = _connect(monkeypatch)

    assert connect_args["encryption"] == "request"


@pytest.mark.parametrize("level", ["require", "request", "off", "REQUIRE", " off "])
def test_encryption_is_configurable_per_connector(monkeypatch, level):
    _, connect_args = _connect(monkeypatch, encryption=level)

    assert connect_args["encryption"] == level.strip().lower()


def test_a_bare_yaml_off_means_off(monkeypatch):
    """`encryption: off` in a YAML entry loads as the boolean False."""
    _, connect_args = _connect(monkeypatch, encryption=False)

    assert connect_args["encryption"] == "off"


@pytest.mark.parametrize("level", ["default", "strict", "yes", True, 1])
def test_an_unknown_encryption_level_is_refused(monkeypatch, level):
    """`default` is pymssql's, and defers to a FreeTDS configuration this
    connector cannot see, so the posture it reports would be a guess."""
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock())
    with (
        patch("nlqueries.connectors.mssql.create_engine") as mock_create_engine,
        pytest.raises(ValueError, match="encryption must be one of"),
    ):
        MSSQLConnector().connect({**_CREDS, "encryption": level})
    mock_create_engine.assert_not_called()


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("require", "encryption is 'require'"),
        ("request", "leaves the session unencrypted unless the server forces"),
        ("off", "not encrypted"),
    ],
)
def test_the_connector_reports_its_encryption_posture(monkeypatch, level, expected):
    connector, _ = _connect(monkeypatch, encryption=level)

    assert connector.tls is not None
    assert expected in connector.tls.summary()
    # Certificate checking is FreeTDS configuration this connector does not set.
    assert connector.tls.is_fully_verified is False


def test_no_posture_before_connect():
    assert MSSQLConnector().tls is None


def test_health_shows_the_encryption_posture_for_an_mssql_connector(monkeypatch):
    """`nlqueries health` reads `connector.tls`; for SQL Server that is the
    encryption level, reported as a transport line beside the connection."""
    from nlqueries.cli import main as cli_main
    from nlqueries.connectors.base import QueryResult

    class _Answering(MSSQLConnector):
        def __init__(self):
            super().__init__()
            granted(self)

        def _execute_query(self, sql, timeout_seconds=None, max_rows=None):
            return QueryResult(
                columns=["x"], rows=[[1]], row_count=1, execution_time_ms=1.0, error=None
            )

    monkeypatch.setitem(sys.modules, "pymssql", MagicMock(__version__="2.4.5"))
    monkeypatch.setattr(
        cli_main,
        "_load_connectors",
        lambda: {"sql1": {"db_type": "mssql", "database": "analytics", "encryption": "require"}},
    )
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: _Answering)
    monkeypatch.setattr(
        cli_main, "credentials_for", lambda cid, cfg: {**_CREDS, "encryption": cfg["encryption"]}
    )
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)

    with patch("nlqueries.connectors.mssql.create_engine"):
        results = cli_main._check_connectors(None)

    transport = [r for r in results if r.service == "Database (sql1) transport"]
    assert len(transport) == 1, [(r.service, r.status, r.detail, r.error) for r in results]
    assert transport[0].status == "warn"
    assert "encryption is 'require'" in transport[0].detail


# ---------------------------------------------------------------------------
# Query history scope
# ---------------------------------------------------------------------------


def test_history_is_limited_to_the_connected_database():
    """`sys.dm_exec_query_stats` is server-wide. Without the filter a login with
    VIEW SERVER STATE mined every database on the instance into this
    connector's knowledge base. The plan's `dbid` attribute is used because
    `dm_exec_sql_text.dbid` is NULL for ad hoc and prepared statements."""
    connector = MSSQLConnector()
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value = []
    connector._engine = engine

    assert connector.extract_query_history() == []

    sql = " ".join(str(conn.execute.call_args.args[0]).split())
    assert "sys.dm_exec_plan_attributes(qs.plan_handle)" in sql
    assert "pa.attribute = 'dbid'" in sql
    assert "plan_db.dbid = DB_ID()" in sql


def test_history_rows_are_read_while_the_connection_is_checked_out():
    """Read after the `with`, the result belonged to a connection that had gone
    back to the pool, and on a real server it came back empty every time, so
    `process-history` mined nothing from SQL Server. The fake hands out rows
    only while its connection is open."""

    class _Result:
        def __init__(self, conn):
            self._conn = conn

        def mappings(self):
            if not self._conn.open:
                return iter([])
            return iter(
                [
                    {
                        "query": "SELECT n FROM orders",
                        "execution_count": 4,
                        "avg_duration_ms": 1.5,
                        "last_execution_time": "2026-10-01 00:00:00",
                    }
                ]
            )

    class _Conn:
        open = False

        def __enter__(self):
            self.open = True
            return self

        def __exit__(self, *exc):
            self.open = False

        def execute(self, *args, **kwargs):
            return _Result(self)

    class _Engine:
        def connect(self):
            return _Conn()

    connector = MSSQLConnector()
    connector._engine = _Engine()

    history = connector.extract_query_history()

    assert [record.sql for record in history] == ["SELECT n FROM orders"]
    assert history[0].execution_count == 4


def test_require_is_refused_on_a_pymssql_that_ignores_it(monkeypatch):
    """pymssql 2.3.0 to 2.4.2 accept `encryption` and connect unencrypted
    whatever it says -- measured against SQL Server 2022 -- so a connector asked
    to require encryption would open a plaintext session and report otherwise."""
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock(__version__="2.3.2"))
    with (
        patch("nlqueries.connectors.mssql.create_engine") as mock_create_engine,
        pytest.raises(RuntimeError, match="pymssql 2.3.2 ignores"),
    ):
        MSSQLConnector().connect({**_CREDS, "encryption": "require"})
    mock_create_engine.assert_not_called()


@pytest.mark.parametrize("level", [None, "request", "off"])
def test_an_older_pymssql_still_connects_and_says_the_setting_is_not_applied(monkeypatch, level):
    """Nothing these ask for is lost on an older driver, so it connects -- but
    health says the setting is not in force rather than describing the level."""
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock(__version__="2.4.2"))
    connector = MSSQLConnector()
    entry = {} if level is None else {"encryption": level}
    with patch("nlqueries.connectors.mssql.create_engine"):
        connector.connect({**_CREDS, **entry})

    assert connector.tls is not None
    assert "pymssql 2.4.2 ignores the encryption setting" in connector.tls.summary()


@pytest.mark.parametrize(
    ("version", "applies"),
    [("2.4.3", True), ("2.4.5", True), ("2.10.0", True), ("2.4.2", False), ("2.3.13", False)],
)
def test_the_first_pymssql_that_applies_encryption_is_2_4_3(version, applies):
    from nlqueries.connectors.mssql import _pymssql_applies_encryption

    assert _pymssql_applies_encryption(version) is applies


def test_an_unreadable_pymssql_version_is_taken_as_not_applying():
    from nlqueries.connectors.mssql import _pymssql_applies_encryption

    assert _pymssql_applies_encryption("") is False
    assert _pymssql_applies_encryption(MagicMock()) is False
