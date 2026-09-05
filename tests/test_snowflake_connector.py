"""
Tests for SnowflakeConnector (nlqueries.connectors.snowflake).

Unlike the PostgresConnector tests (which spin up a real database via
testcontainers), these tests mock the ``snowflake-connector-python`` driver
with ``unittest.mock`` — there is no free, ephemeral Snowflake instance to
test against, so we exercise the connector's logic against a fake driver
surface (connections, cursors, and query results).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import ColumnSpec, QueryRecord, QueryResult, SchemaSpec, TableSpec
from nlqueries.connectors.snowflake import SnowflakeConnector

from tests.conftest import granted

CREDENTIALS = {
    "account": "acme-prod",
    "user": "alice",
    "password": "test-password",
    "warehouse": "COMPUTE_WH",
    "database": "ANALYTICS",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_snowflake_is_registered_under_snowflake_key():
    assert CONNECTOR_REGISTRY["snowflake"] is SnowflakeConnector


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_builds_connection_with_expected_kwargs(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect(CREDENTIALS)

    mock_connect.assert_called_once_with(
        account="acme-prod",
        user="alice",
        password="test-password",
        warehouse="COMPUTE_WH",
        database="ANALYTICS",
    )
    assert connector._connection is mock_connect.return_value
    assert connector._database == "ANALYTICS"


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_includes_schema_only_when_provided(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect({**CREDENTIALS, "schema": "PUBLIC"})

    _, kwargs = mock_connect.call_args
    assert kwargs["schema"] == "PUBLIC"
    assert connector._db_schema == "PUBLIC"


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_connect_omits_schema_when_not_provided(mock_connect):
    mock_connect.return_value = MagicMock()

    connector = granted(SnowflakeConnector())
    connector.connect(CREDENTIALS)

    _, kwargs = mock_connect.call_args
    assert "schema" not in kwargs


def test_methods_behave_before_connect_is_called():
    connector = granted(SnowflakeConnector())

    # _require_connection() raises directly...
    with pytest.raises(RuntimeError):
        connector._require_connection()

    # ...but test_connection() and execute_query() catch it and surface it
    # gracefully (False / QueryResult.error) rather than propagating.
    assert connector.test_connection() is False

    result = connector.execute_query("SELECT 1")
    assert result.error is not None
    assert "connect()" in result.error


def _connector_with_mock_connection() -> tuple[SnowflakeConnector, MagicMock]:
    """Build a connector whose ``_connection`` is a fully-mocked driver connection."""
    connector = granted(SnowflakeConnector())
    connector._connection = MagicMock()
    connector._database = "ANALYTICS"
    return connector, connector._connection


def test_test_connection_returns_true_when_query_succeeds():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor

    assert connector.test_connection() is True
    mock_cursor.execute.assert_called_once_with("SELECT CURRENT_VERSION()")
    mock_cursor.close.assert_called_once()


def test_test_connection_returns_false_on_driver_error(caplog):
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("boom")
    mock_connection.cursor.return_value = mock_cursor

    with caplog.at_level(logging.ERROR, logger="nlqueries.connectors.snowflake"):
        assert connector.test_connection() is False

    assert any("test_connection failed" in r.message for r in caplog.records)
    # Cursor must still be closed even when execute() raises.
    mock_cursor.close.assert_called_once()


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def test_execute_query_returns_columns_and_rows():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.description = [("ONE", None), ("TWO", None)]
    mock_cursor.fetchall.return_value = [(1, "two")]
    mock_cursor.__iter__.return_value = iter([(1, "two")])
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("SELECT 1 AS one, 'two' AS two")

    assert isinstance(result, QueryResult)
    assert result.error is None
    assert result.columns == ["ONE", "TWO"]
    assert result.rows == [[1, "two"]]
    assert result.row_count == 1
    assert result.execution_time_ms >= 0
    mock_cursor.close.assert_called_once()


def _mock_cursor() -> tuple[SnowflakeConnector, MagicMock]:
    connector, mock_connection = _connector_with_mock_connection()
    cursor = MagicMock()
    cursor.description = None
    mock_connection.cursor.return_value = cursor
    return connector, cursor


def _query_call(cursor, sql="SELECT 1"):
    """The execute call carrying *sql*, not simply the last one.

    Execution is wrapped in `BEGIN` ... `ROLLBACK` now, so `call_args` is the
    rollback. Selecting the call by its statement keeps these tests about the
    timeout rather than about the position of the query in the sequence.
    """
    for call in cursor.execute.call_args_list:
        if call.args and call.args[0] == sql:
            return call
    raise AssertionError(f"{sql!r} was never executed: {cursor.execute.call_args_list}")


def test_execute_query_passes_explicit_timeout_to_cursor():
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1", timeout_seconds=30)
    assert _query_call(cursor).kwargs.get("timeout") == 30


def test_execute_query_applies_default_timeout(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 45)
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1")
    assert _query_call(cursor).kwargs.get("timeout") == 45


def test_execution_is_wrapped_in_a_transaction_that_is_rolled_back():
    """Snowflake autocommits each statement, so a write that reached the driver
    was permanent the moment it ran. BEGIN turns that off for the query and the
    ROLLBACK undoes it, on the success path as much as the failure one."""
    connector, cursor = _mock_cursor()

    connector.execute_query("SELECT 1")

    statements = [c.args[0] for c in cursor.execute.call_args_list if c.args]
    assert statements[0] == "BEGIN"
    assert statements[-1] == "ROLLBACK"
    assert "SELECT 1" in statements
    assert not any(s == "COMMIT" for s in statements)


def test_execute_query_no_timeout_when_disabled(monkeypatch):
    """`_query_call`, not `call_args`.

    Execution is wrapped in `BEGIN` ... `ROLLBACK`, so `call_args` is the
    rollback, whose kwargs are always empty -- the assertion held whatever the
    query carried, and would have stayed green against a connector that sent a
    timeout unconditionally.
    """
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    connector, cursor = _mock_cursor()
    connector.execute_query("SELECT 1")
    assert "timeout" not in _query_call(cursor).kwargs


def test_execute_query_handles_statements_with_no_result_set():
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.description = None
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("CREATE TABLE foo (id INT)")

    assert result.error is None
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0


def test_a_failing_begin_is_surfaced_and_nothing_is_committed():
    """The path the old fixture was accidentally exercising, made deliberate.

    Opening the transaction is itself a statement and can fail -- a dropped
    session is the realistic case. It has to surface as an error rather than
    fall through to run the query outside a transaction, which is the one
    outcome this connector's guard exists to prevent.
    """
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()

    def _execute(sql, *args, **kwargs):
        if sql == "BEGIN":
            raise RuntimeError("connection is closed")
        return None

    mock_cursor.execute.side_effect = _execute
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("SELECT 1")

    assert result.error is not None
    statements = [c.args[0] for c in mock_cursor.execute.call_args_list if c.args]
    assert "SELECT 1" not in statements, (
        "the query ran after BEGIN failed, so it ran outside a transaction"
    )


def test_execute_query_captures_errors_without_raising():
    """The *query* fails, not the BEGIN in front of it.

    A bare `side_effect` now fires on `BEGIN`, so this exercised a failing
    transaction start and stopped covering the path it is named for. The driver
    error a caller actually sees comes from the query.
    """
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()

    def _execute(sql, *args, **kwargs):
        if sql == "SELECT * FROM this_table_does_not_exist":
            raise RuntimeError("SQL compilation error: table does not exist")
        return None

    mock_cursor.execute.side_effect = _execute
    mock_connection.cursor.return_value = mock_cursor

    result = connector.execute_query("SELECT * FROM this_table_does_not_exist")

    statements = [c.args[0] for c in mock_cursor.execute.call_args_list if c.args]
    assert statements[0] == "BEGIN", "the transaction never opened, so the query never ran"

    assert isinstance(result, QueryResult)
    assert result.error is not None
    assert "does not exist" in result.error
    assert result.columns == []
    assert result.rows == []
    assert result.row_count == 0
    mock_cursor.close.assert_called_once()


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------

_TABLES_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "ROW_COUNT": 2,
        "COMMENT": "Customer accounts",
    },
    {"TABLE_SCHEMA": "PUBLIC", "TABLE_NAME": "ORDERS", "ROW_COUNT": 5, "COMMENT": None},
]

_COLUMNS_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "COLUMN_NAME": "ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": "Primary key",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "COLUMN_NAME": "EMAIL",
        "DATA_TYPE": "TEXT",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "COLUMN_NAME": "ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "COLUMN_NAME": "CUSTOMER_ID",
        "DATA_TYPE": "NUMBER",
        "IS_NULLABLE": "NO",
        "COMMENT": None,
    },
]

_CONSTRAINT_ROWS = [
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "CUSTOMERS",
        "CONSTRAINT_TYPE": "PRIMARY KEY",
        "COLUMN_NAME": "ID",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "CONSTRAINT_TYPE": "PRIMARY KEY",
        "COLUMN_NAME": "ID",
    },
    {
        "TABLE_SCHEMA": "PUBLIC",
        "TABLE_NAME": "ORDERS",
        "CONSTRAINT_TYPE": "FOREIGN KEY",
        "COLUMN_NAME": "CUSTOMER_ID",
    },
]


def _query_side_effect(_connection, sql):
    if "INFORMATION_SCHEMA.TABLES" in sql:
        return list(_TABLES_ROWS)
    if "INFORMATION_SCHEMA.COLUMNS" in sql:
        return list(_COLUMNS_ROWS)
    if "TABLE_CONSTRAINTS" in sql:
        return list(_CONSTRAINT_ROWS)
    raise AssertionError(f"unexpected query: {sql}")


def test_extract_schema_returns_full_schema_spec():
    connector, _ = _connector_with_mock_connection()

    with patch.object(SnowflakeConnector, "_query", side_effect=_query_side_effect):
        schema = connector.extract_schema()

    assert isinstance(schema, SchemaSpec)
    assert schema.database == "ANALYTICS"
    assert schema.extracted_at  # non-empty ISO timestamp

    tables_by_name = {t.name: t for t in schema.tables}
    assert {"CUSTOMERS", "ORDERS"} == set(tables_by_name)

    customers = tables_by_name["CUSTOMERS"]
    assert isinstance(customers, TableSpec)
    assert customers.schema == "PUBLIC"
    assert customers.row_count == 2
    assert customers.description == "Customer accounts"

    customers_columns = {c.name: c for c in customers.columns}
    assert isinstance(customers_columns["ID"], ColumnSpec)
    assert customers_columns["ID"].is_primary_key is True
    assert customers_columns["ID"].is_foreign_key is False
    assert customers_columns["ID"].nullable is False
    assert customers_columns["EMAIL"].type == "TEXT"

    orders = tables_by_name["ORDERS"]
    orders_columns = {c.name: c for c in orders.columns}
    assert orders_columns["CUSTOMER_ID"].is_foreign_key is True
    assert orders_columns["CUSTOMER_ID"].is_primary_key is False
    # Resolving the referenced table/column requires REFERENTIAL_CONSTRAINTS,
    # which is intentionally out of scope — `references` stays None.
    assert orders_columns["CUSTOMER_ID"].references is None
    assert orders_columns["ID"].is_primary_key is True


def test_extract_schema_queries_use_the_connected_database():
    connector, _ = _connector_with_mock_connection()
    seen_sql: list[str] = []

    def record_and_delegate(_connection, sql):
        seen_sql.append(sql)
        return _query_side_effect(_connection, sql)

    with patch.object(SnowflakeConnector, "_query", side_effect=record_and_delegate):
        connector.extract_schema()

    assert any("ANALYTICS.INFORMATION_SCHEMA.TABLES" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.COLUMNS" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in sql for sql in seen_sql)
    assert any("ANALYTICS.INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in sql for sql in seen_sql)


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------

_HISTORY_ROW = {
    "QUERY_TEXT": "SELECT * FROM customers",
    "EXECUTION_COUNT": 42,
    "AVG_DURATION_MS": 123.4,
    "LAST_EXECUTED": "2026-06-01 00:00:00",
}


def test_extract_query_history_prefers_account_usage():
    connector, _ = _connector_with_mock_connection()

    with patch.object(SnowflakeConnector, "_query", return_value=[_HISTORY_ROW]) as mock_query:
        history = connector.extract_query_history(days=30)

    assert history == [
        QueryRecord(
            sql="SELECT * FROM customers",
            execution_count=42,
            avg_duration_ms=123.4,
            last_executed="2026-06-01 00:00:00",
        )
    ]
    sql_used = mock_query.call_args[0][1]
    assert "SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY" in sql_used


def test_extract_query_history_falls_back_to_information_schema(caplog):
    connector, _ = _connector_with_mock_connection()

    def side_effect(_connection, sql):
        if "ACCOUNT_USAGE" in sql:
            raise RuntimeError("SQL access control error: ACCOUNT_USAGE not authorized")
        return [_HISTORY_ROW]

    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=side_effect) as mock_query,
    ):
        history = connector.extract_query_history(days=7)

    assert len(history) == 1
    assert history[0].execution_count == 42
    assert any(
        "falling back to INFORMATION_SCHEMA.QUERY_HISTORY" in r.message for r in caplog.records
    )

    # First call hit ACCOUNT_USAGE, second hit the INFORMATION_SCHEMA fallback.
    assert mock_query.call_count == 2
    second_sql = mock_query.call_args_list[1][0][1]
    assert "INFORMATION_SCHEMA.QUERY_HISTORY" in second_sql


def test_extract_query_history_returns_empty_list_when_both_sources_fail(caplog):
    connector, _ = _connector_with_mock_connection()

    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=RuntimeError("nope")),
    ):
        history = connector.extract_query_history(days=30)

    assert history == []
    assert any("returning an empty query history" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# list_security_policies (row-access + masking introspection — Block G)
# ---------------------------------------------------------------------------


def test_list_security_policies_maps_masking_and_row_access():
    connector, _ = _connector_with_mock_connection()
    refs = [
        {
            "POLICY_NAME": "mask_email",
            "POLICY_KIND": "MASKING_POLICY",
            "REF_SCHEMA_NAME": "PUBLIC",
            "REF_ENTITY_NAME": "CUSTOMERS",
            "REF_COLUMN_NAME": "EMAIL",
        },
        {
            "POLICY_NAME": "region_rap",
            "POLICY_KIND": "ROW_ACCESS_POLICY",
            "REF_SCHEMA_NAME": "PUBLIC",
            "REF_ENTITY_NAME": "ORDERS",
            "REF_COLUMN_NAME": None,
        },
        # A policy on the INFORMATION_SCHEMA is a system artifact → skipped.
        {
            "POLICY_NAME": "sys",
            "POLICY_KIND": "MASKING_POLICY",
            "REF_SCHEMA_NAME": "INFORMATION_SCHEMA",
            "REF_ENTITY_NAME": "X",
            "REF_COLUMN_NAME": "Y",
        },
    ]
    with patch.object(SnowflakeConnector, "_query", return_value=refs):
        report = connector.list_security_policies()

    assert report.supported is True
    by_name = {p.name: p for p in report.policies}
    assert set(by_name) == {"mask_email", "region_rap"}  # system schema dropped

    mask = by_name["mask_email"]
    assert mask.kind == "column"
    assert mask.columns == ["EMAIL"]
    assert mask.table == "CUSTOMERS"

    rap = by_name["region_rap"]
    assert rap.kind == "row"
    assert rap.columns == []
    # Row-access bodies aren't fetched → empty expression → caller marks untranslatable.
    assert rap.expression == ""


def test_list_security_policies_degrades_when_account_usage_unavailable(caplog):
    connector, _ = _connector_with_mock_connection()
    with (
        caplog.at_level(logging.WARNING, logger="nlqueries.connectors.snowflake"),
        patch.object(SnowflakeConnector, "_query", side_effect=RuntimeError("not granted")),
    ):
        report = connector.list_security_policies()

    assert report.supported is True
    assert report.policies == []
    assert any("POLICY_REFERENCES unavailable" in r.message for r in caplog.records)


def test_two_concurrent_queries_do_not_share_one_transaction():
    """A Snowflake transaction belongs to the session, not the cursor.

    This connector holds one connection for its lifetime, `loader.py` caches
    connector instances, and callers arrive through `asyncio.to_thread` -- so two
    `execute_query` calls really can overlap on one session. Unserialised they
    interleave as BEGIN(A), BEGIN(B) (ignored, a transaction is already open),
    query(A), ROLLBACK(A) -- which ends the transaction both were in, leaving B's
    statement running under autocommit with nothing left to roll back and B's own
    ROLLBACK a no-op.

    The guard would disappear in exactly the situation `capabilities.py` and
    `docs/database-hardening.md` tell an operator it holds, and silently: both
    queries return results and no error is raised.

    Asserted as a property of the emitted sequence rather than by racing threads,
    which would pass or fail depending on the scheduler: every statement between
    a BEGIN and its ROLLBACK must belong to the thread that opened it.
    """
    import threading

    connector, mock_connection = _connector_with_mock_connection()

    events: list[tuple[str, str]] = []
    events_lock = threading.Lock()
    both_inside = threading.Event()

    def _make_cursor(name: str) -> MagicMock:
        cur = MagicMock()
        cur.description = None

        def _execute(sql, *a, **kw):
            with events_lock:
                events.append((name, str(sql).split()[0].upper()))
            # After BEGIN, give the other thread every chance to interleave.
            if str(sql).upper().startswith("BEGIN"):
                both_inside.wait(timeout=0.5)
            return None

        cur.execute.side_effect = _execute
        return cur

    cursors = {"A": _make_cursor("A"), "B": _make_cursor("B")}
    order = iter(["A", "B"])
    mock_connection.cursor.side_effect = lambda: cursors[next(order)]

    def run(name: str) -> None:
        connector.execute_query(f"SELECT {name}")

    t1 = threading.Thread(target=run, args=("A",))
    t2 = threading.Thread(target=run, args=("B",))
    t1.start()
    t2.start()
    both_inside.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "a query did not finish"

    # Walk the sequence: between a BEGIN and the matching ROLLBACK, every
    # statement must come from the same thread.
    owner: str | None = None
    for who, verb in events:
        if verb == "BEGIN":
            assert owner is None, (
                f"{who} sent BEGIN while {owner}'s transaction was still open; "
                f"Snowflake ignores it and {who} runs under autocommit. {events}"
            )
            owner = who
        elif verb == "ROLLBACK":
            assert owner == who, f"{who} rolled back {owner}'s transaction. {events}"
            owner = None
        else:
            assert owner == who, f"{who} ran a statement inside {owner}'s transaction. {events}"


def test_a_failing_rollback_is_logged_rather_than_swallowed(caplog):
    """This connector cannot borrow the pool's justification.

    `mssql.py` and `sqlalchemy_connector.py` catch a failing rollback on the
    grounds that the connection is reset when the pool takes it back. That does
    not transfer here: this connector holds one session for the life of the
    object, so a rollback that fails can leave an explicit transaction open on
    that session until some later query happens to end it.

    Nothing is committed either way, so it is a visibility gap rather than an
    exposure -- which is the kind that must not be silent.
    """
    import logging

    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.description = None

    def _execute(sql, *args, **kwargs):
        if sql == "ROLLBACK":
            raise RuntimeError("session is gone")
        return None

    mock_cursor.execute.side_effect = _execute
    mock_connection.cursor.return_value = mock_cursor

    with caplog.at_level(logging.WARNING):
        result = connector.execute_query("SELECT 1")

    assert result.error is None, f"a failing rollback became the query's error: {result.error}"
    assert any("ROLLBACK" in r.getMessage() for r in caplog.records), (
        "the rollback failed and left no trace"
    )
    mock_cursor.close.assert_called_once()


def test_a_held_lock_fails_the_query_instead_of_blocking_forever(monkeypatch):
    """Serialising must not turn one stuck query into a stuck connector.

    The lock is held across the whole span, including the row fetch, so a query
    that never returns would block every later query on this connector. With
    `CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 0` -- documented and supported --
    nothing bounds the holder, and because `loader.py` caches the connector and
    callers arrive through `asyncio.to_thread`, the waiters occupy pool threads.

    So the *waiter* is bounded even when the holder is not, and the wait ends in
    an error a caller can read rather than in silence.
    """
    monkeypatch.setattr(config, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr("nlqueries.connectors.snowflake._LOCK_WAIT_WITHOUT_TIMEOUT_SECONDS", 0.05)
    connector, cursor = _mock_cursor()

    connector._txn_lock.acquire()  # noqa: SLF001 - stand in for a query still running
    try:
        result = connector.execute_query("SELECT 1")
    finally:
        connector._txn_lock.release()  # noqa: SLF001

    assert result.error is not None, "the query waited on the lock indefinitely"
    assert "busy" in result.error.lower(), f"unhelpful error: {result.error}"
    assert cursor.execute.call_count == 0, "the query ran without holding the lock"


def test_the_lock_is_released_when_the_query_raises(monkeypatch):
    """Otherwise one failure wedges the connector for every later caller.

    The release is in a `finally` for this reason: a query that raises on the way
    through must not leave the next caller to wait out its whole timeout.
    """
    monkeypatch.setattr("nlqueries.connectors.snowflake._LOCK_WAIT_WITHOUT_TIMEOUT_SECONDS", 0.05)
    connector, mock_connection = _connector_with_mock_connection()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("boom")
    mock_connection.cursor.return_value = mock_cursor

    assert connector.execute_query("SELECT 1").error is not None

    acquired = connector._txn_lock.acquire(timeout=0.5)  # noqa: SLF001
    assert acquired, "the lock was not released after the query raised"
    connector._txn_lock.release()  # noqa: SLF001
