"""
Tests for nlqueries.connectors.sqlite.SQLiteConnector — the stdlib file-based
SQLite connector. Runs against real in-memory / temp-file SQLite databases (no
mocks needed), mirroring test_sqlalchemy_connector.py's SQLite coverage.
"""

from __future__ import annotations

import faulthandler
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.base import SchemaSpec
from nlqueries.connectors.sqlite import SQLiteConnector, _quote_ident

from tests.conftest import granted


def _seeded(tmp_path: Path) -> SQLiteConnector:
    """A connector on a temp-file DB with a two-table FK schema and some rows."""
    db = tmp_path / "shop.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE customers (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            note TEXT
        );
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER,
            total       REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        INSERT INTO customers (id, name) VALUES (1, 'Ada'), (2, 'Grace');
        INSERT INTO orders (id, customer_id, total) VALUES (1, 1, 9.5), (2, 1, 3.0), (3, 2, 7.0);
        """
    )
    raw.commit()
    raw.close()

    c = granted(SQLiteConnector())
    c.connect({"database": str(db)})
    return c


# ---------------------------------------------------------------------------
# connect / test_connection
# ---------------------------------------------------------------------------


def test_connect_defaults_to_in_memory() -> None:
    c = granted(SQLiteConnector())
    c.connect({})  # no "database" key
    assert c._database == ":memory:"
    assert c.test_connection() is True


def test_connect_ignores_server_credential_keys() -> None:
    # host/port/user/password are accepted (same dict shape as server connectors)
    # and ignored — only "database" matters.
    c = granted(SQLiteConnector())
    c.connect({"database": ":memory:", "host": "x", "port": 5432, "user": "u", "password": "p"})
    assert c.test_connection() is True


def test_test_connection_false_before_connect() -> None:
    assert SQLiteConnector().test_connection() is False


# ---------------------------------------------------------------------------
# extract_query_history
# ---------------------------------------------------------------------------


def test_extract_query_history_always_empty(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    assert c.extract_query_history() == []
    assert c.extract_query_history(days=90, limit=10) == []


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


def test_execute_query_returns_rows(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT id, name FROM customers ORDER BY id")
    assert result.error is None
    assert result.columns == ["id", "name"]
    assert result.rows == [[1, "Ada"], [2, "Grace"]]
    assert result.row_count == 2
    assert result.execution_time_ms >= 0


def test_a_write_is_refused_by_the_database_itself(tmp_path: Path) -> None:
    """This asserted the opposite until the database was opened read-only.

    It checked that a non-SELECT came back with no columns, which required the
    UPDATE to succeed -- so the suite recorded "this connector can write to a
    customer's database" as the expected behaviour. It cannot any more, and the
    refusal comes from SQLite rather than from anything here inspecting how the
    statement was spelled.
    """
    c = _seeded(tmp_path)
    result = c.execute_query("UPDATE customers SET note = 'x' WHERE id = 1")
    assert result.error is not None
    assert "readonly" in result.error.lower()
    assert result.rows == []


def test_execute_query_surfaces_error(tmp_path: Path) -> None:
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT * FROM does_not_exist")
    assert result.error is not None
    assert "does_not_exist" in result.error
    assert result.rows == []


def test_execute_query_with_timeout_budget_still_returns(tmp_path: Path) -> None:
    # A fast query under a generous budget completes normally (watchdog cancelled).
    c = _seeded(tmp_path)
    result = c.execute_query("SELECT COUNT(*) FROM orders", timeout_seconds=30)
    assert result.error is None
    assert result.rows == [[3]]


# ---------------------------------------------------------------------------
# extract_schema
# ---------------------------------------------------------------------------


def test_extract_schema_builds_spec(tmp_path: Path) -> None:
    spec = _seeded(tmp_path).extract_schema()
    assert isinstance(spec, SchemaSpec)
    by_name = {t.name: t for t in spec.tables}
    assert set(by_name) == {"customers", "orders"}

    customers = by_name["customers"]
    assert customers.schema == "main"
    assert customers.row_count == 2
    cols = {col.name: col for col in customers.columns}
    assert cols["id"].is_primary_key is True
    assert cols["id"].type == "INTEGER"
    assert cols["name"].nullable is False  # NOT NULL
    assert cols["note"].nullable is True

    orders = by_name["orders"]
    assert orders.row_count == 3
    ocols = {col.name: col for col in orders.columns}
    assert ocols["customer_id"].is_foreign_key is True
    assert ocols["customer_id"].references == "customers.id"
    assert ocols["total"].is_foreign_key is False


def test_extract_schema_skips_internal_tables(tmp_path: Path) -> None:
    # Force creation of an internal sqlite_sequence table via AUTOINCREMENT.
    db = tmp_path / "auto.db"
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    raw.execute("INSERT INTO t (v) VALUES ('a')")
    raw.commit()
    raw.close()
    c = granted(SQLiteConnector())
    c.connect({"database": str(db)})
    names = {t.name for t in c.extract_schema().tables}
    assert names == {"t"}
    assert not any(n.startswith("sqlite_") for n in names)


# ---------------------------------------------------------------------------
# registry + helpers
# ---------------------------------------------------------------------------


def test_registered_in_connector_registry() -> None:
    assert CONNECTOR_REGISTRY.get("sqlite") is SQLiteConnector


def test_quote_ident_escapes_double_quotes() -> None:
    assert _quote_ident("orders") == '"orders"'
    assert _quote_ident('we"ird') == '"we""ird"'


def test_build_url_for_sqlite() -> None:
    from nlqueries.cli.main import _build_url

    assert _build_url("sqlite", "", 0, "/data/app.db", "", "") == "sqlite:////data/app.db"
    assert _build_url("sqlite", "", 0, ":memory:", "", "") == "sqlite:///:memory:"
    assert _build_url("sqlite", "", 0, "", "", "") == "sqlite:///:memory:"


# ---------------------------------------------------------------------------
# Concurrent use
# ---------------------------------------------------------------------------


def test_four_threads_on_one_registered_connector_all_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four threads running twenty queries each through one connector the loader
    opened: what the MCP server, or any multi-threaded caller, does.

    The pooled connector is one sqlite3 connection with a Python authorizer, and
    two threads on it at once deadlocked the process: one held the connection's
    mutex and waited for the GIL to call the authorizer, the other held the GIL
    and waited for the mutex.

    A deadlock on the GIL freezes every thread, this one included, so no join
    timeout could report it. faulthandler's watchdog runs without the GIL, and
    ends the run with every thread's stack rather than hanging it.
    """
    from nlqueries.connectors.loader import open_connector_for_agent
    from nlqueries.execution import ExecutionPolicy

    db = tmp_path / "rows.db"
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT)")
    raw.executemany(
        "INSERT INTO t (status) VALUES (?)", ((f"Value {i % 500}",) for i in range(50_000))
    )
    raw.commit()
    raw.close()
    connectors_file = tmp_path / "connectors.yaml"
    connectors_file.write_text(
        yaml.safe_dump({"agent1": {"db_type": "sqlite", "url": f"sqlite:///{db.as_posix()}"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("nlqueries.config.CONNECTORS_FILE", connectors_file)
    connector = open_connector_for_agent("agent1", ExecutionPolicy.execute_read_only())
    assert connector is not None

    results: dict[int, list[Any]] = {}
    errors: dict[int, BaseException] = {}
    barrier = threading.Barrier(4)

    def _worker(n: int) -> None:
        try:
            barrier.wait(10)
            results[n] = [
                connector.execute_query(
                    "SELECT DISTINCT status FROM t "
                    f"WHERE LOWER(TRIM(status)) = 'value {n * 20 + i}' LIMIT 2"
                )
                for i in range(20)
            ]
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[n] = exc

    faulthandler.dump_traceback_later(60, exit=True, file=sys.__stderr__)
    try:
        threads = [threading.Thread(target=_worker, args=(n,), daemon=True) for n in range(4)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, 10 - (time.monotonic() - started)))
        elapsed = time.monotonic() - started
    finally:
        faulthandler.cancel_dump_traceback_later()

    assert not any(t.is_alive() for t in threads), f"still running after {elapsed:.1f}s"
    assert errors == {}
    assert sorted(results) == [0, 1, 2, 3]
    for n, answers in results.items():
        assert [a.error for a in answers] == [None] * 20
        assert [[tuple(r) for r in a.rows] for a in answers] == [
            [(f"Value {n * 20 + i}",)] for i in range(20)
        ]


def test_a_waiting_statements_timeout_does_not_stop_the_running_one() -> None:
    """``interrupt`` stops whatever runs on the connection. A watchdog started
    before its thread held the connection would, on running out while it
    waited, stop the other thread's statement instead of its own."""
    connector = granted(SQLiteConnector())
    connector.connect({"database": ":memory:"})
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 10000000) "
        "SELECT COUNT(*) FROM c"
    )
    results: dict[str, Any] = {}

    def _run(name: str, sql: str, timeout: float) -> None:
        results[name] = connector.execute_query(sql, timeout)

    running = threading.Thread(target=_run, args=("slow", slow, 30.0))
    running.start()
    time.sleep(0.3)  # the slow statement now holds the connection
    waiting = threading.Thread(target=_run, args=("quick", "SELECT 1", 0.2))
    waiting.start()
    running.join(30)
    waiting.join(30)

    assert results["slow"].error is None, results["slow"].error
    assert [tuple(r) for r in results["slow"].rows] == [(10_000_000,)]
    assert results["quick"].error is None and [tuple(r) for r in results["quick"].rows] == [(1,)]


@pytest.mark.parametrize(
    "use",
    [
        lambda c: c.execute_query("SELECT 1"),
        lambda c: c.extract_schema(),
        lambda c: c.test_connection(),
    ],
    ids=["execute_query", "extract_schema", "test_connection"],
)
def test_every_use_of_the_connection_waits_for_the_one_in_progress(
    tmp_path: Path, use: Any
) -> None:
    connector = _seeded(tmp_path)
    done = threading.Event()

    def _call() -> None:
        use(connector)
        done.set()

    with connector._lock:  # another thread's statement, in progress
        caller = threading.Thread(target=_call, daemon=True)
        caller.start()
        assert not done.wait(0.3), "used the connection while another thread held it"
    assert done.wait(10)
