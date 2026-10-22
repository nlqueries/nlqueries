"""The corpus against the engines that run in-process: SQLite and DuckDB.

Each query the policy allows must run on the fixture data, through a read-only
connection, and read no table the policy did not see it read. The engine is
asked which tables it read -- SQLite through its authorizer, DuckDB through its
plan -- rather than trusting sqlglot's account of the statement, since the
policy's judgement is only as good as that account. PostgreSQL's half is in
``tests/security/test_sql_corpus_postgres.py``, beside the lab database it uses.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from nlqueries.sql_policy import evaluate

from tests.sql_corpus import CORPUS, FIXTURE_COLUMNS, FIXTURE_ROWS, cte_names, policy_tables


def _load(conn: Any) -> None:
    for table, columns in FIXTURE_COLUMNS.items():
        conn.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in columns)})")
        marks = ", ".join("?" for _ in columns)
        for row in FIXTURE_ROWS[table]:
            conn.execute(f"INSERT INTO {table} VALUES ({marks})", row)


@pytest.fixture(scope="module")
def sqlite_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "shop.db"
    conn = sqlite3.connect(path)
    _load(conn)
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("sql", CORPUS["sqlite"], ids=range(len(CORPUS["sqlite"])))
def test_sqlite_runs_the_corpus_and_reads_only_what_the_policy_saw(
    sqlite_file: Path, sql: str
) -> None:
    assert evaluate(sql, "sqlite").allowed
    read: set[str] = set()

    def authorizer(action: int, arg1: Any, *_rest: Any) -> int:
        if action == sqlite3.SQLITE_READ and not str(arg1).startswith("sqlite_"):
            read.add(str(arg1))
        return sqlite3.SQLITE_OK

    conn = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro", uri=True)
    try:
        conn.set_authorizer(authorizer)
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    # A CTE SQLite materialises is reported under its own name; it is not a table.
    read -= cte_names(sql, "sqlite")
    assert rows
    assert read
    assert read <= policy_tables(sql, "sqlite")


@pytest.fixture(scope="module")
def duckdb_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path_factory.mktemp("corpus") / "shop.duckdb"
    conn = duckdb.connect(str(path))
    _load(conn)
    conn.close()
    return path


def _scanned(plan: Any, found: set[str]) -> set[str]:
    if isinstance(plan, dict):
        extra = plan.get("extra_info")
        if isinstance(extra, dict) and "Table" in extra:
            found.add(str(extra["Table"]).rsplit(".", 1)[-1])
        for value in plan.values():
            _scanned(value, found)
    elif isinstance(plan, list):
        for value in plan:
            _scanned(value, found)
    return found


@pytest.mark.parametrize("sql", CORPUS["duckdb"], ids=range(len(CORPUS["duckdb"])))
def test_duckdb_runs_the_corpus_and_reads_only_what_the_policy_saw(
    duckdb_file: Path, sql: str
) -> None:
    import duckdb

    assert evaluate(sql, "duckdb").allowed
    conn = duckdb.connect(str(duckdb_file), read_only=True)
    try:
        rows = conn.execute(sql).fetchall()
        try:
            (_kind, plan), *_ = conn.execute("EXPLAIN (FORMAT JSON) " + sql).fetchall()
        except duckdb.Error as exc:  # pragma: no cover - an older DuckDB
            pytest.skip(f"this DuckDB cannot explain as JSON: {exc}")
    finally:
        conn.close()

    read = _scanned(json.loads(plan), set())
    assert rows
    assert read
    assert read <= policy_tables(sql, "duckdb")
