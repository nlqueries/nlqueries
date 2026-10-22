"""The PostgreSQL corpus against a real server.

Each query the policy allows must run on the fixture data in a read-only
transaction, and read no table the policy did not see it read: PostgreSQL's
own plan (``EXPLAIN (FORMAT JSON)``) names the relations, rather than sqlglot's
account of the statement, which is what the policy judged. The SQLite and
DuckDB halves are in ``tests/test_sql_corpus_engines.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from nlqueries.sql_policy import evaluate

from tests.sql_corpus import CORPUS, FIXTURE_COLUMNS, FIXTURE_ROWS, policy_tables

_PG_TYPES = {"JSON": "JSONB", "TEXT": "TEXT"}


@pytest.fixture(scope="module")
def corpus_engine(lab_postgres: Any) -> Iterator[sa.Engine]:
    engine = sa.create_engine(lab_postgres.get_connection_url())
    with engine.begin() as conn:
        conn.execute(sa.text("DROP SCHEMA IF EXISTS corpus CASCADE"))
        conn.execute(sa.text("CREATE SCHEMA corpus"))
        for table, columns in FIXTURE_COLUMNS.items():
            ddl = ", ".join(f"{c} {_PG_TYPES.get(t, t)}" for c, t in columns)
            conn.execute(sa.text(f"CREATE TABLE corpus.{table} ({ddl})"))
            names = [c for c, _ in columns]
            insert = sa.text(
                f"INSERT INTO corpus.{table} VALUES ({', '.join(':' + n for n in names)})"
            )
            for row in FIXTURE_ROWS[table]:
                conn.execute(insert, dict(zip(names, row, strict=True)))
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.execute(sa.text("DROP SCHEMA corpus CASCADE"))
        engine.dispose()


def _relations(plan: Any, found: set[str]) -> set[str]:
    if isinstance(plan, dict):
        if "Relation Name" in plan:
            found.add(str(plan["Relation Name"]))
        for value in plan.values():
            _relations(value, found)
    elif isinstance(plan, list):
        for value in plan:
            _relations(value, found)
    return found


@pytest.mark.parametrize("sql", CORPUS["postgres"], ids=range(len(CORPUS["postgres"])))
def test_postgres_runs_the_corpus_and_reads_only_what_the_policy_saw(
    corpus_engine: sa.Engine, sql: str
) -> None:
    assert evaluate(sql, "postgres").allowed
    # A DBAPI cursor with no parameters, as the connector runs a statement:
    # given parameters, even none, psycopg2 reads `%` in `LIKE '%x'` as a
    # placeholder.
    raw = corpus_engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SET LOCAL search_path = corpus")
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
        ((plan,),) = cursor.fetchall()
        raw.rollback()
    finally:
        raw.close()

    read = _relations(plan, set())
    assert rows
    assert read
    assert read <= policy_tables(sql, "postgres")
