"""The plan check: a subquery the database would re-run for every outer row.

The PostgreSQL, MySQL and MariaDB plans here are real ``EXPLAIN`` output,
captured from PostgreSQL 16, MySQL 8.4 and MariaDB 11.8 over three tables
(``tests/fixtures/plans``; each file records the engine version and the table
sizes). SQLite's are produced live, from a real database file. What these pin:
the shapes each engine re-runs per row are found, and the shapes it plans once
-- an uncorrelated subquery, a hashed one, a semi-join, an index lookup, the
same question written as a join -- are not; nothing is sent to the database
for a statement with no subquery in an expression, or for an engine that is
not checked; and a rewrite replaces the statement only when it validates and
its own plan is clean.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nlqueries.connectors.base import QueryResult
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.plan_check import (
    PerRowScan,
    _mysql_scans,
    _postgres_scans,
    _sqlite_scans,
    check_plan,
    per_row_scans,
)
from nlqueries.orchestrator.provenance import Provenance, use_provenance
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.query_settings import QuerySettings, use_query_settings

_FIXTURES = Path(__file__).parent / "fixtures" / "plans"

#: The shapes each engine runs once per outer row, reading ``b`` in full. On
#: MySQL and MariaDB the correlated IN becomes a semi-join; on PostgreSQL it
#: stays a subplan.
_PG_PER_ROW = {
    "correlated_scalar",
    "correlated_in_select_list",
    "correlated_after_join",
    "correlated_in_list",
}
_MYSQL_PER_ROW = {"correlated_scalar", "correlated_in_select_list", "correlated_after_join"}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _as_result(entry: dict[str, Any]) -> QueryResult:
    return QueryResult(
        columns=entry["columns"],
        rows=entry["rows"],
        row_count=len(entry["rows"]),
        execution_time_ms=1.0,
        error=None,
    )


def _kb(sizes: dict[str, int]) -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {
                    "name": name,
                    "row_count": rows,
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": ""},
                        {"name": "customer_id", "type": "INTEGER", "description": ""},
                        {"name": "total", "type": "REAL", "description": ""},
                    ],
                }
                for name, rows in sizes.items()
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

_PG = _fixture("postgres")
_PG_SIZES = {"a": 200_000, "b": 200_000, "c": 200_000}


#: Shapes with more to them than one scan of `b`.
_PG_EXPECTED_TABLES = {
    # A join inside the subplan, hashing `b WHERE b.k = a.k`: the hash depends
    # on the outer row, so it is rebuilt each run, and both sides are read.
    "correlated_with_hashed_join_inside": ["c", "b"],
    # A subplan inside a subplan: each is counted once, the inner one for
    # every row the outer one reads.
    "correlated_nested": ["b", "b"],
}


@pytest.mark.parametrize("shape", sorted(_PG["plans"]))
def test_postgres_finds_exactly_the_subplans_run_per_row(shape: str) -> None:
    scans = _postgres_scans([[_PG["plans"][shape]["plan"]]], _PG_SIZES)
    if shape in _PG_PER_ROW:
        assert [s.table for s in scans] == ["b"]
        assert scans[0].rows_per_run == 200_000
        # PostgreSQL's own estimate of the outer rows, recovered from its costs.
        assert 100_000 <= scans[0].runs <= 200_001
    else:
        assert [s.table for s in scans] == _PG_EXPECTED_TABLES.get(shape, [])


def test_postgres_nested_subplan_runs_for_each_row_its_parent_reads() -> None:
    outer, inner = _postgres_scans([[_PG["plans"]["correlated_nested"]["plan"]]], _PG_SIZES)
    assert outer.runs == pytest.approx(200_000, rel=0.01)
    # Once per row of `b` that each outer run reads.
    assert inner.runs == pytest.approx(200_000 * 200_000, rel=0.01)


def test_postgres_subplan_run_once_is_not_counted() -> None:
    # `a.id = 7` leaves one outer row: the subplan is evaluated once.
    plan = _PG["plans"]["correlated_one_outer_row"]["plan"]
    assert plan[0]["Plan"]["Plans"][0]["Parent Relationship"] == "SubPlan"
    assert _postgres_scans([[plan]], _PG_SIZES) == []


def test_postgres_counts_runs_of_a_subplan_under_a_nested_loop() -> None:
    # The subplan's parent is the inner index scan of a nested loop, which
    # costs one row per run: the runs are the outer side's rows.
    plan = _PG["plans"]["correlated_after_join"]["plan"]
    [scan] = _postgres_scans([[plan]], _PG_SIZES)
    assert scan.runs == pytest.approx(plan[0]["Plan"]["Plans"][0]["Plan Rows"], rel=0.01)


def test_postgres_plan_returned_as_text_reads_the_same() -> None:
    plan = _PG["plans"]["correlated_scalar"]["plan"]
    assert _postgres_scans([[json.dumps(plan)]], _PG_SIZES) == _postgres_scans([[plan]], _PG_SIZES)


def test_postgres_table_of_unknown_size_is_not_counted() -> None:
    plan = _PG["plans"]["correlated_scalar"]["plan"]
    assert _postgres_scans([[plan]], {"a": 200_000}) == []


def test_postgres_hashed_subplan_is_built_once() -> None:
    plan = _PG["plans"]["uncorrelated_not_in"]["plan"]
    subplan = plan[0]["Plan"]["Plans"][0]
    assert subplan["Parent Relationship"] == "SubPlan"
    assert "hashed SubPlan 1" in plan[0]["Plan"]["Filter"]
    assert _postgres_scans([[plan]], _PG_SIZES) == []
    # The same plan with the hash taken away is a scan per row.
    unhashed = json.loads(json.dumps(plan).replace("hashed SubPlan 1", "SubPlan 1"))
    assert [s.table for s in _postgres_scans([[unhashed]], _PG_SIZES)] == ["b"]


# ---------------------------------------------------------------------------
# MySQL and MariaDB
# ---------------------------------------------------------------------------

_MYSQL = _fixture("mysql")
_MARIADB = _fixture("mariadb")


@pytest.mark.parametrize(
    ("engine", "shape"),
    [("mysql", s) for s in sorted(_MYSQL["plans"])]
    + [("mariadb", s) for s in sorted(_MARIADB["plans"])],
)
def test_mysql_finds_exactly_the_dependent_subqueries_that_scan(engine: str, shape: str) -> None:
    entry = (_MYSQL if engine == "mysql" else _MARIADB)["plans"][shape]
    scans = _mysql_scans(_as_result(entry))
    if shape in _MYSQL_PER_ROW:
        assert [s.table for s in scans] == ["b"]
        assert scans[0].rows_per_run == 50386
    else:
        assert scans == []


def test_mysql_runs_follow_the_outer_join_and_its_filters() -> None:
    # `c` (49,123 rows, 33.33% kept) then `a` by primary key: the subquery,
    # attached to `a`, runs for each joined row.
    [scan] = _mysql_scans(_as_result(_MYSQL["plans"]["correlated_after_join"]))
    assert scan.runs == int(49123 * 0.3333 * 1)
    [simple] = _mysql_scans(_as_result(_MYSQL["plans"]["correlated_scalar"]))
    assert simple.runs == 50386


def test_mysql_subquery_runs_for_every_row_the_last_table_examines() -> None:
    # The last table's `filtered` includes the subquery's own condition, which
    # is evaluated on each row examined to decide whether it is kept.
    entry = json.loads(json.dumps(_MYSQL["plans"]["correlated_scalar"]))
    entry["rows"][0][entry["columns"].index("filtered")] = "33.33"
    [scan] = _mysql_scans(_as_result(entry))
    assert scan.runs == 50386


def test_mysql_typed_values_read_as_the_text_ones() -> None:
    # The driver returns `rows` as an int and `filtered` as a float; the
    # fixtures, from the command line, hold text.
    entry = _MYSQL["plans"]["correlated_after_join"]
    typed = dict(entry)
    cols = entry["columns"]
    rows_at, filtered_at = cols.index("rows"), cols.index("filtered")
    typed["rows"] = [
        [int(v) if i == rows_at else float(v) if i == filtered_at else v for i, v in enumerate(row)]
        for row in entry["rows"]
    ]
    assert _mysql_scans(_as_result(typed)) == _mysql_scans(_as_result(entry))


def test_mysql_output_in_another_format_is_not_read() -> None:
    qr = QueryResult(
        columns=["EXPLAIN"], rows=[["-> Filter ..."]], row_count=1, execution_time_ms=1, error=None
    )
    assert _mysql_scans(qr) == []


# ---------------------------------------------------------------------------
# SQLite, live
# ---------------------------------------------------------------------------

_SQLITE_SIZES = {"orders": 100_000, "customers": 20_000}


@pytest.fixture
def sqlite_connector(tmp_path: Path) -> Any:
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INT, total REAL);"
        "CREATE TABLE customers(id INTEGER PRIMARY KEY, customer_id INT, total REAL, region INT);"
        "CREATE INDEX customers_by_customer ON customers(customer_id);"
        "INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 30.0), (3, 2, 5.0);"
        "INSERT INTO customers VALUES (1, 1, 0.0, 1), (2, 2, 0.0, 2);"
    )
    conn.commit()
    conn.close()
    connector = SQLiteConnector()
    connector.connect({"database": str(path)})
    connector.bind_execution_policy(ExecutionPolicy.execute_read_only())
    yield connector
    connector.close()


_CORRELATED = (
    "SELECT o.id FROM orders o WHERE o.total > "
    "(SELECT AVG(i.total) FROM orders i WHERE i.customer_id = o.customer_id)"
)
_AS_A_JOIN = (
    "SELECT o.id FROM orders o JOIN (SELECT customer_id, AVG(total) AS m FROM orders "
    "GROUP BY customer_id) g ON g.customer_id = o.customer_id WHERE o.total > g.m"
)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # Reported by the table's name, not the alias the plan shows.
        (_CORRELATED, [("orders", 100_000, 100_000)]),
        (
            "SELECT c.id, (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) "
            "FROM customers c",
            [("orders", 20_000, 100_000)],
        ),
        # An index lookup per row, not a scan.
        (
            "SELECT o.id FROM orders o WHERE o.total > "
            "(SELECT AVG(c.total) FROM customers c WHERE c.customer_id = o.customer_id)",
            [],
        ),
        ("SELECT id FROM orders WHERE total > (SELECT AVG(total) FROM orders)", []),
        (_AS_A_JOIN, []),
        # The outer side is a five-row derived table that shares a table's
        # name: not that table's size.
        (
            "SELECT orders.id FROM (SELECT * FROM orders LIMIT 5) AS orders WHERE orders.total > "
            "(SELECT AVG(i.total) FROM orders i WHERE i.customer_id = orders.customer_id)",
            [],
        ),
        # One alias for two tables, and the plan says only `SCAN x` for each:
        # the smaller table's size for both.
        (
            "SELECT x.id FROM customers x WHERE x.total > "
            "(SELECT AVG(x.total) FROM orders x WHERE x.customer_id = region)",
            [("customers", 20_000, 20_000)],
        ),
    ],
)
def test_sqlite_finds_correlated_subqueries_that_scan(
    sqlite_connector: Any, sql: str, expected: list[tuple[str, int, int]]
) -> None:
    scans = asyncio.run(per_row_scans(sql, "sqlite", sqlite_connector, _kb(_SQLITE_SIZES)))
    assert [(s.table, s.runs, s.rows_per_run) for s in scans or []] == expected


def test_sqlite_before_3_36_names_the_table_and_its_alias() -> None:
    qr = QueryResult(
        columns=["id", "parent", "notused", "detail"],
        rows=[
            [2, 0, 0, "SCAN TABLE orders AS customers"],
            [7, 0, 0, "CORRELATED SCALAR SUBQUERY 1"],
            [12, 7, 0, "SCAN TABLE customers AS i"],
        ],
        row_count=3,
        execution_time_ms=1,
        error=None,
    )
    # `customers` is also an alias, for `orders`; the old form says which
    # table each scan reads, so the alias is not consulted.
    sql = (
        "SELECT customers.id FROM orders AS customers WHERE customers.total > "
        "(SELECT AVG(i.total) FROM customers AS i WHERE i.customer_id = customers.customer_id)"
    )
    [scan] = _sqlite_scans(qr, sql, _SQLITE_SIZES, "sqlite")
    assert (scan.table, scan.runs, scan.rows_per_run) == ("customers", 100_000, 20_000)


# ---------------------------------------------------------------------------
# What is sent to the database
# ---------------------------------------------------------------------------


def _plan_result(rows: list[list[Any]] | None = None, error: str | None = None) -> QueryResult:
    return QueryResult(
        columns=["id", "parent", "notused", "detail"],
        rows=rows or [],
        row_count=len(rows or []),
        execution_time_ms=1,
        error=error,
    )


def test_a_statement_with_no_expression_subquery_is_not_explained() -> None:
    connector = MagicMock()
    sql = "SELECT o.id FROM orders o JOIN (SELECT customer_id FROM orders) g USING (customer_id)"
    assert asyncio.run(per_row_scans(sql, "sqlite", connector, _kb(_SQLITE_SIZES))) == []
    connector.execute_query.assert_not_called()


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "tsql"])
def test_an_engine_not_checked_is_not_explained(dialect: str) -> None:
    connector = MagicMock()
    assert asyncio.run(per_row_scans(_CORRELATED, dialect, connector, _kb({}))) is None
    connector.execute_query.assert_not_called()


@pytest.mark.parametrize(
    ("dialect", "prefix"),
    [
        ("postgres", "EXPLAIN (FORMAT JSON) "),
        ("postgresql", "EXPLAIN (FORMAT JSON) "),
        ("mysql", "EXPLAIN "),
        ("mariadb", "EXPLAIN "),
        ("sqlite", "EXPLAIN QUERY PLAN "),
    ],
)
def test_the_plan_is_read_with_a_plain_explain(dialect: str, prefix: str) -> None:
    connector = MagicMock()
    connector.execute_query.return_value = _plan_result()
    asyncio.run(per_row_scans(_CORRELATED + " ;", dialect, connector, _kb({})))
    sent = connector.execute_query.call_args.args[0]
    assert sent == prefix + _CORRELATED, "never ANALYZE, and no trailing semicolon"
    assert "ANALYZE" not in sent


def test_a_statement_the_policy_refuses_is_not_explained() -> None:
    connector = MagicMock()
    sql = "SELECT id FROM orders WHERE total > (SELECT load_extension('x'))"
    assert asyncio.run(per_row_scans(sql, "sqlite", connector, _kb({}))) is None
    connector.execute_query.assert_not_called()


@pytest.mark.parametrize(
    "outcome", [RuntimeError("row filter refused it"), _plan_result(error="no such table")]
)
def test_a_plan_that_cannot_be_read_is_not_checked(outcome: Any) -> None:
    connector = MagicMock()
    if isinstance(outcome, Exception):
        connector.execute_query.side_effect = outcome
    else:
        connector.execute_query.return_value = outcome
    assert asyncio.run(per_row_scans(_CORRELATED, "sqlite", connector, _kb({}))) is None


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


class _LLM:
    supports_prompt_caching = False

    def __init__(
        self, reply: str | Exception, delay: float = 0.0, until: float | None = None
    ) -> None:
        """*delay* is how long the call takes; *until*, a ``time.monotonic()``
        value, makes it last until then instead, however long setup took."""
        self._reply = reply
        self._delay = delay
        self._until = until
        self.calls: list[str] = []

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        self.calls.append(user)
        delay = self._delay if self._until is None else self._until - time.monotonic()
        await asyncio.sleep(max(0.0, delay))
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    async def astream(self, system: Any, user: str) -> Any:
        for t in ():
            yield t


def _first(sql: str = _CORRELATED) -> SQLGenerationResult:
    return SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )


#: The provenance the last `_check` recorded, and whether it asked the model.
_RECORDED: dict[str, Provenance] = {}
_ASKED: dict[str, bool] = {}


def _check(
    connector: Any,
    llm: _LLM,
    *,
    sizes: dict[str, int] | None = None,
    started: float | None = None,
    deadline: float | None = None,
) -> SQLGenerationResult:
    first = _first()
    _RECORDED["last"] = collected = Provenance()
    with use_provenance(collected):
        checked = asyncio.run(
            check_plan(
                first,
                connector,
                question="which orders are above their customer's average",
                knowledge_base=_kb(_SQLITE_SIZES if sizes is None else sizes),
                dialect="sqlite",
                llm=llm,  # type: ignore[arg-type]
                system="system",
                started=started,
                deadline=deadline,
            )
        )
    _ASKED["last"] = checked.asked_model
    return checked.result


def _rewrite_record() -> dict[str, Any] | None:
    return _RECORDED["last"].to_dict()["plan_rewrite"]


def test_a_clean_rewrite_replaces_the_statement(sqlite_connector: Any) -> None:
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    checked = _check(sqlite_connector, llm)
    assert checked.sql == _AS_A_JOIN
    assert checked.is_valid
    assert checked.attempt_count == 2
    [prompt] = llm.calls
    assert "which orders are above their customer's average" in prompt
    assert _CORRELATED in prompt
    assert "reads orders in full (about 100,000 rows)" in prompt
    assert "10,000,000,000 rows read in all" in prompt
    # The answer-changing mistake the usual rewrite makes.
    assert "LEFT JOIN" in prompt
    # Provenance keeps the statement as generated, beside why it was replaced.
    assert _rewrite_record() == {
        "attempted": True,
        "changed": True,
        "original_sql": _CORRELATED,
        "findings": [PerRowScan("orders", 100_000, 100_000).describe()],
        "reason": None,
    }


def test_the_rewrite_answers_as_the_original_did(sqlite_connector: Any) -> None:
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    checked = _check(sqlite_connector, llm)
    before = sqlite_connector.execute_query(_CORRELATED)
    after = sqlite_connector.execute_query(checked.sql)
    assert sorted(after.rows) == sorted(before.rows) == [[2]]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        # Still re-runs per row, written differently.
        (
            "<sql>SELECT o.id FROM orders o WHERE o.total > (SELECT AVG(i.total) FROM orders i "
            "WHERE i.customer_id = o.customer_id AND i.total > 0)</sql>",
            "rejected: still re-runs a subquery per row",
        ),
        # Refused by validation: a table the knowledge base does not list.
        (
            "<sql>SELECT o.id FROM orders o JOIN averages g ON g.customer_id = o.customer_id</sql>",
            "rejected: refused by validation",
        ),
        # The same statement back.
        (f"<sql>{_CORRELATED}</sql>", "rejected: returned unchanged"),
        (RuntimeError("provider unavailable"), "failed: the model call raised"),
    ],
    ids=["still-per-row", "invalid", "unchanged", "llm-failed"],
)
def test_otherwise_the_statement_runs_as_written(
    sqlite_connector: Any, reply: Any, reason: str
) -> None:
    llm = _LLM(reply)
    checked = _check(sqlite_connector, llm)
    assert checked.sql == _CORRELATED
    assert checked.attempt_count == 1
    assert len(llm.calls) == 1
    record = _rewrite_record()
    assert record is not None
    assert (record["attempted"], record["changed"], record["reason"]) == (True, False, reason)
    assert record["original_sql"] == _CORRELATED


def test_a_rewrite_whose_plan_cannot_be_read_is_not_run(sqlite_connector: Any) -> None:
    real = sqlite_connector.execute_query
    calls: list[str] = []

    def explain(sql: str, *args: Any) -> QueryResult:
        calls.append(sql)
        if len(calls) == 1:
            return real(sql, *args)
        return _plan_result(error="database is locked")

    connector = MagicMock()
    connector.execute_query.side_effect = explain
    # A rewrite with a subquery in an expression, so its plan must be read.
    with_subquery = (
        "SELECT o.id FROM orders o WHERE o.customer_id IN (SELECT customer_id FROM customers)"
    )
    llm = _LLM(f"<sql>{with_subquery}</sql>")
    assert _check(connector, llm).sql == _CORRELATED
    assert len(calls) == 2
    record = _rewrite_record()
    assert record is not None and record["reason"] == "rejected: its plan could not be read"


def test_no_rewrite_starts_with_less_time_left_than_the_attempt_has_taken(
    sqlite_connector: Any,
) -> None:
    # Three seconds left, ten already spent: a rewrite would outlast the deadline.
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    now = time.monotonic()
    assert _check(sqlite_connector, llm, started=now - 10, deadline=now + 3).sql == _CORRELATED
    assert llm.calls == []
    assert _rewrite_record() == {
        "attempted": False,
        "changed": False,
        "original_sql": _CORRELATED,
        "findings": [PerRowScan("orders", 100_000, 100_000).describe()],
        "reason": "skipped: too little time before the deadline",
    }


def test_the_room_for_a_rewrite_is_judged_after_the_plan_is_read(
    sqlite_connector: Any,
) -> None:
    # Room before the EXPLAIN (3 s left, none spent); none after it (1.6 s of
    # it spent, 1.4 s left).
    def slow(sql: str, *args: Any) -> QueryResult:
        time.sleep(1.6)
        return sqlite_connector.execute_query(sql, *args)

    connector = MagicMock()
    connector.execute_query.side_effect = slow
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    now = time.monotonic()
    assert _check(connector, llm, started=now, deadline=now + 3).sql == _CORRELATED
    assert llm.calls == []


def test_no_rewrite_starts_under_a_second_before_the_deadline(sqlite_connector: Any) -> None:
    # 1.3 s left when the plan is read, under a second once it has been.
    def slow(sql: str, *args: Any) -> QueryResult:
        time.sleep(0.5)
        return sqlite_connector.execute_query(sql, *args)

    connector = MagicMock()
    connector.execute_query.side_effect = slow
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    assert _check(connector, llm, deadline=time.monotonic() + 1.3).sql == _CORRELATED
    assert llm.calls == []


def test_the_rewrite_plan_is_bounded_by_the_time_left_too(sqlite_connector: Any) -> None:
    connector = MagicMock(wraps=sqlite_connector)
    with_subquery = (
        "SELECT o.id FROM orders o WHERE o.customer_id IN (SELECT customer_id FROM customers)"
    )
    llm = _LLM(f"<sql>{with_subquery}</sql>")
    now = time.monotonic()
    assert _check(connector, llm, started=now, deadline=now + 4).sql == with_subquery
    explains = [c.args for c in connector.execute_query.call_args_list]
    assert [a[0].split(" ", 3)[:3] for a in explains] == [["EXPLAIN", "QUERY", "PLAN"]] * 2
    assert all(a[1] <= 4 for a in explains)


def test_with_time_for_it_a_rewrite_starts(sqlite_connector: Any) -> None:
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    now = time.monotonic()
    assert _check(sqlite_connector, llm, started=now - 2, deadline=now + 30).sql == _AS_A_JOIN


def test_each_explain_is_bounded_by_the_time_left() -> None:
    connector = MagicMock()
    connector.execute_query.return_value = _plan_result()
    _check(connector, _LLM(""), deadline=time.monotonic() + 4)
    timeout = connector.execute_query.call_args.args[1]
    assert 1.0 <= timeout <= 4 < 5.0
    _check(connector, _LLM(""))
    assert connector.execute_query.call_args.args[1] == 5.0


def test_under_a_second_before_the_deadline_no_plan_is_read() -> None:
    connector = MagicMock()
    assert _check(connector, _LLM(""), deadline=time.monotonic() + 0.5).sql == _CORRELATED
    connector.execute_query.assert_not_called()


def test_below_the_limit_no_rewrite_is_asked_for(sqlite_connector: Any) -> None:
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    # 31,622 x 31,622 is just under a billion row reads.
    assert _check(sqlite_connector, llm, sizes={"orders": 31_622}).sql == _CORRELATED
    assert llm.calls == []
    with use_query_settings(QuerySettings(plan_check_min_row_reads=31_622 * 31_622)):
        assert _check(sqlite_connector, llm, sizes={"orders": 31_622}).sql == _AS_A_JOIN


def test_switched_off_the_plan_is_not_read() -> None:
    connector = MagicMock()
    with use_query_settings(QuerySettings(plan_check=False)):
        assert _check(connector, _LLM("")).sql == _CORRELATED
    connector.execute_query.assert_not_called()


def test_nothing_is_recorded_when_no_rewrite_is_due(sqlite_connector: Any) -> None:
    _check(sqlite_connector, _LLM(""), sizes={"orders": 10})
    assert _rewrite_record() is None


def test_describe_names_the_table_and_the_estimate() -> None:
    scan = PerRowScan("orders", 40_000, 40_000)
    assert scan.row_reads == 1_600_000_000
    assert scan.describe() == (
        "a subquery reads orders in full (about 40,000 rows) once for each of about "
        "40,000 rows of the outer query, about 1,600,000,000 rows read in all"
    )


# ---------------------------------------------------------------------------
# In the orchestrator
# ---------------------------------------------------------------------------


def _ask(
    connector: Any, llm: _LLM, *, deadline: float | None = None, generating: float = 0.0
) -> tuple[dict, Any]:
    """Ask through the orchestrator; *generating* is how long generation takes."""
    first = _first()

    async def generate(*_a: Any, **_kw: Any) -> SQLGenerationResult:
        await asyncio.sleep(generating)
        return first

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        safe = re.sub(r"[^\w.-]", "_", "agent1")
        (kb_path / f"{safe}.yaml").write_text(yaml.dump(_kb(_SQLITE_SIZES)), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(side_effect=generate),
            ),
            patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
            patch("nlqueries.orchestrator.orchestrator.get_tracer") as tracer,
        ):
            cfg.KB_PATH = kb_path
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0
            span = tracer.return_value.start_as_current_span.return_value.__enter__.return_value

            async def collect() -> list[str]:
                return [
                    tok
                    async for tok in Orchestrator().handle_question(
                        "which orders are above their customer's average",
                        "agent1",
                        dialect="sqlite",
                        execution=ExecutionPolicy.execute_read_only(),
                        deadline=deadline,
                    )
                ]

            tokens = asyncio.run(collect())
    return json.loads(tokens[-1]), span


def _ran(connector: MagicMock, sql: str) -> list[Any]:
    """The timeouts *sql* was executed with."""
    return [c.args[1] for c in connector.execute_query.call_args_list if c.args[0] == sql]


def test_the_orchestrator_runs_the_rewrite_and_reports_it(sqlite_connector: Any) -> None:
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    frame, span = _ask(sqlite_connector, llm)
    assert frame["sql"] == _AS_A_JOIN, "the frame names the statement behind the rows"
    assert frame["sql_table"]["rows"] == [[2]]
    assert frame["attempt_count"] == 2
    span.set_attribute.assert_any_call("plan_rewrite", True)


def test_the_orchestrator_bounds_the_rewrite_by_the_time_left(sqlite_connector: Any) -> None:
    connector = MagicMock(wraps=sqlite_connector)
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    frame, _span = _ask(connector, llm, deadline=time.monotonic() + 60)
    assert frame["sql"] == _AS_A_JOIN
    [timeout] = _ran(connector, _AS_A_JOIN)
    # Not the 120 s statement timeout: what is left of the caller's 60.
    assert timeout is not None and 50 < timeout <= 60


@pytest.mark.parametrize(
    ("reply", "expected"),
    [(f"<sql>{_AS_A_JOIN}</sql>", _AS_A_JOIN), ("<sql>SELECT id FROM ghost</sql>", _CORRELATED)],
    ids=["rewrite-kept", "rewrite-refused"],
)
def test_whatever_runs_after_the_rewrite_call_is_bounded_by_the_time_left(
    sqlite_connector: Any, reply: str, expected: str
) -> None:
    # Kept or refused, the statement starts after a model call the caller's
    # 60 s also had to pay for.
    connector = MagicMock(wraps=sqlite_connector)
    frame, _span = _ask(connector, _LLM(reply), deadline=time.monotonic() + 60)
    assert frame["sql"] == expected
    [timeout] = _ran(connector, expected)
    assert timeout is not None and 50 < timeout <= 60


def test_with_under_a_second_left_after_the_rewrite_call_nothing_runs(
    sqlite_connector: Any,
) -> None:
    # Room for the call when it starts (8 s left, far less spent); the call
    # ends half a second before the deadline. Timed against the deadline, not
    # from the start, so a slow setup cannot take the room away first.
    connector = MagicMock(wraps=sqlite_connector)
    deadline = time.monotonic() + 8
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>", until=deadline - 0.5)
    frame, _span = _ask(connector, llm, deadline=deadline)
    assert len(llm.calls) == 1
    assert _ran(connector, _CORRELATED) == [] and _ran(connector, _AS_A_JOIN) == []
    assert frame["sql_table"]["error"].startswith("Not run: the caller's deadline")


@pytest.mark.parametrize(
    ("sizes", "reply", "asked"),
    [
        ({"orders": 10}, "", False),
        (None, f"<sql>{_AS_A_JOIN}</sql>", True),
        (None, f"<sql>{_CORRELATED}</sql>", True),
        (None, RuntimeError("provider unavailable"), True),
    ],
    ids=["no-rewrite-due", "kept", "unchanged", "llm-failed"],
)
def test_the_caller_is_told_whether_the_model_was_asked(
    sqlite_connector: Any, sizes: Any, reply: Any, asked: bool
) -> None:
    _check(sqlite_connector, _LLM(reply), sizes=sizes)
    assert _ASKED["last"] is asked


def test_the_caller_is_told_the_model_was_not_asked_when_time_ran_short(
    sqlite_connector: Any,
) -> None:
    now = time.monotonic()
    _check(sqlite_connector, _LLM(f"<sql>{_AS_A_JOIN}</sql>"), started=now - 10, deadline=now + 3)
    assert _ASKED["last"] is False


def test_the_orchestrator_counts_the_time_generation_took(sqlite_connector: Any) -> None:
    # Generation took 3 s and 2 s are left: a rewrite would take about as long
    # as the attempt so far, so it is not started.
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    frame, _span = _ask(sqlite_connector, llm, deadline=time.monotonic() + 5, generating=3.0)
    assert llm.calls == []
    assert frame["sql"] == _CORRELATED


def test_the_orchestrator_runs_the_statement_when_the_deadline_is_near(
    sqlite_connector: Any,
) -> None:
    connector = MagicMock(wraps=sqlite_connector)
    llm = _LLM(f"<sql>{_AS_A_JOIN}</sql>")
    frame, span = _ask(connector, llm, deadline=time.monotonic() + 0.5)
    assert llm.calls == []
    assert frame["sql"] == _CORRELATED
    assert frame["sql_table"]["rows"] == [[2]]
    assert ("plan_rewrite", True) not in [c.args for c in span.set_attribute.call_args_list]
    # The statement as generated keeps its own timeout, as before the check.
    assert _ran(connector, _CORRELATED) == [None]
