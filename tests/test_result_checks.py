"""Signs, read from a result, that a statement which ran missed the question.

Each check here runs no query and calls no model. The ones marked repairable
lead the orchestrator to ask for one correction; the others become notes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.result_checks import check_result


def _result(columns: list[str], rows: list[list[Any]], **kw: Any) -> QueryResult:
    return QueryResult(
        columns=columns, rows=rows, row_count=len(rows), execution_time_ms=1.0, error=None, **kw
    )


def _kinds(sql: str, result: QueryResult, dialect: str = "postgres") -> dict[str, bool]:
    return {c.kind: c.repairable for c in check_result(sql, result, dialect)}


# ---------------------------------------------------------------------------
# Repeated rows
# ---------------------------------------------------------------------------


def test_repeated_rows_are_repairable() -> None:
    sql = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id"
    result = _result(["name"], [["Ann"], ["Bo"], ["Ann"], ["Ann"]])
    concerns = check_result(sql, result, "postgres")

    assert [(c.kind, c.repairable) for c in concerns] == [("duplicate_rows", True)]
    assert concerns[0].detail.startswith("2 of the 4 rows")


def test_distinct_rows_are_fine() -> None:
    sql = "SELECT name, city FROM customers"
    assert _kinds(sql, _result(["name", "city"], [["Ann", "Oslo"], ["Ann", "Rome"]])) == {}


def test_rows_of_unhashable_values_are_compared() -> None:
    """A JSON column comes back as a dict or a list."""
    sql = "SELECT profile FROM customers"
    result = _result(["profile"], [[{"tier": "gold"}], [{"tier": "gold"}]])
    assert _kinds(sql, result) == {"duplicate_rows": True}


# ---------------------------------------------------------------------------
# A ranking headed by a row with no value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name, born FROM drivers ORDER BY born ASC LIMIT 1",
        "SELECT name, born AS birth FROM drivers ORDER BY birth LIMIT 3",
        "SELECT name, MIN(born) FROM drivers GROUP BY name ORDER BY MIN(born) LIMIT 1",
    ],
)
def test_a_ranking_headed_by_null_is_repairable(sql: str) -> None:
    result = _result(["name", "born"], [["Ray", None], ["Ann", "1950-01-01"]])
    assert _kinds(sql, result) == {"null_heads_a_ranking": True}


def test_a_ranking_headed_by_a_value_is_fine() -> None:
    sql = "SELECT name, born FROM drivers ORDER BY born ASC LIMIT 1"
    assert _kinds(sql, _result(["name", "born"], [["Ann", "1950-01-01"]])) == {}


def test_nulls_first_asked_for_in_so_many_words_is_fine() -> None:
    sql = "SELECT name, born FROM drivers ORDER BY born DESC NULLS FIRST LIMIT 1"
    assert _kinds(sql, _result(["name", "born"], [["Ray", None]])) == {}


def test_nulls_first_by_default_is_still_checked() -> None:
    """PostgreSQL sorts NULL first in a descending sort without being asked; sqlglot
    records that default as though it were written, so it is not trusted."""
    sql = "SELECT name, score FROM drivers ORDER BY score DESC LIMIT 1"
    assert _kinds(sql, _result(["name", "score"], [["Ray", None]])) == {
        "null_heads_a_ranking": True
    }


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM drivers ORDER BY born LIMIT 1",  # key not returned
        "SELECT name, born FROM drivers ORDER BY born",  # no LIMIT: not a ranking
    ],
)
def test_a_ranking_whose_key_cannot_be_seen_is_not_judged(sql: str) -> None:
    assert _kinds(sql, _result(["name", "born"], [[None, None]])) == {}


# ---------------------------------------------------------------------------
# A percentage outside 0 to 100
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [988424.0, -3, Decimal("100.5")])
def test_a_percentage_out_of_range_is_repairable(value: Any) -> None:
    sql = "SELECT region, share_percent FROM shares"
    result = _result(["region", "share_percent"], [["North", 40], ["South", value]])
    concerns = check_result(sql, result, "postgres")

    assert [(c.kind, c.repairable) for c in concerns] == [("percentage_out_of_range", True)]
    assert str(value) in concerns[0].detail


@pytest.mark.parametrize("column", ["pct_change", "growth_percent", "yoy_pct", "percent_increase"])
def test_a_change_in_percent_may_leave_the_range(column: str) -> None:
    sql = f"SELECT region, {column} FROM shares"
    assert _kinds(sql, _result(["region", column], [["North", 250.0]])) == {}


def test_a_percentage_in_range_is_fine() -> None:
    sql = "SELECT region, pct FROM shares"
    assert (
        _kinds(sql, _result(["region", "pct"], [["North", 0], ["South", 100], ["East", None]]))
        == {}
    )


def test_a_false_aggregate_is_an_answer_not_a_total_of_nothing() -> None:
    """False compares equal to 0, and is not a zero total."""
    sql = "SELECT MAX(is_active) FROM users WHERE region = 'north'"
    assert _kinds(sql, _result(["max"], [[False]])) == {}


# ---------------------------------------------------------------------------
# No rows, and totals of nothing: noted, not repaired
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM orders WHERE status = 'shipped'",
        "SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id",
        "SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 5",
    ],
)
def test_no_rows_from_a_filter_is_a_note(sql: str) -> None:
    assert _kinds(sql, _result(["id"], [])) == {"empty_result": False}


def test_no_rows_without_a_filter_is_fine() -> None:
    """An empty table is simply empty."""
    assert _kinds("SELECT id FROM orders", _result(["id"], [])) == {}


@pytest.mark.parametrize("row", [[0], [None], [0, None]])
def test_totals_of_nothing_are_a_note(row: list[Any]) -> None:
    columns = ["total", "average"][: len(row)]
    sql = (
        "SELECT SUM(total) FROM orders WHERE status = 'x'"
        if len(row) == 1
        else "SELECT SUM(total), AVG(total) FROM orders WHERE status = 'x'"
    )
    assert _kinds(sql, _result(columns, [row])) == {"blank_aggregate": False}


@pytest.mark.parametrize(
    ("sql", "row"),
    [
        ("SELECT COUNT(*) FROM orders", [12]),  # a count that found something
        ("SELECT status, COUNT(*) FROM orders GROUP BY status", ["new", 0]),  # not all aggregates
    ],
)
def test_totals_that_found_something_are_fine(sql: str, row: list[Any]) -> None:
    assert _kinds(sql, _result(["a", "b"][: len(row)], [row])) == {}


# ---------------------------------------------------------------------------
# What is not judged
# ---------------------------------------------------------------------------


def test_a_failed_statement_is_not_judged() -> None:
    result = QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=1.0, error="boom")
    assert check_result("SELECT id FROM orders WHERE x = 1", result, "postgres") == []


def test_a_statement_that_does_not_parse_is_judged_on_its_rows() -> None:
    """The rows can still repeat; the statement-shaped checks have nothing to read."""
    result = _result(["name"], [["Ann"], ["Ann"]])
    assert _kinds("SELECT ((( name FROM", result) == {"duplicate_rows": True}
