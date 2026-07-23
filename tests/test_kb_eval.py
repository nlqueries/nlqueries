"""
Tests for nlqueries.kb_eval (SYL-3.2) — the community KB regression check: re-ask
capsules/golden questions and assert the generated SQL parses and references only
known tables.
"""

from __future__ import annotations

import pytest
from nlqueries.kb_eval import (
    CaseOutcome,
    EvalQuestion,
    build_cases,
    check_generated_sql,
    known_tables,
    run_eval,
)

_KB = {
    "schema": {"tables": [{"name": "orders"}, {"name": "customers"}]},
    "query_capsules": [
        {"intent": "how many orders?", "template": "SELECT COUNT(*) FROM orders"},
        {"intent": "top customers", "template": "SELECT * FROM customers"},
        {"frequency": 1},  # no intent → skipped
    ],
}
_KNOWN = {"orders", "customers"}


# ---------------------------------------------------------------------------
# known_tables
# ---------------------------------------------------------------------------


def test_known_tables() -> None:
    assert known_tables(_KB) == {"orders", "customers"}


def test_known_tables_tolerates_partial() -> None:
    assert known_tables({}) == set()
    assert known_tables({"schema": {"tables": [{}, "bad", {"name": "t"}]}}) == {"t"}


# ---------------------------------------------------------------------------
# check_generated_sql
# ---------------------------------------------------------------------------


def test_valid_sql_passes() -> None:
    assert check_generated_sql("SELECT COUNT(*) FROM orders", _KNOWN) == (True, "ok")


def test_join_of_known_tables_passes() -> None:
    sql = "SELECT o.id FROM orders o JOIN customers c ON o.cid = c.id"
    ok, _ = check_generated_sql(sql, _KNOWN)
    assert ok


def test_unknown_table_fails() -> None:
    ok, reason = check_generated_sql("SELECT * FROM secrets", _KNOWN)
    assert not ok
    assert "secrets" in reason


def test_cte_name_is_not_an_unknown_table() -> None:
    sql = "WITH t AS (SELECT * FROM orders) SELECT * FROM t"
    assert check_generated_sql(sql, _KNOWN)[0] is True


@pytest.mark.parametrize("sql", [None, "", "SELCT ??? FROM"])
def test_bad_sql_fails(sql: str | None) -> None:
    assert check_generated_sql(sql, _KNOWN)[0] is False


def test_non_select_fails() -> None:
    ok, reason = check_generated_sql("DELETE FROM orders", _KNOWN)
    assert not ok
    assert "SELECT" in reason


def test_expect_tables_present_passes() -> None:
    sql = "SELECT o.id FROM orders o JOIN customers c ON o.cid = c.id"
    assert check_generated_sql(sql, _KNOWN, expect_tables=["orders", "customers"])[0] is True


def test_expect_tables_missing_fails() -> None:
    ok, reason = check_generated_sql("SELECT * FROM orders", _KNOWN, expect_tables=["customers"])
    assert not ok
    assert "customers" in reason


# ---------------------------------------------------------------------------
# build_cases
# ---------------------------------------------------------------------------


def test_build_cases_from_capsules() -> None:
    cases = build_cases(_KB, "a", max_cases=50)
    assert [c.question for c in cases] == ["how many orders?", "top customers"]
    assert all(c.source == "capsule" for c in cases)


def test_build_cases_merges_golden_scoped_to_agent() -> None:
    golden = [
        {"q": "revenue?", "agent": "a", "expect_tables": ["orders"]},
        {"q": "for another agent", "agent": "b"},
        {"q": "unscoped", "expect_tables": ["customers"]},
    ]
    cases = build_cases(_KB, "a", golden=golden, max_cases=50)
    golden_cases = [c for c in cases if c.source == "golden"]
    questions = {c.question for c in golden_cases}
    assert questions == {"revenue?", "unscoped"}  # agent "b" excluded
    revenue = next(c for c in golden_cases if c.question == "revenue?")
    assert revenue.expect_tables == ["orders"]


def test_build_cases_respects_cap() -> None:
    assert len(build_cases(_KB, "a", max_cases=1)) == 1
    assert build_cases(_KB, "a", max_cases=0) == []


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


def test_run_eval_classifies_pass_and_fail() -> None:
    cases = [
        EvalQuestion("q1", "capsule"),
        EvalQuestion("q2", "capsule"),
        EvalQuestion("q3", "golden", expect_tables=["orders"]),
    ]

    def generate(question: str) -> str | None:
        return {
            "q1": "SELECT * FROM orders",  # ok
            "q2": "SELECT * FROM nope",  # unknown table
            "q3": "SELECT * FROM customers",  # missing expected 'orders'
        }[question]

    outcomes = run_eval(_KB, cases, generate)
    assert isinstance(outcomes[0], CaseOutcome)
    assert [o.ok for o in outcomes] == [True, False, False]
    assert "nope" in outcomes[1].reason
    assert "orders" in outcomes[2].reason


def test_run_eval_treats_generation_error_as_failure() -> None:
    def boom(question: str) -> str | None:
        raise RuntimeError("llm exploded")

    outcomes = run_eval(_KB, [EvalQuestion("q", "capsule")], boom)
    assert outcomes[0].ok is False
    assert outcomes[0].sql is None
