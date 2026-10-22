"""Only listed constructs are allowed, per dialect, and the rendered form must read the same.

The corpus in ``tests/sql_corpus.py`` is what the lists must keep allowing;
the rest of this file is what they must not.
"""

from __future__ import annotations

import pytest
from nlqueries import sql_policy
from nlqueries.sql_constructs import (
    DIALECT_STRUCTURE,
    allowed_constructs,
    unresolved_names,
)
from nlqueries.sql_policy import POLICY_VERSION, evaluate
from nlqueries.sql_policy_report import build_inventory

from tests.sql_corpus import CORPUS

_CASES = [(dialect, sql) for dialect, queries in CORPUS.items() for sql in queries]


@pytest.mark.parametrize(
    ("dialect", "sql"), _CASES, ids=[f"{d}-{n}" for n, (d, _) in enumerate(_CASES)]
)
def test_every_corpus_query_is_allowed(dialect: str, sql: str) -> None:
    decision = evaluate(sql, dialect)
    assert decision.allowed, decision.summary()


def test_the_corpus_covers_every_dialect_with_a_list() -> None:
    assert set(CORPUS) == set(DIALECT_STRUCTURE)


def test_every_listed_name_is_a_class_in_this_sqlglot() -> None:
    """A misspelt or renamed entry would quietly allow nothing."""
    assert unresolved_names() == []


def test_the_version_records_the_change() -> None:
    assert POLICY_VERSION == "4"


#: Calls that read files, sleep, lock, reach the network, run commands, or read
#: the server's settings, per dialect. Each must be refused.
_DANGEROUS: dict[str, list[str]] = {
    "postgres": [
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_sleep(10)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT * FROM dblink('host=x', 'SELECT 1') AS t(a int)",
        "SELECT set_config('search_path', 'x', false)",
        "SELECT pg_terminate_backend(1)",
        "SELECT current_user",
        "SELECT $1",
    ],
    "redshift": ["SELECT pg_read_file('/etc/passwd')", "SELECT current_setting('x')"],
    "mysql": [
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT SLEEP(10)",
        "SELECT BENCHMARK(1000000, MD5('a'))",
        "SELECT GET_LOCK('a', 10)",
        "SELECT @@datadir",
    ],
    "tsql": [
        "SELECT * FROM OPENROWSET('SQLNCLI', 'Server=x;', 'SELECT 1')",
        "SELECT * FROM OPENDATASOURCE('SQLNCLI', 'Data Source=x').db.dbo.t",
        "SELECT @@version",
        "SELECT @x",
    ],
    "snowflake": ["SELECT SYSTEM$WAIT(10)", "SELECT CURRENT_USER()"],
    "bigquery": ["SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')", "SELECT SESSION_USER()"],
    "duckdb": [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM glob('/*')",
        "SELECT current_setting('home_directory')",
        "SELECT getenv('HOME')",
    ],
    "sqlite": [
        "SELECT load_extension('x')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/x', 'y')",
    ],
}


@pytest.mark.parametrize(
    ("dialect", "sql"),
    [(d, s) for d, calls in _DANGEROUS.items() for s in calls],
    ids=lambda v: v if len(v) < 12 else v[7:40],
)
def test_reading_files_sleeping_and_administration_are_refused(dialect: str, sql: str) -> None:
    assert not evaluate(sql, dialect).allowed


@pytest.mark.parametrize(
    ("dialect", "sql", "construct"),
    [
        ("tsql", "SELECT @@version", "Parameter"),
        ("mysql", "SELECT @@datadir", "SessionParameter"),
        ("postgres", "SELECT ?", "Placeholder"),
        ("postgres", "SELECT current_user", "CurrentUser"),
    ],
)
def test_what_a_read_query_has_no_use_for_is_refused_by_name(
    dialect: str, sql: str, construct: str
) -> None:
    """These parse as classes sqlglot models, which the old rule let through."""
    decision = evaluate(sql, dialect)
    assert not decision.allowed
    assert construct in decision.unlisted_constructs
    assert f"not on the {sql_policy._sqlglot_dialect(dialect)} list" in decision.summary()


def test_one_engines_clause_is_not_allowed_in_anothers_statement() -> None:
    qualify = "SELECT a FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY a) = 1"
    assert evaluate(qualify, "snowflake").allowed
    decision = evaluate(qualify, "postgres")
    assert not decision.allowed
    assert "Qualify" in decision.unlisted_constructs


def test_a_subclass_of_a_listed_class_is_not_allowed_by_inheriting() -> None:
    """Hive's ``SORT BY`` parses as ``Sort``, a subclass of the listed ``Order``."""
    decision = evaluate("SELECT a FROM t SORT BY a", "spark")
    assert not decision.allowed
    assert decision.unlisted_constructs == ("Sort",)


def test_a_class_sqlglot_does_not_export_is_refused_not_raised() -> None:
    """A node whose class is not on `sqlglot.expressions` under its own name, as
    `_TimeUnit` is not: named in the refusal, not an AttributeError."""
    import sqlglot
    from sqlglot import exp

    class _Unexported(exp.Expression):
        pass

    sql = "SELECT a FROM t"
    tree = sqlglot.parse_one(sql, read="postgres")
    tree.find(exp.Column).replace(_Unexported())
    assert not hasattr(exp, "_Unexported")

    reading = sql_policy._read(tree, sql, "postgres")

    assert reading.unlisted == ("_Unexported",)


def test_an_unmodelled_aggregate_is_judged_by_name() -> None:
    """ClickHouse's combinators parse as an unmodelled aggregate class that the
    old rule, which looked only for `Anonymous`, never saw."""
    decision = evaluate("SELECT sumIf(x, y > 0) FROM t", "clickhouse")
    assert "sumif" in decision.anonymous_functions
    assert not decision.allowed


def test_the_lists_are_per_dialect_and_a_dialect_without_one_gets_the_shared_set() -> None:
    from sqlglot import exp

    assert exp.Qualify in allowed_constructs("bigquery")
    assert exp.Qualify not in allowed_constructs("sqlite")
    assert exp.Qualify not in allowed_constructs("oracle")
    assert exp.Select in allowed_constructs("oracle")


# --- the rendered form ------------------------------------------------------------


def test_canonicalising_is_not_a_difference() -> None:
    """sqlglot renders MySQL's IF as CASE and T-SQL's JSON_VALUE as an ISNULL of two
    calls; the policy reads both the same, so neither is refused."""
    assert evaluate("SELECT IF(total > 50, 'large', 'small') FROM orders", "mysql").allowed
    assert evaluate("SELECT JSON_VALUE(profile, '$.tier') FROM customers", "tsql").allowed


@pytest.mark.parametrize(
    ("rendered", "reason"),
    [
        ("SELECT a FROM secrets", "reads other tables"),
        ("SELECT a FROM t; SELECT 1", "parses as 2 statements"),
        ("SELECT pg_sleep(1) FROM t", "rendered form is refused"),
        ("SELECT (", "could not be parsed again"),
        ("SELECT a FROM t UNION SELECT a FROM t", "parses as a Union"),
    ],
)
def test_a_statement_whose_rendering_reads_differently_is_refused(
    monkeypatch, rendered: str, reason: str
) -> None:
    """A parser and generator that disagree, constructed: the statement reads
    `t` and calls nothing, and its rendering says otherwise."""
    sql = "SELECT a FROM t"
    assert evaluate(sql, "postgres").allowed
    monkeypatch.setattr(sql_policy, "_rendered", lambda tree, dialect: rendered)

    decision = evaluate(sql, "postgres")

    assert not decision.allowed
    assert reason in decision.summary()


def test_the_rendered_form_is_not_consulted_for_a_statement_already_refused(monkeypatch) -> None:
    def fail(tree, dialect):
        raise AssertionError("rendered")

    monkeypatch.setattr(sql_policy, "_rendered", fail)
    assert not evaluate("SELECT pg_sleep(1)", "postgres").allowed


# --- the report -------------------------------------------------------------------


def test_the_report_counts_constructs_not_on_the_list() -> None:
    report = build_inventory(["SELECT @@version", "SELECT @@version", "SELECT 1"], "tsql")
    assert report.unlisted_counts["Parameter"] == 2
    assert report.refused == 2
    rendered = report.render()
    assert "Constructs not on the list" in rendered
    assert "Parameter" in rendered
