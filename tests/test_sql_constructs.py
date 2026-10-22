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
    assert POLICY_VERSION == "5"


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
        "SELECT SUM(total) FROM orders WITH (TABLOCKX)",
        "SELECT SUM(total) FROM orders WITH (UPDLOCK, HOLDLOCK)",
        # The same hints without WITH, which SQL Server still honours.
        "SELECT SUM(total) FROM orders o (TABLOCKX)",
        "SELECT SUM(o.total) FROM orders AS o (UPDLOCK, HOLDLOCK)",
        "SELECT SUM(total) FROM orders (UPDLOCK)",
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


def test_a_tsql_derived_table_may_still_name_its_columns() -> None:
    """Only a table's alias is read as a hint; a subquery's column list is not one."""
    import sqlglot
    from sqlglot import exp

    assert evaluate("SELECT v.x FROM (SELECT 1 AS a) AS v(x)", "tsql").allowed
    sql = "SELECT SUM(total) FROM orders o (TABLOCKX)"
    refused = evaluate(sql, "tsql")
    assert not refused.allowed
    # Refused whichever way sqlglot reads it. Later releases parse the hint as
    # a WithTableHint, which is not listed; earlier ones, as a column list on
    # the alias, which is what the T-SQL rule exists for.
    as_column_list = any(
        (alias := table.args.get("alias")) is not None and alias.args.get("columns")
        for table in sqlglot.parse_one(sql, read="tsql").find_all(exp.Table)
    )
    if as_column_list:
        assert "SQL Server reads as a table hint" in refused.summary()
    else:
        assert "WithTableHint" in refused.unlisted_constructs
    # Elsewhere a column list on a table's alias is an ordinary rename.
    assert evaluate("SELECT o.a FROM orders AS o(a, b)", "postgres").allowed


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


def test_a_statement_too_deep_to_render_is_refused_not_raised() -> None:
    """The parser builds a cast chain without recursing; the generator recurses."""
    sql = "SELECT a" + "::INT" * 1000 + " FROM t"
    decision = evaluate(sql, "postgres")
    assert not decision.allowed
    assert "could not be produced or parsed again" in decision.summary()


def test_a_root_that_subclasses_select_is_still_checked_by_class() -> None:
    """It passes the root rule's isinstance; the exact-class rule must catch it."""
    import sqlglot
    from sqlglot import exp

    class _SelectSomehow(exp.Select):
        pass

    sql = "SELECT a FROM t"
    tree = _SelectSomehow(**sqlglot.parse_one(sql, read="postgres").args)

    reading = sql_policy._read(tree, sql, "postgres")

    assert reading.unlisted == ("_SelectSomehow",)


def test_set_operations_are_allowed_below_the_root_and_refused_at_it() -> None:
    inner = (
        "WITH kept AS (SELECT a FROM t EXCEPT SELECT a FROM u) SELECT * FROM kept",
        "WITH kept AS (SELECT a FROM t INTERSECT SELECT a FROM u) SELECT * FROM kept",
    )
    for sql in inner:
        assert evaluate(sql, "postgres").allowed
    root = evaluate("SELECT a FROM t EXCEPT SELECT a FROM u", "postgres")
    assert not root.allowed
    assert "not a query" in root.summary()


def test_an_unmodelled_aggregate_is_judged_by_name() -> None:
    """ClickHouse's combinators parse as an unmodelled aggregate class that the
    old rule, which looked only for `Anonymous`, never saw."""
    decision = evaluate("SELECT sumIf(x, y > 0) FROM t", "clickhouse")
    assert "sumif" in decision.anonymous_functions
    assert not decision.allowed


@pytest.mark.parametrize(
    ("dialect", "sql", "bare", "qualified"),
    [
        ("postgres", "SELECT some_schema.age(a, b) FROM t", "age", "some_schema.age"),
        ("mysql", "SELECT x.now() FROM t", "now", "x.now"),
        ("duckdb", "SELECT main.sch.age(a) FROM t", "age", "main.sch.age"),
    ],
)
def test_a_qualified_call_is_judged_by_its_whole_name(
    dialect: str, sql: str, bare: str, qualified: str
) -> None:
    """The bare name is listed; the function that runs is whatever the qualifier
    finds, which the entry does not vouch for."""
    from nlqueries.sql_policy import allowed_anonymous

    assert bare in allowed_anonymous(dialect)
    decision = evaluate(sql, dialect)
    assert not decision.allowed
    assert decision.anonymous_functions == (qualified,)
    assert f"calls unrecognised function(s): {qualified}" in decision.summary()


@pytest.mark.parametrize(
    ("dialect", "sql", "qualified"),
    [
        (
            "postgres",
            "SELECT some_schema.age(a, b) FILTER (WHERE a > 0) FROM t",
            "some_schema.age",
        ),
        (
            "postgres",
            "SELECT some_schema.json_agg(a) FILTER (WHERE a IS NOT NULL) FROM t",
            "some_schema.json_agg",
        ),
        (
            "postgres",
            "SELECT some_schema.json_agg(a) FILTER (WHERE a IS NOT NULL) OVER () FROM t",
            "some_schema.json_agg",
        ),
        (
            "postgres",
            "SELECT some_schema.age(a, b) WITHIN GROUP (ORDER BY a) FROM t",
            "some_schema.age",
        ),
        ("duckdb", "SELECT main.s.age(a) FILTER (WHERE a > 0) FROM t", "main.s.age"),
        ("snowflake", "SELECT s.every(a) IGNORE NULLS FROM t", "s.every"),
    ],
)
def test_a_suffix_does_not_hide_the_qualifier(dialect: str, sql: str, qualified: str) -> None:
    """FILTER, WITHIN GROUP and IGNORE NULLS wrap the call before the qualifier
    is attached, which puts it in a Column, or a Dot around the wrapper."""
    decision = evaluate(sql, dialect)
    assert not decision.allowed
    assert qualified in decision.anonymous_functions


@pytest.mark.parametrize(
    ("dialect", "sql", "qualified"),
    [
        # In FROM the qualifier is the Table's schema and catalog.
        ("postgres", "SELECT * FROM some_schema.age(1, 2)", "some_schema.age"),
        ("postgres", "SELECT * FROM db.some_schema.age(1, 2)", "db.some_schema.age"),
        ("postgres", "SELECT * FROM t, LATERAL some_schema.age(t.a, 2) AS x", "some_schema.age"),
        # A call sqlglot models is judged by name too once qualified, spelled
        # as the dialect spells it rather than by its sqlglot class.
        (
            "postgres",
            "SELECT * FROM some_schema.generate_series(1, 3)",
            "some_schema.generate_series",
        ),
        (
            "postgres",
            "SELECT * FROM t, LATERAL some_schema.generate_series(1, t.a) AS g",
            "some_schema.generate_series",
        ),
        ("duckdb", "SELECT * FROM main.range(3)", "main.range"),
        # A Column under the Dot keeps all of its parts.
        ("postgres", "SELECT a.b.c.d.age(x) FILTER (WHERE x > 0) FROM t", "a.b.c.d.age"),
    ],
)
def test_a_qualified_call_in_any_position_is_judged_by_its_whole_name(
    dialect: str, sql: str, qualified: str
) -> None:
    decision = evaluate(sql, dialect)
    assert not decision.allowed
    assert decision.anonymous_functions == (qualified,)
    assert f"calls unrecognised function(s): {qualified}" in decision.summary()


@pytest.mark.parametrize(
    ("dialect", "sql"),
    [
        ("postgres", "SELECT * FROM generate_series(1, 3)"),
        ("postgres", "SELECT * FROM t, LATERAL generate_series(1, t.a) AS g"),
        ("duckdb", "SELECT * FROM range(3)"),
    ],
)
def test_an_unqualified_modelled_call_is_still_judged_by_its_class(dialect: str, sql: str) -> None:
    decision = evaluate(sql, dialect)
    assert decision.allowed, decision.summary()
    assert decision.anonymous_functions == ()


def test_a_qualified_call_written_without_brackets_is_named_by_its_rendering() -> None:
    """``CURRENT_DATE`` has no argument list; named, it can be listed, where
    unnamed it could only be refused. Built by hand: sqlglot 30.17 reads
    ``t.current_date`` as a column."""
    import sqlglot
    from sqlglot import exp

    sql = "SELECT t.x FROM t"
    tree = sqlglot.parse_one(sql, read="postgres")
    tree.find(exp.Column).replace(
        exp.Dot(this=exp.to_identifier("t"), expression=exp.CurrentDate())
    )

    reading = sql_policy._read(tree, sql, "postgres")

    assert reading.anonymous == ("t.current_date",)
    assert "calls a function whose name could not be read" not in reading.reasons


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM public.mod(5, 2)",
        "SELECT * FROM public.mod(length(a), 2)",
        "SELECT * FROM t, LATERAL s.mod(t.a, 2) AS m",
    ],
)
def test_a_qualifier_on_an_operator_is_refused_and_not_lent_to_its_operand(
    monkeypatch, sql: str
) -> None:
    """``mod`` here parses as the operator `Mod`. Its qualifier is not
    ``length``'s, so listing ``public.length`` does not let it through."""
    monkeypatch.setitem(
        sql_policy.ALLOWED_ANONYMOUS,
        "postgres",
        sql_policy.ALLOWED_ANONYMOUS["postgres"] | {"public.length"},
    )
    decision = evaluate(sql, "postgres")
    assert not decision.allowed
    assert "puts a qualifier on what is not a name or a function: Mod" in decision.summary()
    assert "public.length" not in decision.anonymous_functions


def test_an_unqualified_operator_is_not_said_to_carry_a_qualifier() -> None:
    decision = evaluate("SELECT * FROM mod(5, 2)", "postgres")
    assert "puts a qualifier" not in decision.summary()


def test_a_suffixed_qualified_call_listed_as_written_is_allowed(monkeypatch) -> None:
    """The suffix wraps the call, not something else the qualifier could be on."""
    monkeypatch.setitem(
        sql_policy.ALLOWED_ANONYMOUS,
        "postgres",
        sql_policy.ALLOWED_ANONYMOUS["postgres"] | {"some_schema.age"},
    )
    for sql in (
        "SELECT some_schema.age(a, b) FILTER (WHERE a > 0) FROM t",
        "SELECT some_schema.age(a, b) FILTER (WHERE a > 0) OVER () FROM t",
    ):
        decision = evaluate(sql, "postgres")
        assert decision.allowed, decision.summary()


def test_a_qualified_call_rendered_as_an_operator_takes_sqlglots_name() -> None:
    """Named by its rendering it would be ``s.a -> 'b'``, a name that changes with
    the operands. Built by hand, as no FROM spelling parses to it on 30.17."""
    import sqlglot
    from sqlglot import exp

    extract = sqlglot.parse_one("SELECT a -> 'b' FROM t", read="postgres").selects[0]
    assert isinstance(extract, exp.JSONExtract)
    sql = "SELECT * FROM x"
    tree = sqlglot.parse_one(sql, read="postgres")
    tree.find(exp.Table).replace(exp.Table(this=extract, db=exp.to_identifier("s")))

    reading = sql_policy._read(tree, sql, "postgres")

    assert reading.anonymous == ("s.json_extract",)


def test_a_qualified_modelled_call_is_allowed_when_listed_as_written(monkeypatch) -> None:
    monkeypatch.setitem(
        sql_policy.ALLOWED_ANONYMOUS,
        "postgres",
        sql_policy.ALLOWED_ANONYMOUS["postgres"] | {"pg_catalog.generate_series"},
    )
    assert evaluate("SELECT * FROM pg_catalog.generate_series(1, 3)", "postgres").allowed


def test_a_suffix_on_an_unqualified_call_changes_nothing() -> None:
    for sql in (
        "SELECT age(a, b) FILTER (WHERE a > 0) FROM t",
        "SELECT json_agg(a) FILTER (WHERE a IS NOT NULL) OVER () FROM t",
    ):
        decision = evaluate(sql, "postgres")
        assert decision.allowed, decision.summary()


def test_a_qualified_call_is_allowed_when_listed_as_written(monkeypatch) -> None:
    monkeypatch.setitem(
        sql_policy.ALLOWED_ANONYMOUS, "postgres", frozenset({"age", "some_schema.age"})
    )
    assert evaluate("SELECT Some_Schema.AGE(a, b) FROM t", "postgres").allowed


def test_a_function_whose_name_cannot_be_read_is_refused() -> None:
    """Judged by name and exempt from the class list, so with no readable name it
    would otherwise be judged by neither."""
    import sqlglot
    from sqlglot import exp

    sql = "SELECT age(a, b) FROM t"
    tree = sqlglot.parse_one(sql, read="postgres")
    tree.find(exp.Anonymous).set("this", exp.Tuple())
    assert tree.find(exp.Anonymous).name == ""

    reading = sql_policy._read(tree, sql, "postgres")

    assert "calls a function whose name could not be read" in reading.reasons
    assert reading.anonymous == ()


def test_a_qualifier_whose_name_cannot_be_read_is_refused() -> None:
    import sqlglot
    from sqlglot import exp

    sql = "SELECT some_schema.age(a, b) FROM t"
    tree = sqlglot.parse_one(sql, read="postgres")
    tree.find(exp.Dot).set("this", exp.Tuple())

    reading = sql_policy._read(tree, sql, "postgres")

    assert "calls a function whose name could not be read" in reading.reasons


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
        ("SELECT (", "could not be produced or parsed again"),
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
    """Recorded, not raised: anything raised while rendering is now a refusal."""
    rendered: list[object] = []
    monkeypatch.setattr(sql_policy, "_rendered", lambda tree, dialect: rendered.append(tree) or "")

    decision = evaluate("SELECT pg_sleep(1)", "postgres")

    assert not decision.allowed
    assert rendered == []
    assert decision.reasons == ("calls unrecognised function(s): pg_sleep",)


# --- the report -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rendered", "anonymous", "unlisted"),
    [
        ("SELECT pg_sleep(1) FROM t", ("pg_sleep",), ()),
        ("SELECT a FROM t WHERE a = ?", (), ("Placeholder",)),
        # Refused for its tables, not for what it calls: listing that would
        # change nothing, so nothing is counted.
        ("SELECT xmin_age(a) FROM secrets", (), ()),
    ],
)
def test_the_report_counts_what_only_the_rendered_form_uses(
    monkeypatch, rendered: str, anonymous: tuple[str, ...], unlisted: tuple[str, ...]
) -> None:
    """A construct that appears only once sqlglot has canonicalised the statement
    is what the refusal is for; the report names it, not only the reason."""
    sql = "SELECT a FROM t"
    monkeypatch.setattr(sql_policy, "_rendered", lambda tree, dialect: rendered)

    decision = evaluate(sql, "postgres")
    report = build_inventory([sql], "postgres")

    assert not decision.allowed
    assert decision.anonymous_functions == anonymous
    assert decision.unlisted_constructs == unlisted
    assert [name for name, _ in report.candidates] == list(anonymous)
    assert tuple(report.unlisted_counts) == unlisted


def test_an_allowed_statement_reports_only_what_it_was_written_with(monkeypatch) -> None:
    """The rendering's own spelling is not counted when it changes nothing."""
    sql = "SELECT a FROM t"
    monkeypatch.setattr(sql_policy, "_rendered", lambda tree, dialect: "SELECT age(a, a) FROM t")

    decision = evaluate(sql, "postgres")

    assert decision.allowed
    assert decision.anonymous_functions == ()


def test_the_report_counts_constructs_not_on_the_list() -> None:
    report = build_inventory(["SELECT @@version", "SELECT @@version", "SELECT 1"], "tsql")
    assert report.unlisted_counts["Parameter"] == 2
    assert report.refused == 2
    rendered = report.render()
    assert "Constructs not on the list" in rendered
    assert "Parameter" in rendered
