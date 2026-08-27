"""
The SQL policy: what it refuses, and what it must not.

Rules established by measurement against sqlglot 30.9 and PostgreSQL 16. The
corpus tests in ``tests/security/test_payload_corpus.py`` assert the same
payloads against a real database; these assert the policy's verdict without one.
"""

from __future__ import annotations

import pytest
from nlqueries.sql_policy import MAX_BYTES, MAX_DEPTH, POLICY_VERSION, evaluate

from tests.security.payloads import POSTGRES, SAFE_POSTGRES


@pytest.mark.parametrize("payload", POSTGRES, ids=lambda p: p.id)
def test_every_audit_payload_is_refused(payload) -> None:
    """Each of these is a syntactically valid SELECT, which is what the gates
    this policy replaces checked for."""
    decision = evaluate(payload.sql, "postgres")

    assert not decision.allowed, f"{payload.id} was allowed: {payload.effect}"
    assert decision.reasons


@pytest.mark.parametrize("sql", SAFE_POSTGRES, ids=lambda s: s[:38])
def test_ordinary_analytics_are_allowed(sql: str) -> None:
    """The control. A policy that refuses the payloads and also refuses a
    GROUP BY has made the product unusable rather than safer."""
    decision = evaluate(sql, "postgres")

    assert decision.allowed, decision.summary()


class TestRules:
    def test_a_second_statement_is_refused(self) -> None:
        """`parse_one` returns the first statement and discards the rest, so
        the policy uses `parse` and counts what it gets."""
        decision = evaluate("SELECT 1 /* comment */ ; DROP TABLE orders", "postgres")

        assert not decision.allowed
        assert "exactly one statement" in decision.summary()

    @pytest.mark.parametrize(
        "sql",
        [
            "VACUUM orders",
            "CALL some_procedure()",
            "DO $$ BEGIN PERFORM 1; END $$",
            "COPY orders TO '/tmp/x.csv'",
            "SET search_path = evil",
            "GRANT SELECT ON orders TO public",
            "TRUNCATE orders",
        ],
    )
    def test_non_query_statements_are_refused(self, sql: str) -> None:
        """Allowed root types are listed rather than denied: these parse to
        Command, Copy, Set, Grant and TruncateTable respectively, and a
        denylist would need to name each one."""
        assert not evaluate(sql, "postgres").allowed

    def test_dml_inside_a_cte_is_refused(self) -> None:
        """The root is a Select. The Insert is two levels down."""
        sql = "WITH w AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM w"

        decision = evaluate(sql, "postgres")

        assert not decision.allowed
        assert "Insert" in decision.summary()

    def test_prose_is_refused(self) -> None:
        """A model that answers in words rather than SQL raises TokenError,
        which is a sibling of ParseError rather than a subclass."""
        decision = evaluate("I'm sorry, I cannot answer that question.", "postgres")

        assert not decision.allowed
        assert "could not be parsed" in decision.summary()

    def test_an_oversized_statement_is_refused_before_parsing(self) -> None:
        assert not evaluate("x" * (MAX_BYTES + 1), "postgres").allowed

    def test_deep_nesting_is_refused_before_parsing(self) -> None:
        """sqlglot's parser is recursive and exhausts the stack on input this
        deep, raising RecursionError, which is not a SqlglotError. A check on
        the parsed tree would never run, so the depth is counted from the text.
        """
        sql = "SELECT " + "(" * (MAX_DEPTH + 50) + "1" + ")" * (MAX_DEPTH + 50)

        decision = evaluate(sql, "postgres")

        assert not decision.allowed
        assert "nested deeper" in decision.summary()

    def test_nesting_within_the_cap_is_allowed(self) -> None:
        assert evaluate("SELECT " + "(" * 20 + "1" + ")" * 20, "postgres").allowed


class TestTheAllowlist:
    def test_an_allowlisted_function_is_permitted(self) -> None:
        assert evaluate("SELECT age(now(), created_at) FROM t", "postgres").allowed

    def test_an_unknown_function_is_refused_and_named(self) -> None:
        decision = evaluate("SELECT custom_score(total) FROM t", "postgres")

        assert not decision.allowed
        assert "custom_score" in decision.summary()

    def test_the_allowlist_is_per_dialect(self) -> None:
        """`age` is a PostgreSQL function. A dialect that does not list it does
        not inherit it."""
        assert evaluate("SELECT age(a, b) FROM t", "postgres").allowed
        assert not evaluate("SELECT age(a, b) FROM t", "snowflake").allowed


class TestDialectNames:
    def test_every_registered_connector_name_is_usable(self) -> None:
        """The policy is called with the connector's own dialect name, which is
        not always sqlglot's. Measured on sqlglot 30.9: `mssql` is `tsql`
        there, and `sqlalchemy` is not a dialect at all.
        """
        from nlqueries.connectors import CONNECTOR_REGISTRY

        unusable = [
            name
            for name in CONNECTOR_REGISTRY
            if name != "sqlalchemy" and not evaluate("SELECT count(*) FROM orders", name).allowed
        ]

        assert not unusable, f"no grammar resolved for: {', '.join(sorted(unusable))}"

    def test_an_unknown_dialect_is_refused_rather_than_raised(self) -> None:
        """sqlglot raises a plain ValueError for a dialect it does not know,
        which is not a SqlglotError. Parsing with a different grammar would
        mean the statement was not checked.
        """
        decision = evaluate("SELECT 1", "not-a-dialect")

        assert not decision.allowed
        assert "no grammar available" in decision.summary()

    def test_the_generic_connector_must_supply_a_real_dialect(self) -> None:
        """`sqlalchemy` reaches many engines and identifies no single grammar,
        so a statement for it is refused until the caller supplies one."""
        assert not evaluate("SELECT 1", "sqlalchemy").allowed

    def test_anonymous_functions_are_reported_even_when_allowed(self) -> None:
        """The inventory needs to see them to report which allowlist entries
        are still used."""
        decision = evaluate("SELECT age(a, b) FROM t", "postgres")

        assert decision.allowed
        assert "age" in decision.anonymous_functions


def test_every_decision_records_the_policy_version() -> None:
    """A stored verdict has to be distinguishable from a current one."""
    for sql in ("SELECT 1", "DROP TABLE t"):
        assert evaluate(sql, "postgres").policy_version == POLICY_VERSION


class TestTheGatesUseThePolicy:
    """The policy is only useful where the query path consults it."""

    def test_the_cache_replay_gate_refuses_every_payload(self, caplog) -> None:
        """A cache hit reaches the database with a stored statement and no
        model in front of it, so this is the only check on that path."""
        from nlqueries.orchestrator.multi_agent_orchestrator import _is_executable_select

        allowed = [p.id for p in POSTGRES if _is_executable_select(p.sql, "postgres")]

        assert not allowed, f"cache replay would run: {', '.join(allowed)}"

    def test_the_cache_replay_gate_allows_ordinary_analytics(self) -> None:
        from nlqueries.orchestrator.multi_agent_orchestrator import _is_executable_select

        refused = [sql for sql in SAFE_POSTGRES if not _is_executable_select(sql, "postgres")]

        assert not refused, f"cache replay would refuse: {refused}"

    def test_the_validator_refuses_a_second_statement(self) -> None:
        """`_validate_sql` used `parse_one`, which returns the first statement
        and discards the rest, so this passed validation before."""
        from nlqueries.orchestrator.sql_generation import _validate_sql

        kb = {"schema": {"tables": [{"name": "orders"}]}}
        error = _validate_sql("SELECT 1 /* c */ ; DROP TABLE orders", kb, "postgres")

        assert error is not None
        assert "one statement" in error

    def test_the_validator_refuses_dml_in_a_cte(self) -> None:
        """The root is a Select, which is all the previous check asked."""
        from nlqueries.orchestrator.sql_generation import _validate_sql

        kb = {"schema": {"tables": [{"name": "orders"}]}}
        sql = "WITH w AS (INSERT INTO orders VALUES (1) RETURNING *) SELECT * FROM w"

        error = _validate_sql(sql, kb, "postgres")

        assert error is not None
        assert "Insert" in error


class TestDialectFromUrl:
    """Dialect resolution for the generic SQLAlchemy connector.

    Its `db_type` is `sqlalchemy` for every engine it reaches, so the dialect is
    taken from the URL. Backend names measured against SQLAlchemy 2.0 and
    checked against sqlglot 30.9.
    """

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("postgresql+psycopg2://u:p@h/db", "postgres"),
            ("postgresql://u:p@h/db", "postgres"),
            ("mysql+pymysql://u:p@h/db", "mysql"),
            ("mariadb+pymysql://u:p@h/db", "mysql"),
            ("mssql+pyodbc://u:p@h/db", "tsql"),
            ("sqlite:///x.db", "sqlite"),
            ("snowflake://u:p@acct/db", "snowflake"),
            ("bigquery://project/dataset", "bigquery"),
            ("duckdb:///x.duckdb", "duckdb"),
        ],
    )
    def test_known_backends_resolve(self, url: str, expected: str) -> None:
        from nlqueries.sql_policy import dialect_from_url

        assert dialect_from_url(url) == expected

    @pytest.mark.parametrize("url", ["not a url at all", "", "://"])
    def test_an_unreadable_url_resolves_to_nothing(self, url: str) -> None:
        """Returns None so the caller decides. A statement checked against a
        different grammar is not checked."""
        from nlqueries.sql_policy import dialect_from_url

        assert dialect_from_url(url) is None

    def test_every_resolved_dialect_is_one_sqlglot_accepts(self) -> None:
        """The mapping exists because three backend names are not sqlglot
        dialects. This fails if a mapping stops being correct."""
        from nlqueries.sql_policy import dialect_from_url

        for url in (
            "postgresql://u:p@h/db",
            "mysql://u:p@h/db",
            "mariadb://u:p@h/db",
            "mssql://u:p@h/db",
            "sqlite:///x.db",
        ):
            dialect = dialect_from_url(url)
            assert dialect is not None
            assert evaluate("SELECT 1", dialect).allowed, dialect


def test_a_refused_root_is_not_also_reported_as_contained() -> None:
    """`DROP TABLE t` is a Drop at the root. Reporting it as both "is a Drop"
    and "contains Drop" states one fact twice."""
    summary = evaluate("DROP TABLE t", "postgres").summary()

    assert summary == "statement is a Drop, not a query"
