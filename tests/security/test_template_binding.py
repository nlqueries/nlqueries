"""A cached template may only ever receive values, never syntax.

FINDING-001 of the September 2026 external review reported SQL injection through
Tier 2 cache template binding. It is not exploitable as described, for three
independent reasons -- the entity regexes kept the question's quote characters,
the stored template already wrapped its placeholder in quotes, and `put()`
parameterises with `skip_string_literals=True` so no stored template contains a
VARCHAR placeholder to bind into at all.

None of that was a defence. It was two bugs cancelling out plus a third
coincidence, and the obvious fix for the functional bug underneath (stop keeping
the quote characters) would have made the reported injection real. The binder was
rebuilt to bind through the sqlglot AST instead of `str.replace`, and these tests
are what hold it there.

They are written against the payloads from the review, and they assert the
property rather than the mechanism: whatever a question contains, the bound
statement must have the same *shape* as its template and differ only in the
values of literals. That is what makes it impossible for a value to become a
`UNION`, a comment, or a second statement, regardless of how any future edit
decides to quote things.
"""

from __future__ import annotations

import pytest
import sqlglot
from nlqueries.cache.semantic_cache import (
    _bind_entities,
    _coerce_entity,
    _literal_shape,
    _sqlglot_name,
)
from sqlglot import exp

pytestmark = pytest.mark.security

DIALECTS = ("postgres", "mysql", "bigquery", "snowflake", "tsql")

#: The shape `put()` actually stores: the placeholder already inside quotes.
#: Testing a bare `= [region:VARCHAR]` template would be testing a shape that
#: never reaches this code, which is the mistake the original review made.
STRING_TEMPLATE = "SELECT region, amount FROM sales WHERE region = '[region:VARCHAR]'"

#: The shape a stored template actually takes for a date, which is the case that
#: reaches production: `put()` skips string literals, so DATE is the placeholder
#: type Tier 2 hits carry.
DATE_TEMPLATE = "SELECT * FROM orders WHERE d >= '[d:DATE]'"


def _one_select(sql: str, dialect: str) -> exp.Expression:
    """Parse *sql* and assert it is exactly one SELECT and nothing else."""
    statements = sqlglot.parse(sql, read=dialect)
    assert len(statements) == 1, f"{dialect}: bound SQL became {len(statements)} statements"
    tree = statements[0]
    assert isinstance(tree, exp.Select), f"{dialect}: bound SQL is a {type(tree).__name__}"
    assert not list(tree.find_all(exp.Union)), f"{dialect}: bound SQL contains a UNION"
    return tree


@pytest.mark.parametrize("dialect", DIALECTS)
def test_double_quoted_union_payload_binds_as_a_literal(dialect: str) -> None:
    """The review's payload, verbatim.

    It must survive as a string that happens to contain the words SELECT and
    UNION, compared against a column, and match nothing.
    """
    payload = "' UNION SELECT password, 1 FROM users --"
    bound = _bind_entities(f'Show me sales for "{payload}"', STRING_TEMPLATE, dialect)

    assert bound is not None, f"{dialect}: the payload should bind, inertly, not be refused"
    tree = _one_select(bound, dialect)
    literals = [node.this for node in tree.find_all(exp.Literal)]
    assert literals == [payload], f"{dialect}: expected the payload as one literal, got {literals}"


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_single_quoted_entity_cannot_carry_the_payload_at_all(dialect: str) -> None:
    """The single-quoted form of the review's payload, and why it is different.

    `_SQUOTE_RE` matches the *first* balanced pair, so in
    ``sales for 'x' UNION SELECT ...`` the entity is `x` and everything after it
    stays in the question. The injection text never reaches the SQL to be made
    inert, because it is never extracted as a value in the first place.

    Asserted because the obvious expectation -- that the whole payload binds as
    one literal, as it does in the double-quoted case -- is wrong, and a test
    written to that expectation would have to be "fixed" by someone who might
    fix it in the wrong direction.
    """
    bound = _bind_entities(
        "Show me sales for 'x' UNION SELECT password, 1 FROM users --", STRING_TEMPLATE, dialect
    )

    assert bound is not None, dialect
    tree = _one_select(bound, dialect)
    assert [node.this for node in tree.find_all(exp.Literal)] == ["x"], dialect
    assert "password" not in bound, f"{dialect}: injection text reached the SQL"


def test_backslash_payload_cannot_escape_on_mysql() -> None:
    """The case a hand-written `''`-doubling escaper gets wrong.

    On MySQL a backslash escapes the character after it inside a string, so a
    value ending in one swallows the closing quote and everything after it
    becomes SQL. Postgres does not treat it that way, which is why one rule
    cannot serve both and why rendering is left to sqlglot.
    """
    template = "SELECT * FROM t WHERE a = '[a:VARCHAR]' AND b = '[b:VARCHAR]'"
    question = 'rows where a is "' + chr(92) + '" and b is " UNION SELECT 1 -- "'

    bound = _bind_entities(question, template, "mysql")
    assert bound is not None

    tree = _one_select(bound, "mysql")
    literals = [node.this for node in tree.find_all(exp.Literal)]
    assert len(literals) == 2, f"expected two string literals, got {literals}"
    assert literals[0] == chr(92)
    assert "UNION" in literals[1], "the second value must survive as a value"


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_value_can_never_change_the_statement_shape(dialect: str) -> None:
    """The invariant, asserted directly rather than through examples.

    Every payload above is one instance of this. If a future edit changes how
    values are quoted, the examples might start passing for the wrong reason;
    this cannot.
    """
    payloads = [
        "' UNION SELECT password, 1 FROM users --",
        "'; DROP TABLE users; --",
        "') OR ('1'='1",
        chr(92) + "' OR 1=1 --",
        "*/ UNION ALL SELECT NULL /*",
    ]
    template_shape = _literal_shape(
        sqlglot.parse_one(STRING_TEMPLATE.replace("[region:VARCHAR]", "x"), read=dialect)
    )
    for payload in payloads:
        bound = _bind_entities(f'sales for "{payload}"', STRING_TEMPLATE, dialect)
        assert bound is not None, f"{dialect}: {payload!r} was refused rather than bound inertly"
        assert _literal_shape(sqlglot.parse_one(bound, read=dialect)) == template_shape, (
            f"{dialect}: {payload!r} changed the statement's shape"
        )


def test_the_structural_gate_refuses_a_shape_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is load-bearing, not decoration.

    Coercion is the first defence and the AST is the second; this is the third,
    and it is the only one that holds if the other two are wrong. Simulated by
    making the renderer emit a value as raw SQL, which is precisely what a
    regression to string substitution would do.
    """
    import nlqueries.cache.semantic_cache as sc

    real_string = exp.Literal.string

    def hostile(value: str) -> exp.Expression:
        # What `str.replace` used to do: the value arrives as SQL, not as data.
        if "UNION" in str(value):
            return sqlglot.parse_one("1 UNION SELECT password FROM users", read="postgres")
        return real_string(value)

    monkeypatch.setattr(sc.exp.Literal, "string", staticmethod(hostile))

    bound = _bind_entities(
        'sales for "x UNION SELECT password FROM users"', STRING_TEMPLATE, "postgres"
    )
    assert bound is None, "a bound statement whose shape differs from its template must be refused"


@pytest.mark.parametrize(
    ("value", "ptype", "accepted"),
    [
        ("42", "INT", True),
        ("4.2", "INT", False),
        ("' OR 1=1 --", "INT", False),
        ("4.2", "DECIMAL", True),
        ("2024-06-01", "DATE", True),
        ("2024-6-1", "DATE", False),
        ("yesterday", "DATE", False),
        ("2024-06-01 10:30", "TIMESTAMP", True),
        ("anything at all", "VARCHAR", True),
        ("x" * 513, "VARCHAR", False),
        ("nul\x00byte", "VARCHAR", False),
    ],
)
def test_type_coercion_admits_only_legal_values(value: str, ptype: str, accepted: bool) -> None:
    """A numeric placeholder can only ever receive digits, so no amount of SQL in
    a question reaches one. VARCHAR takes what a filter value legitimately
    contains and leaves the AST to make it inert."""
    assert (_coerce_entity(value, ptype) is not None) is accepted


@pytest.mark.parametrize("dialect", DIALECTS)
def test_coercion_refuses_a_value_the_binder_would_otherwise_have_taken(dialect: str) -> None:
    """End to end, and arranged so that coercion is the only thing that can refuse it.

    The obvious version of this test -- an INT placeholder and a date in the
    question -- passes whether or not coercion exists, because a date produces no
    NUMBER entity and the binder gives up earlier, at the entity-count check.
    Verified by deleting the coercion call: that version stayed green.

    `4.2` is a NUMBER entity, so it reaches an INT placeholder and is available
    to bind. Only coercion refuses it, so this fails if coercion is removed.
    """
    template = "SELECT * FROM t WHERE year = [year:INT]"
    assert _bind_entities("rows for 4.2", template, dialect) is None

    # The control: the same placeholder does bind a legal value, so the test
    # above is not passing because nothing binds here at all.
    assert _bind_entities("rows for 2024", template, dialect) == (
        "SELECT * FROM t WHERE year = 2024"
    )


@pytest.mark.parametrize("dialect", DIALECTS)
def test_bound_sql_passes_the_sql_policy(dialect: str) -> None:
    """The last gate on the replay path still accepts what binding produces, for
    a benign value. A binder that produced something the policy refuses would be
    safe and useless."""
    from nlqueries.sql_policy import evaluate

    bound = _bind_entities('sales for "East"', STRING_TEMPLATE, dialect)
    assert bound is not None
    assert evaluate(bound, dialect).allowed is True, f"{dialect}: {bound}"


#: Names a deployment actually uses, several of which sqlglot rejects outright.
#: `CAPABILITIES` keys SQL Server as `mssql`, the MCP `query` tool documents
#: `mssql` and `mysql`, and a binding may carry `postgresql` or `mariadb`.
CALLER_DIALECTS = ("mssql", "postgresql", "mariadb", "postgres", "mysql", "tsql", "bigquery")


@pytest.mark.parametrize("dialect", CALLER_DIALECTS)
def test_a_caller_s_dialect_name_still_binds(dialect: str) -> None:
    """The names that reach this code are not all sqlglot dialect names.

    `sqlglot.parse_one(..., read="mssql")` raises `ValueError: Unknown dialect`,
    and so do `postgresql` and `mariadb`. Untranslated, that error is swallowed
    and every Tier 2 hit becomes a miss on those deployments -- silently, and
    including the numeric templates that bound correctly before binding moved to
    the AST. A regression rather than a missing feature, which is why it is
    parameterised over the names callers use rather than the ones sqlglot knows.
    """
    bound = _bind_entities("orders after 2024-06-01", DATE_TEMPLATE, dialect)
    assert bound == "SELECT * FROM orders WHERE d >= '2024-06-01'", dialect


@pytest.mark.parametrize("dialect", CALLER_DIALECTS)
def test_a_caller_s_dialect_name_still_makes_a_payload_inert(dialect: str) -> None:
    """The translation must not quietly drop the dialect either: a value is only
    escaped correctly if the renderer knows which engine it is writing for."""
    payload = "' UNION SELECT password, 1 FROM users --"
    bound = _bind_entities(f'sales for "{payload}"', STRING_TEMPLATE, dialect)

    assert bound is not None, dialect
    tree = _one_select(bound, _sqlglot_name(dialect) or dialect)
    assert [node.this for node in tree.find_all(exp.Literal)] == [payload], dialect


def test_an_unreplaced_sentinel_is_refused_rather_than_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate the shape check cannot provide.

    Blanking every literal makes an unreplaced `'__nlq_bind_0__'` look exactly
    like a bound value, so such a statement would pass the shape gate, pass the
    re-parse, pass the SQL policy and run -- comparing a column against the
    string `__nlq_bind_0__` and returning nothing, which reads as "no matching
    rows" rather than as a fault.

    Simulated by making the replacement a no-op, which is what a dialect that
    parsed a sentinel as something other than a literal would produce.
    """
    import nlqueries.cache.semantic_cache as sc

    monkeypatch.setattr(sc.exp.Expression, "replace", lambda self, other: self)

    assert _bind_entities('sales for "East"', STRING_TEMPLATE, "postgres") is None
