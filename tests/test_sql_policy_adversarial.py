"""
Attempts to get something past the SQL policy (W4-5).

``tests/test_sql_policy.py`` asserts the rules. This file attempts to defeat
them: obfuscations of a payload, functions that might be modelled rather than
anonymous, and inputs designed to make the evaluator raise rather than decide.

Two results here are worth stating rather than only asserting.

The obfuscations do not work, and the reason is structural: the policy inspects
the parsed tree, so case, comments, whitespace, quoting and nesting are all
normalised away before any rule is applied. A check performed on the text would
have to defeat each of them separately.

Of 25 functions that read files, sleep, take locks, mutate state or reach the
network, 23 fall back to ``exp.Anonymous`` and are refused. The two that do not
are recorded in :data:`TYPED_AND_ALLOWED` with the reason each is acceptable.
"""

from __future__ import annotations

import pytest
import sqlglot
from nlqueries.sql_policy import _FORBIDDEN_NODES, evaluate
from sqlglot import exp

_READ_FILE = "pg_read_file('/etc/hostname')"

#: The same payload, written sixteen ways.
OBFUSCATIONS: tuple[tuple[str, str], ...] = (
    ("baseline", f"SELECT {_READ_FILE}"),
    ("upper case", "SELECT PG_READ_FILE('/etc/hostname')"),
    ("mixed case", "SELECT Pg_ReAd_FiLe('/etc/hostname')"),
    ("schema qualified", "SELECT pg_catalog.pg_read_file('/etc/hostname')"),
    ("quoted identifier", "SELECT \"pg_read_file\"('/etc/hostname')"),
    ("comment between name and parens", "SELECT pg_read_file/**/('/etc/hostname')"),
    ("newline between name and parens", "SELECT pg_read_file\n('/etc/hostname')"),
    ("nested in a subquery", f"SELECT * FROM (SELECT {_READ_FILE} AS x) t"),
    ("nested in a CTE", f"WITH c AS (SELECT {_READ_FILE} AS x) SELECT * FROM c"),
    ("inside a CASE", f"SELECT CASE WHEN true THEN {_READ_FILE} ELSE '' END"),
    ("inside a WHERE", f"SELECT 1 FROM t WHERE {_READ_FILE} = 'x'"),
    ("inside an ORDER BY", f"SELECT 1 FROM t ORDER BY {_READ_FILE}"),
    ("in one arm of a UNION", f"SELECT 1 UNION SELECT length({_READ_FILE})"),
    ("as a table function", f"SELECT * FROM {_READ_FILE}"),
    ("behind a leading comment", "/* hi */ SELECT pg_sleep(30)"),
    ("with a trailing semicolon", "SELECT pg_sleep(30);"),
)

#: Functions that read files, sleep, take locks, mutate state or reach the
#: network. Each must be refused.
DANGEROUS_CALLS: tuple[str, ...] = (
    "pg_sleep(10)",
    "pg_read_file('/e')",
    "pg_read_binary_file('/e')",
    "pg_ls_dir('/')",
    "pg_stat_file('/e')",
    "lo_import('/e')",
    "lo_export(1,'/e')",
    "pg_terminate_backend(1)",
    "pg_cancel_backend(1)",
    "pg_reload_conf()",
    "pg_advisory_lock(1)",
    "pg_advisory_xact_lock(1)",
    "nextval('s')",
    "setval('s',1)",
    "currval('s')",
    "dblink('','')",
    "query_to_xml('SELECT 1',true,true,'')",
    "pg_logical_emit_message(true,'a','b')",
    "set_config('x','y',false)",
    "current_setting('x')",
    "txid_current()",
    "pg_backend_pid()",
    "clock_timestamp()",
)

#: Functions sqlglot models as typed classes, so the Anonymous signal does not
#: fire, and which are permitted. Recorded rather than left implicit: each is a
#: decision, and a reader should be able to see that it was made.
TYPED_AND_ALLOWED: dict[str, str] = {
    "random()": "non-deterministic, but reads nothing and changes nothing",
    "version()": (
        "returns the server version string. An information disclosure rather "
        "than an action, and less than the connector's own identity check "
        "already reports to an operator."
    ),
}

#: An unpaired surrogate. Written with chr() rather than as a literal: a
#: source-file emoji is a properly paired character and encodes cleanly, which
#: is the opposite of the case being tested.
LONE_SURROGATE = chr(0xD83D)

#: Inputs intended to make the evaluator raise rather than return a decision.
MALFORMED: tuple[str, ...] = (
    "",
    "   ",
    "\n\n\n",
    "\t",
    "\x00",
    "SELECT \x00",
    "--",
    "/*",
    "/* unterminated",
    "'",
    '"',
    ";;;;;",
    "SELECT 1;;SELECT 2",
    "😀 SELECT 1",
    LONE_SURROGATE + " SELECT 1",
    LONE_SURROGATE,
    "SELECT '" + "a" * 50_000 + "'",
    "SELECT " + "1+" * 5_000 + "1",
    "SELECT " + ",".join(str(i) for i in range(5_000)),
    "SELECT * FROM " + "t," * 2_000 + "t",
    "SELECT " + "(" * 500 + "1" + ")" * 500,
)


@pytest.mark.parametrize(("label", "sql"), OBFUSCATIONS, ids=[o[0] for o in OBFUSCATIONS])
def test_no_obfuscation_of_a_payload_is_allowed(label: str, sql: str) -> None:
    """The policy reads the parsed tree, so these normalise to the same thing."""
    assert not evaluate(sql, "postgres").allowed, label


@pytest.mark.parametrize("call", DANGEROUS_CALLS, ids=lambda c: c.split("(")[0])
def test_dangerous_functions_are_refused(call: str) -> None:
    decision = evaluate(f"SELECT {call}", "postgres")

    assert not decision.allowed
    assert "unrecognised function" in decision.summary()


@pytest.mark.parametrize("call", sorted(TYPED_AND_ALLOWED), ids=lambda c: c.split("(")[0])
def test_the_recorded_typed_functions_are_still_typed(call: str) -> None:
    """If sqlglot stops modelling one of these, it becomes anonymous and is
    refused. That is a safe direction, but the note here would be wrong, so
    this fails and asks for the note to be removed."""
    tree = sqlglot.parse_one(f"SELECT {call}", read="postgres")

    assert not list(tree.find_all(exp.Anonymous)), call
    assert evaluate(f"SELECT {call}", "postgres").allowed


@pytest.mark.parametrize("sql", MALFORMED, ids=lambda s: repr(s[:24]))
def test_malformed_input_produces_a_decision_rather_than_an_exception(sql: str) -> None:
    """Every caller treats a refusal as an answer. An exception here reaches
    the user as a 500, which is the outcome these gates exist to prevent.

    Three of these have found real defects: prose raises TokenError, which is a
    sibling of ParseError rather than a subclass; deep nesting exhausts the
    parser's stack with RecursionError, which is not a SqlglotError at all; and
    an unpaired surrogate raises UnicodeEncodeError from the byte-length check
    before any parsing happens.
    """
    decision = evaluate(sql, "postgres")

    assert isinstance(decision.allowed, bool)


def test_an_unpaired_surrogate_is_refused() -> None:
    """A Python string can hold one -- decoding JSON produces one from a lone
    ``\\ud83d`` escape -- and encoding it raises rather than returning bytes.

    A properly paired emoji is the other case, and a different one: it encodes
    without complaint and is refused by the parser instead. Writing this test
    with a source-file emoji asserted the wrong thing.
    """
    decision = evaluate(LONE_SURROGATE + " SELECT 1", "postgres")

    assert not decision.allowed
    assert "not valid text" in decision.summary()

    paired = evaluate("😀 SELECT 1", "postgres")

    assert not paired.allowed
    assert "could not be parsed" in paired.summary()


@pytest.mark.parametrize("sql", MALFORMED, ids=lambda s: repr(s[:24]))
def test_anything_allowed_satisfies_the_policy_s_own_invariant(sql: str) -> None:
    """Self-consistency: a statement the policy allows must parse to exactly one
    query containing none of the forbidden nodes.

    This catches a rule that is checked but not enforced -- the shape the depth
    cap had before it was moved ahead of parsing.
    """
    decision = evaluate(sql, "postgres")
    if not decision.allowed:
        return

    statements = sqlglot.parse(sql, read="postgres")
    parsed = [s for s in statements if isinstance(s, exp.Expression)]

    assert len(parsed) == 1
    assert isinstance(parsed[0], exp.Select | exp.Union)
    assert not list(parsed[0].find_all(*_FORBIDDEN_NODES))
