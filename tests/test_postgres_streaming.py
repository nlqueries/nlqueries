"""Which statements the Postgres connector streams through a server-side cursor.

A read that is not streamed is loaded whole by psycopg2 before the result
budget sees a single row, so a large one can exhaust the process. The check
looked at the first word of the raw text, and the SQL console prefixes every
statement with its cancellation tag (``/* nlq-console:<id> */``): not one
console query was streamed.

No database here; the connector's own tests show a streamed read on a real
Postgres.
"""

from __future__ import annotations

import pytest
from nlqueries.connectors.postgres import _returns_rows

#: (statement, streamed)
_STATEMENTS: list[tuple[str, bool]] = [
    ("SELECT a FROM t", True),
    ("  select a from t", True),
    ("WITH x AS (SELECT 1) SELECT * FROM x", True),
    ("TABLE t", True),
    ("VALUES (1), (2)", True),
    ("(SELECT 1) UNION (SELECT 2)", True),
    # Leading comments say nothing about what a statement does.
    ("/* nlq-console:2a2e3031-f2d4-4d30-841b-1c355affb0dd */ SELECT * FROM t", True),
    ("/* one */ /* two */ SELECT a FROM t", True),
    ("/* spans\nlines */ SELECT a FROM t", True),
    ("-- a note\nSELECT a FROM t", True),
    ("-- one\n  -- two\n( SELECT 1 )", True),
    ("/** stars **/ SELECT 1", True),
    # Postgres nests block comments: the first */ does not end this one.
    ("/* outer /* inner */ still a comment */ SELECT 1", True),
    ("/* /* */ SELECT */ DELETE FROM t", False),
    # Not reads.
    ("INSERT INTO t VALUES (1)", False),
    ("/* nlq-console:abc */ INSERT INTO t VALUES (1)", False),
    ("-- SELECT\nDELETE FROM t", False),
    ("UPDATE t SET a = 1", False),
    ("CREATE TABLE t (a int)", False),
    ("EXPLAIN SELECT 1", False),
    # A word that merely begins with a keyword is not the keyword.
    ("withdraw_funds()", False),
    ("selectivity_check()", False),
    ("values_of_t", False),
    # Nothing after the comments.
    ("/* only a comment */", False),
    ("/* never closed SELECT 1", False),
    ("-- SELECT 1", False),
    ("", False),
]


@pytest.mark.parametrize(("statement", "streamed"), _STATEMENTS)
def test_reads_are_streamed_whatever_comments_lead_them(statement: str, streamed: bool) -> None:
    assert _returns_rows(statement) is streamed
