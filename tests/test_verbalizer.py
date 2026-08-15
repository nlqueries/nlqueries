"""
Tests for nlqueries.verbalizer (ACE-1.1) — the deterministic SQL-to-English
paraphraser.

Covers:
  - ~35 golden paraphrases locking the controlled-English style (declarative
    "Shows …"; clause order subject → from → joins → filters → grouping →
    sort/limit) across postgres/snowflake/bigquery/tsql;
  - graceful degradation: window functions, arithmetic, aggregate-of-expression,
    subqueries, CTEs, set operations, non-SELECT, and parse errors all quote the
    unhandled fragment verbatim (or yield empty text) and set complete=False;
  - the property that verbalize NEVER raises over 1000+ generated + garbage
    inputs;
  - the vocab layer (humanize, build_vocab, label fallback) and Paraphrase.to_dict.
"""

from __future__ import annotations

import itertools

import pytest
from nlqueries.verbalizer import Paraphrase, Vocab, build_vocab, humanize, verbalize

# A small business vocabulary shared by the golden cases.
VOCAB = Vocab(
    tables={
        "orders": "orders",
        "customers": "customers",
        "order_items": "line items",
        "employees": "employees",
    },
    columns={
        "orders.total": "order total",
        "orders.status": "status",
        "orders.customer_id": "customer",
        "customers.region": "region",
        "customers.name": "customer name",
        "customers.signup_date": "signup date",
        "order_items.qty": "quantity",
        "total": "order total",
        "status": "status",
        "region": "region",
    },
)


# ---------------------------------------------------------------------------
# Golden paraphrases — the locked style (postgres)
# ---------------------------------------------------------------------------

GOLDENS: list[tuple[str, str]] = [
    ("SELECT id, total FROM orders", "Shows id and order total from orders."),
    ("SELECT * FROM customers", "Shows all columns from customers."),
    ("SELECT COUNT(*) FROM orders", "Counts all orders."),
    (
        "SELECT COUNT(*) FROM orders WHERE total > 100",
        "Counts orders where order total is greater than 100.",
    ),
    ("SELECT COUNT(id) FROM orders", "Shows the number of id from orders."),
    ("SELECT SUM(total) FROM orders", "Shows the sum of order total from orders."),
    ("SELECT AVG(total) FROM orders", "Shows the average of order total from orders."),
    (
        "SELECT MIN(total), MAX(total) FROM orders",
        "Shows the smallest order total and the largest order total from orders.",
    ),
    (
        "SELECT status, COUNT(*) FROM orders GROUP BY status",
        "Shows status and the number of rows from orders, grouped by status.",
    ),
    (
        "SELECT SUM(total) AS revenue FROM orders",
        "Shows the sum of order total (as revenue) from orders.",
    ),
    (
        "SELECT c.region, SUM(o.total) FROM orders o "
        "JOIN customers c ON o.customer_id=c.id GROUP BY c.region",
        "Shows region and the sum of order total from orders, joined to customers, "
        "grouped by region.",
    ),
    (
        "SELECT c.name, SUM(o.total) AS revenue FROM customers c "
        "LEFT JOIN orders o ON o.customer_id=c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
        "Shows customer name and the sum of order total (as revenue) from customers, "
        "left-joined to orders, grouped by customer name, sorted by revenue (highest first), "
        "top 10.",
    ),
    (
        "SELECT id FROM orders WHERE status='shipped' AND total>100",
        "Shows id from orders where status is 'shipped' and order total is greater than 100.",
    ),
    (
        "SELECT id FROM orders WHERE total BETWEEN 10 AND 20",
        "Shows id from orders where order total is between 10 and 20.",
    ),
    (
        "SELECT id FROM orders WHERE customer_id IN (1,2,3)",
        "Shows id from orders where customer id is one of 1, 2, 3.",
    ),
    (
        "SELECT id FROM orders WHERE customer_id NOT IN (1,2)",
        "Shows id from orders where customer id is not one of 1, 2.",
    ),
    ("SELECT id FROM orders WHERE status IS NULL", "Shows id from orders where status is missing."),
    (
        "SELECT id FROM orders WHERE status IS NOT NULL",
        "Shows id from orders where status is present.",
    ),
    (
        "SELECT name FROM customers WHERE name LIKE 'A%'",
        "Shows name from customers where name matches 'A%'.",
    ),
    (
        "SELECT name FROM customers WHERE name NOT LIKE 'A%'",
        "Shows name from customers where name does not match 'A%'.",
    ),
    (
        "SELECT id FROM orders WHERE total>=50 AND total<=500",
        "Shows id from orders where order total is at least 50 and order total is at most 500.",
    ),
    ("SELECT id FROM orders WHERE total<>0", "Shows id from orders where order total is not 0."),
    (
        "SELECT id, total FROM orders ORDER BY total ASC",
        "Shows id and order total from orders, sorted by order total (lowest first).",
    ),
    (
        "SELECT id FROM orders ORDER BY total DESC, id ASC",
        "Shows id from orders, sorted by order total (highest first) and id (lowest first).",
    ),
    ("SELECT id FROM orders LIMIT 5", "Shows id from orders, limited to 5 rows."),
    ("SELECT id FROM orders LIMIT 1", "Shows id from orders, limited to 1 row."),
    (
        "SELECT region, status, COUNT(*) FROM orders GROUP BY region, status",
        "Shows region, status and the number of rows from orders, grouped by region and status.",
    ),
    (
        "SELECT o.id FROM orders o JOIN order_items i ON o.id=i.order_id "
        "WHERE i.qty>5 GROUP BY o.id HAVING COUNT(*)>3",
        "Shows id from orders, joined to line items where quantity is greater than 5, "
        "grouped by id, keeping groups where the number of rows is greater than 3.",
    ),
    ("SELECT * FROM a RIGHT JOIN b ON a.bid=b.id", "Shows all columns from a, right-joined to b."),
    (
        "SELECT * FROM a FULL OUTER JOIN b ON a.bid=b.id",
        "Shows all columns from a, outer-joined to b.",
    ),
    (
        "SELECT customer_id, SUM(order_total) FROM sales_orders GROUP BY customer_id",
        "Shows customer id and the sum of order total from sales orders, grouped by customer id.",
    ),
]


@pytest.mark.parametrize("sql,expected", GOLDENS, ids=[g[0][:48] for g in GOLDENS])
def test_golden_paraphrases(sql: str, expected: str) -> None:
    result = verbalize(sql, "postgres", VOCAB)
    assert result.text == expected
    assert result.complete is True


def test_at_least_30_goldens() -> None:
    assert len(GOLDENS) >= 30


# ---------------------------------------------------------------------------
# Multi-dialect (snowflake / bigquery / tsql)
# ---------------------------------------------------------------------------

DIALECT_GOLDENS: list[tuple[str, str, str]] = [
    (
        "snowflake",
        "SELECT O.TOTAL FROM ORDERS O JOIN CUSTOMERS C ON O.CUSTOMER_ID=C.ID",
        "Shows total from orders, joined to customers.",
    ),
    (
        "bigquery",
        "SELECT c.name, SUM(o.total) FROM `p.d.customers` c "
        "LEFT JOIN `p.d.orders` o ON o.customer_id=c.id GROUP BY c.name",
        "Shows name and the sum of order total from customers, left-joined to orders, "
        "grouped by name.",
    ),
    (
        "tsql",
        "SELECT TOP 10 o.Total FROM Orders o JOIN Customers c ON o.CustId=c.Id",
        "Shows total from orders, joined to customers, limited to 10 rows.",
    ),
]


@pytest.mark.parametrize(
    "dialect,sql,expected", DIALECT_GOLDENS, ids=[d[0] for d in DIALECT_GOLDENS]
)
def test_dialect_goldens(dialect: str, sql: str, expected: str) -> None:
    result = verbalize(sql, dialect, Vocab(columns={"total": "order total"}))
    assert result.text == expected
    assert result.complete is True


# ---------------------------------------------------------------------------
# Graceful degradation — quote verbatim, complete=False, never raise
# ---------------------------------------------------------------------------


def test_window_function_is_quoted_and_incomplete() -> None:
    result = verbalize(
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY x ORDER BY total DESC) FROM orders", "postgres"
    )
    assert result.complete is False
    assert result.text.startswith("Shows id and `ROW_NUMBER()")
    assert result.text.endswith("from orders.")
    assert any("ROW_NUMBER" in u for u in result.unhandled)


def test_arithmetic_projection_is_quoted() -> None:
    result = verbalize("SELECT id, total*2 FROM orders", "postgres")
    assert result.complete is False
    assert "`total * 2`" in result.text


def test_aggregate_of_expression_is_quoted() -> None:
    result = verbalize("SELECT SUM(total*2) FROM orders", "postgres", VOCAB)
    assert result.complete is False
    assert "the sum of `total * 2`" in result.text


def test_subquery_from_degrades() -> None:
    result = verbalize("SELECT * FROM (SELECT * FROM orders) t", "postgres")
    assert result.complete is False
    assert "a subquery" in result.text


def test_cte_degrades_but_describes_top_select() -> None:
    result = verbalize("WITH x AS (SELECT * FROM orders) SELECT id FROM x", "postgres")
    assert result.complete is False
    assert result.text == "Shows id from x."


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM a UNION SELECT id FROM b",  # set operation
        "DELETE FROM orders WHERE id=1",  # non-SELECT
        "INSERT INTO t VALUES (1)",  # non-SELECT
        "SELCT ??? FROM",  # parse error
        "",  # empty
        "   ",  # whitespace
    ],
)
def test_non_modelable_yields_empty_incomplete(sql: str) -> None:
    result = verbalize(sql, "postgres")
    assert result.complete is False
    assert result.text == ""


# ---------------------------------------------------------------------------
# Property: verbalize never raises (1000+ generated + garbage inputs)
# ---------------------------------------------------------------------------


def _generated_corpus() -> list[str]:
    selects = ["*", "id", "id, total", "COUNT(*)", "SUM(total)", "AVG(total), MIN(total)"]
    froms = ["orders", "orders o", "sales.orders", "a", "(SELECT 1) t"]
    joins = ["", "JOIN customers c ON o.cid=c.id", "LEFT JOIN b ON a.x=b.x NATURAL JOIN c"]
    wheres = ["", "WHERE total>1", "WHERE a IN (1,2) AND b IS NULL", "WHERE x BETWEEN 1 AND 9"]
    tails = ["", "GROUP BY id", "GROUP BY id HAVING COUNT(*)>1", "ORDER BY total DESC", "LIMIT 5"]
    corpus = [
        f"SELECT {s} FROM {f} {j} {w} {t}".strip()
        for s, f, j, w, t in itertools.product(selects, froms, joins, wheres, tails)
    ]
    corpus += [
        "",
        ";",
        "SELECT",
        "((((",
        "DROP TABLE users",
        "UPDATE t SET x=1",
        "SELECT * FROM a NATURAL JOIN b",
        "WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r",
        "SELECT CASE WHEN a THEN 1 ELSE 2 END FROM t",
        "🙂 not sql at all 🙂",
        "SELECT * FROM t WHERE x IN (SELECT y FROM u)",
    ]
    return corpus


@pytest.mark.parametrize("dialect", ["postgres", "snowflake", "bigquery", "tsql", "mysql"])
def test_never_raises(dialect: str) -> None:
    corpus = _generated_corpus()
    assert len(corpus) >= 200  # ×5 dialects → 1000+ evaluations
    for sql in corpus:
        result = verbalize(sql, dialect, VOCAB)
        assert isinstance(result, Paraphrase)
        assert isinstance(result.text, str)
        assert isinstance(result.complete, bool)


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ident,expected",
    [
        ("customer_id", "customer id"),
        ("TOTAL", "total"),
        ("signup_date", "signup date"),
        ('"Quoted"', "quoted"),
        ("plain", "plain"),
    ],
)
def test_humanize(ident: str, expected: str) -> None:
    assert humanize(ident) == expected


def test_vocab_label_fallback_to_humanize() -> None:
    v = Vocab(columns={"orders.total": "order total"})
    assert v.column_label("orders", "total") == "order total"  # exact key
    assert v.column_label("orders", "created_at") == "created at"  # fallback humanize
    assert v.table_label("sales_orders") == "sales orders"  # no entry → humanize


def test_bare_vocab_still_reads() -> None:
    result = verbalize("SELECT customer_id FROM sales_orders WHERE amount_usd > 0", "postgres")
    assert result.text == "Shows customer id from sales orders where amount usd is greater than 0."
    assert result.complete is True


def test_build_vocab_from_kb() -> None:
    kb = {
        "schema": {
            "tables": [
                {"name": "orders", "columns": [{"name": "total"}, {"name": "customer_id"}]},
            ]
        },
        "business_context": {
            "glossary": [{"term": "MRR", "definition": "monthly recurring revenue"}]
        },
    }
    v = build_vocab(kb)
    assert v.table_label("orders") == "orders"
    assert v.column_label("orders", "customer_id") == "customer id"
    assert v.terms["MRR"] == "monthly recurring revenue"


def test_build_vocab_tolerates_partial_kb() -> None:
    assert isinstance(build_vocab({}), Vocab)
    assert isinstance(build_vocab({"schema": {"tables": [{}, "bad", {"name": "t"}]}}), Vocab)


# ---------------------------------------------------------------------------
# Paraphrase serialization
# ---------------------------------------------------------------------------


def test_paraphrase_to_dict() -> None:
    result = verbalize("SELECT total*2 FROM orders", "postgres")
    d = result.to_dict()
    assert set(d) == {"text", "complete", "unhandled"}
    assert d["complete"] is False
    assert isinstance(d["unhandled"], list)


# ---------------------------------------------------------------------------
# IS NOT NULL — both sqlglot AST shapes
# ---------------------------------------------------------------------------


def test_is_not_null_reads_as_present_whichever_ast_sqlglot_produces() -> None:
    """`IS NOT NULL` has two parse shapes, and getting it wrong inverts the text.

    Older sqlglot builds wrap it as ``Not(Is(col, Null))``; 30.17+ on the
    postgres dialect keeps a single ``Is`` node carrying ``negate=True``. The
    goldens above only exercise whichever shape the installed version happens to
    produce, so both are asserted here against a hand-built AST — a version-drift
    regression that renders "is missing" for `IS NOT NULL` would otherwise only
    surface once CI's resolved sqlglot moved.
    """
    import sqlglot
    from nlqueries.verbalizer.ast_walk import analyze
    from nlqueries.verbalizer.templates import _render_predicate
    from sqlglot import exp

    shape = analyze("SELECT id FROM orders WHERE status IS NOT NULL", "postgres")
    column = sqlglot.parse_one("status", dialect="postgres")

    negate_form = exp.Is(this=column.copy(), expression=exp.Null(), negate=True)
    not_form = exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null()))
    plain_form = exp.Is(this=column.copy(), expression=exp.Null())

    assert _render_predicate(negate_form, shape, Vocab())[0] == "status is present"
    assert _render_predicate(not_form, shape, Vocab())[0] == "status is present"
    assert _render_predicate(plain_form, shape, Vocab())[0] == "status is missing"
