# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.orchestrator.result_checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Signs that a statement which ran returned something other than what was asked.

A statement can be valid, run without error, and still answer another question:
a join that repeats each row, a ranking headed by a row with no value, a
percentage of a million. Nothing upstream sees these, because they are visible
only in the result. The checks here read the statement and its result, run no
query and call no model, and report what they find.

Each finding is either one to repair -- the orchestrator asks the model once to
correct the statement, and keeps the correction only if its result is clean --
or one to tell the user about, because a person can judge it and a rule cannot.
An empty result is the second kind: often it is the answer.
"""

from __future__ import annotations

import decimal
import re
from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

from nlqueries.connectors.base import QueryResult


@dataclass(frozen=True)
class ResultConcern:
    """Something about a result that suggests the statement missed the question."""

    #: A short name, for logs and tests.
    kind: str
    #: What was seen, in a sentence: told to the model when it is asked to
    #: correct the statement, and to the user when the concern stands.
    detail: str
    #: Whether the model is asked to correct the statement for it.
    repairable: bool


#: A column named as a percentage of a whole: 0 to 100. A change, growth or
#: difference expressed in percent can rightly leave that range, so a name that
#: says so is not checked, and nor is a percentile (`latency_95th_percentile`),
#: which is a value of the column it ranks, not a share.
_PERCENT_NAME = re.compile(r"percent(?!ile)|pct|(^|_)perc($|_)", re.IGNORECASE)
_CHANGE_NAME = re.compile(
    r"chang|growth|grow|increase|decrease|diff|delta|var|gain|loss|return|yoy|mom|lift",
    re.IGNORECASE,
)

_AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)

_EXPLICIT_NULLS_FIRST = re.compile(r"\bNULLS\s+FIRST\b", re.IGNORECASE)


def check_result(sql: str, result: QueryResult, dialect: str) -> list[ResultConcern]:
    """What looks wrong about *result*, the rows *sql* returned.

    Nothing for a result that carries an error, which is handled as an error.
    """
    if result.error is not None:
        return []
    statement = _parse(sql, dialect)
    concerns = [
        *_duplicate_rows(statement, result),
        *_null_heads_a_ranking(sql, statement, result),
        *_percentage_out_of_range(result),
        *_empty_result(statement, result),
        *_blank_aggregate(statement, result),
    ]
    return concerns


def _parse(sql: str, dialect: str) -> exp.Expr | None:
    try:
        return sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.SqlglotError, ValueError):
        return None


def _row_key(row: list[Any]) -> tuple[str, ...]:
    """A hashable stand-in for *row*: values of any type, compared as rendered."""
    return tuple(repr(value) for value in row)


def _duplicate_rows(statement: exp.Expr | None, result: QueryResult) -> list[ResultConcern]:
    """Rows that repeat another exactly, from a statement that joins: usually
    the join multiplying them.

    Only where the outer statement joins or reads a derived table. From one
    table, repeats are the data -- two orders of the same amount -- and a
    correction that removed them would drop real rows. Repairable, and the
    model is told it may keep the statement: a list of names can repeat a name
    two different people share.
    """
    if len(result.rows) < 2 or not _combines_rows(statement):
        return []
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for row in result.rows:
        key = _row_key(row)
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    if not repeats:
        return []
    return [
        ResultConcern(
            kind="duplicate_rows",
            detail=(
                f"{repeats} of the {len(result.rows)} rows returned repeat another "
                "row exactly, which a join that matches several rows per item "
                "often causes."
            ),
            repairable=True,
        )
    ]


def _combines_rows(statement: exp.Expr | None) -> bool:
    """Whether the outer statement joins, or reads a derived table, so a row of
    its result can stand for several rows of what it read."""
    select = _outer_select(statement)
    if select is None:
        return False
    # `from_` in current sqlglot, `from` in older releases.
    source = select.args.get("from_") or select.args.get("from")
    return bool(select.args.get("joins")) or (
        source is not None and isinstance(source.this, exp.Subquery)
    )


def _outer_select(statement: exp.Expr | None) -> exp.Select | None:
    return statement if isinstance(statement, exp.Select) else None


def _null_heads_a_ranking(
    sql: str, statement: exp.Expr | None, result: QueryResult
) -> list[ResultConcern]:
    """A ranking (``ORDER BY ... LIMIT``) whose first row has no value for the
    key it was ranked by. Most databases sort NULL first in one direction or
    the other, so "the earliest" or "the highest" can be a row with none.

    Checked only when the key is one of the returned columns; otherwise its
    value is not in the result to see.
    """
    select = _outer_select(statement)
    if select is None or not result.rows:
        return []
    order = select.args.get("order")
    if order is None or select.args.get("limit") is None or not order.expressions:
        return []
    if _EXPLICIT_NULLS_FIRST.search(sql):
        # Asked for in so many words, so a NULL at the top is what was meant.
        # Read from the text: sqlglot fills `nulls_first` in from the dialect's
        # default, so the parsed key cannot tell this from an ORDER BY that
        # never mentioned NULLs.
        return []
    key = order.expressions[0]
    index = _projection_index(select, key.this if isinstance(key, exp.Ordered) else key)
    if index is None or index >= len(result.rows[0]) or result.rows[0][index] is not None:
        return []
    column = result.columns[index] if index < len(result.columns) else "the ranking column"
    return [
        ResultConcern(
            kind="null_heads_a_ranking",
            detail=(
                f"The first row is ranked by {column} and has no value (NULL) for it, "
                "so it heads a ranking it has no place in."
            ),
            repairable=True,
        )
    ]


def _projection_index(select: exp.Select, key: exp.Expression) -> int | None:
    """Which returned column *key* sorts by: one of that name or alias, or the
    same expression."""
    key_sql = key.sql()
    key_name = key.name if isinstance(key, exp.Column | exp.Identifier) else None
    for index, projection in enumerate(select.expressions):
        if key_name and projection.alias_or_name == key_name:
            return index
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        if inner.sql() == key_sql:
            return index
    return None


def _is_number(value: Any) -> bool:
    """A number that compares: not a bool, and not NaN, which a PostgreSQL
    ``numeric`` can hold and which raises when a Decimal is ordered."""
    if isinstance(value, bool) or not isinstance(value, int | float | decimal.Decimal):
        return False
    if isinstance(value, decimal.Decimal):
        return not value.is_nan()
    return value == value  # noqa: PLR0124 -- false only for a float NaN


def _percentage_out_of_range(result: QueryResult) -> list[ResultConcern]:
    """A column named as a percentage of a whole with a value outside 0-100:
    a ratio multiplied twice, or divided the wrong way round."""
    concerns = []
    for index, column in enumerate(result.columns):
        name = str(column)
        if not _PERCENT_NAME.search(name) or _CHANGE_NAME.search(name):
            continue
        outside = [
            row[index]
            for row in result.rows
            if index < len(row) and _is_number(row[index]) and not 0 <= row[index] <= 100
        ]
        if outside:
            concerns.append(
                ResultConcern(
                    kind="percentage_out_of_range",
                    detail=(
                        f"{name} reads as a percentage, and {len(outside)} of its values "
                        f"lie outside 0 to 100 (for example {outside[0]})."
                    ),
                    repairable=True,
                )
            )
    return concerns


def _filters(select: exp.Select) -> bool:
    return any(select.args.get(arg) is not None for arg in ("where", "having", "joins"))


def _empty_result(statement: exp.Expr | None, result: QueryResult) -> list[ResultConcern]:
    """No rows from a statement that filters. Often the true answer, so only
    noted: a value spelled differently in the data, or a filter stricter than
    the question, gives the same result."""
    select = _outer_select(statement)
    if result.rows or result.truncated or select is None or not _filters(select):
        return []
    return [
        ResultConcern(
            kind="empty_result",
            detail=(
                "No rows matched. If you expected some, a filter may be narrower than "
                "intended, for example a value spelled differently in the data."
            ),
            repairable=False,
        )
    ]


def _blank_aggregate(statement: exp.Expr | None, result: QueryResult) -> list[ResultConcern]:
    """One row of aggregates, every one of them 0 or empty: the filters or joins
    let nothing through to be counted or summed. Noted, not repaired: zero is
    often the answer."""
    select = _outer_select(statement)
    if select is None or len(result.rows) != 1 or select.args.get("group") is not None:
        return []
    projections = [p.this if isinstance(p, exp.Alias) else p for p in select.expressions]
    if not projections or not all(isinstance(p, _AGGREGATES) for p in projections):
        return []
    if not all(value is None or (_is_number(value) and value == 0) for value in result.rows[0]):
        return []
    return [
        ResultConcern(
            kind="blank_aggregate",
            detail=(
                "The totals came back zero or empty. If you expected otherwise, a filter "
                "or join may exclude the rows you meant."
            ),
            repairable=False,
        )
    ]
