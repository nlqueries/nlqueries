"""
nlqueries.orchestrator.empty_explain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Which condition left a result empty (owner feedback F20, phase 1).

A statement asks for flights after 20:00 from San Francisco on a night service,
and returns nothing. The generic note says a filter may be narrower than meant,
which is true and of no help: the user cannot tell that there are 23 such
flights and none of them is a night service. Here, after an empty result that
nothing more specific has explained (see :mod:`~nlqueries.orchestrator.
empty_result`), each condition of the statement's ``WHERE`` is left out in turn
and the rows counted. A condition without which rows come back is named, with
the count:

    No rows match every condition. Without night = 'YES', 23 rows match.

Only counts are read, never rows, and through the request's own connector, so
the row filters a caller has bound to it count only the rows the user can see.
A condition on a column whose values are personal (:func:`~nlqueries.knowledge.
kb_generator.is_pii_column`) is never tried, so it is never named; nor is one
with a subquery, whose rows would be counted along with it. Only the top-level
``AND`` conditions of a single ``SELECT`` are tried, at most
:data:`MAX_CONDITIONS`, one after another, each with a timeout of at most
:data:`COUNT_TIMEOUT_SECONDS` and within the request's deadline.

Off unless ``NLQ_EMPTY_RESULT_EXPLAIN`` (``empty_result_explain``) turns it on,
and then only on the engines in :data:`DIALECTS`: on a warehouse that bills by
the data a query reads, each count costs as much as the statement.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import sqlglot
from sqlglot import exp

from nlqueries.knowledge.kb_generator import is_pii_column
from nlqueries.sql_policy import _sqlglot_dialect

_log = logging.getLogger(__name__)

#: The most conditions left out, one per count.
MAX_CONDITIONS = 4
#: The longest any one count may run.
COUNT_TIMEOUT_SECONDS = 2.0
#: Engines it runs on, as sqlglot names them: those whose connector holds each
#: statement to the timeout it is given. SQL Server's applies only its
#: connection-wide timeout (see ``connectors/mssql.py``), so a count there could
#: run for minutes. BigQuery and Snowflake bill by the data a query reads.
DIALECTS = frozenset({"postgres", "mysql", "sqlite", "duckdb"})


def _conditions(where: exp.Expression) -> list[exp.Expression]:
    """The top-level ``AND`` conditions of *where*, in the order written."""
    node = where.this if isinstance(where, exp.Where) else where
    out: list[exp.Expression] = []

    def walk(e: exp.Expression) -> None:
        e = e.unnest()
        if isinstance(e, exp.And):
            walk(e.left)
            walk(e.right)
        else:
            out.append(e)

    walk(node)
    return out


def _triable(condition: exp.Expression) -> bool:
    """Whether *condition* may be left out and named: it reads no subquery and
    no column whose values are personal."""
    if condition.find(exp.Subquery, exp.Select, exp.Exists) is not None:
        return False
    return not any(is_pii_column(c.name) for c in condition.find_all(exp.Column))


def _is_star(column: exp.Expression) -> bool:
    """Whether *column*, an output column, is ``*`` or ``t.*``."""
    e = column.unalias()
    return isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))


def _plain(select: exp.Select) -> bool:
    """Whether each of *select*'s rows is one row it reads: no ``DISTINCT``,
    grouping or aggregate folds rows together."""
    return not (
        select.args.get("distinct")
        or select.args.get("group")
        or select.args.get("having")
        or any(e.find(exp.AggFunc, exp.Window) for e in select.expressions)
    )


def _counted(select: exp.Select, kept: list[exp.Expression]) -> exp.Select:
    """A ``COUNT(*)`` of *select*'s rows with only the *kept* conditions.

    The statement is counted as a derived table, so ``DISTINCT`` and
    ``GROUP BY`` count what they return. Its ordering and row limit are left
    out: one changes nothing counted, the other would cap the count. A plain
    statement counts ``1`` per row instead of its columns. Otherwise each column
    keeps its alias, which ``GROUP BY`` or ``HAVING`` may name, and one with none
    is given a name no other column has, so the derived table has no unnamed
    column.
    A ``WITH`` moves to the outer statement, where every engine takes it.
    """
    inner = select.copy()
    with_ = inner.args.get("with_")
    inner.set("with_", None)
    for key in ("order", "limit", "offset"):
        inner.set(key, None)
    inner.set("where", exp.Where(this=exp.and_(*kept)) if kept else None)
    if _plain(inner):
        inner.set("expressions", [exp.alias_(exp.Literal.number(1), "c1")])
    else:
        taken = {e.alias.lower() for e in inner.expressions if e.alias}
        named: list[exp.Expression] = []
        for i, e in enumerate(inner.expressions, 1):
            if e.alias:
                named.append(e.copy())
                continue
            name = f"c{i}"
            while name in taken:
                name = f"_{name}"
            taken.add(name)
            named.append(exp.alias_(e.copy(), name))
        inner.set("expressions", named)
    outer = exp.select(exp.Count(this=exp.Star())).from_(inner.subquery("nlq_counted"))
    if with_ is not None:
        outer.set("with_", with_)
    return outer


def count_statements(sql: str, dialect: str) -> list[tuple[str, str]]:
    """``(condition, count statement)`` for each condition of *sql* to leave out.

    Empty when *sql* is not one ``SELECT`` with a ``WHERE``, or when any output
    column is a ``*`` the count would have to name.
    """
    dialect = _sqlglot_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 — nothing to explain with
        return []
    if not isinstance(parsed, exp.Select) or parsed.args.get("where") is None:
        return []
    if not _plain(parsed) and any(_is_star(e) for e in parsed.expressions):
        # A plain statement's columns are not counted, so `SELECT *` is fine;
        # `SELECT DISTINCT *` would have to be named column by column.
        return []
    conditions = _conditions(parsed.args["where"])
    out: list[tuple[str, str]] = []
    for i, condition in enumerate(conditions):
        if len(out) == MAX_CONDITIONS:
            break
        if not _triable(condition):
            continue
        kept = [c.copy() for j, c in enumerate(conditions) if j != i]
        try:
            out.append(
                (condition.sql(dialect=dialect), _counted(parsed, kept).sql(dialect=dialect))
            )
        except Exception:  # noqa: BLE001 — that condition is not tried
            _log.debug("Could not build a count leaving out a condition.", exc_info=True)
    return out


def _rows(n: int) -> str:
    return "1 row matches" if n == 1 else f"{n:,} rows match"


async def explain_empty_result(
    sql: str,
    *,
    connector: Any,
    dialect: str,
    budget: Callable[[], tuple[bool, float | None]],
) -> list[str]:
    """Notes naming each condition without which *sql* returns rows; empty when
    none does, or when *dialect* is not one it runs on.

    *budget* says, before each count, whether there is time for one more and the
    timeout to give it (``(runnable, timeout)``, as the orchestrator's
    correction budget does); each count gets at most :data:`COUNT_TIMEOUT_SECONDS`.
    A count that fails or runs out of time is skipped, and so is the rest once
    the budget says there is no time.
    """
    if _sqlglot_dialect(dialect) not in DIALECTS:
        return []
    found: list[str] = []
    for condition, statement in count_statements(sql, dialect):
        runnable, timeout = budget()
        if not runnable:
            break
        # None, or zero and less, is "no timeout" to the connectors: never for a count.
        timeout = (
            COUNT_TIMEOUT_SECONDS
            if timeout is None or timeout <= 0
            else min(timeout, COUNT_TIMEOUT_SECONDS)
        )
        try:
            qr = await asyncio.to_thread(connector.execute_query, statement, timeout)
        except Exception:  # noqa: BLE001 — advisory: that condition goes unnamed
            _log.warning("Counting rows without one condition failed.", exc_info=True)
            continue
        if qr.error is not None:
            # Said, so a statement no count can be built for is seen.
            _log.warning("Counting rows without one condition returned an error.")
            _log.debug("The count's error: %s", qr.error)
            continue
        if not qr.rows:
            continue
        try:
            n = int(qr.rows[0][0])
        except (TypeError, ValueError, IndexError):
            continue
        if n > 0:
            found.append(f"Without {condition}, {_rows(n)}.")
    if not found:
        return []
    return ["No rows match every condition. " + " ".join(found)]
