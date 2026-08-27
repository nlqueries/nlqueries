"""
nlqueries.connectors._budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How much of a result set a connector is willing to hold in memory.

Connectors used to call ``fetchall()`` and build the whole result as Python
lists *before* any row cap was applied — and the caps live two layers up, at 200
rows in the orchestrator and 1000 in the enterprise API. So a question that
generated ``SELECT * FROM big_table`` materialised every row in the worker and
then threw almost all of them away. Five million rows at ten columns is roughly
3.5 GB of Python objects: one such query kills the worker, and the SQL console's
row ceiling makes it reachable deliberately as well as by accident.

This is an availability bound, not an optimisation, and deliberately not a
``LIMIT``. Injecting a LIMIT into the generated SQL would change what an
aggregate means and produce silently wrong answers; stopping the *fetch* only
ever costs completeness, which the result then declares.

Two budgets, because rows are a poor proxy for memory: ten thousand rows of two
integers and ten thousand rows carrying a JSON blob each differ by orders of
magnitude. Whichever is hit first stops the read.

What each dialect actually gains, stated plainly because the difference matters
when someone is diagnosing an out-of-memory kill:

* **Postgres** streams. A server-side cursor with ``stream_results`` means rows
  arrive in batches and the ones past the budget are never sent at all.
* **Everything else** bounds the *Python* side: rows are consumed from the
  cursor incrementally and stop being turned into lists once a budget is hit.
  Some drivers — psycopg2 without a named cursor, notably — still transfer the
  whole result into the client's own buffer during ``execute``, so the raw bytes
  arrive regardless. That buffer is a fraction of the cost: the 3.5 GB in the
  example above is Python list and object overhead, not wire format. Bounding it
  determines whether the worker survives, but it is not equivalent to true
  streaming, and should not be read as such.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Any

from nlqueries import config

ROW_BUDGET = "row_budget"
BYTE_BUDGET = "byte_budget"

# Rows sampled for the size estimate. Deep-sizing every row would cost more than
# the fetch it is protecting; a sample is enough to tell 40-byte rows from
# 40-kilobyte ones, which is the distinction that matters.
_SAMPLE_ROWS = 50


def effective_row_budget(max_rows: int | None) -> int:
    """The row ceiling for this call.

    A caller may ask for less than the configured maximum but never more —
    otherwise a single request could opt out of the protection entirely.
    """
    ceiling = max(1, config.CONNECTOR_MAX_FETCH_ROWS)
    if max_rows is None or max_rows <= 0:
        return ceiling
    return min(max_rows, ceiling)


def _row_bytes(row: Iterable[Any]) -> int:
    """A cheap size estimate for one row.

    ``sys.getsizeof`` per value, not a recursive walk: this runs while the
    caller waits, and being roughly right early beats being exactly right late.
    """
    return sum(sys.getsizeof(value) for value in row)


class BudgetedRows:
    """Accumulates rows until a budget is hit, then stops.

    Usage::

        budget = BudgetedRows(max_rows)
        for row in cursor:
            if budget.add(list(row)):
                break
        rows, truncated, reason = budget.finish()
    """

    def __init__(self, max_rows: int | None) -> None:
        self.limit = effective_row_budget(max_rows)
        self.byte_limit = max(1, config.CONNECTOR_MAX_RESULT_BYTES)
        self.rows: list[list[Any]] = []
        self.reason: str | None = None
        self._bytes = 0
        self._sampled = 0
        self._bytes_per_row = 0.0

    def add(self, row: list[Any]) -> bool:
        """Append *row*. Returns True when the caller should stop reading."""
        if self.reason is not None:
            return True

        if len(self.rows) >= self.limit:
            # One row beyond the budget was read, which is how truncation is
            # detected at all — a result of exactly `limit` rows is otherwise
            # indistinguishable from a table that happens to have that many.
            self.reason = ROW_BUDGET
            return True

        self.rows.append(row)

        if self._sampled < _SAMPLE_ROWS:
            self._bytes += _row_bytes(row)
            self._sampled += 1
            self._bytes_per_row = self._bytes / self._sampled
        else:
            # Extrapolate from the sample rather than measuring every row.
            self._bytes = int(self._bytes_per_row * len(self.rows))

        if self._bytes >= self.byte_limit:
            self.reason = BYTE_BUDGET
            return True

        return False

    def finish(self) -> tuple[list[list[Any]], bool, str | None]:
        return self.rows, self.reason is not None, self.reason


def collect(
    cursor: Iterator[Any], max_rows: int | None
) -> tuple[list[list[Any]], bool, str | None]:
    """Drain *cursor* into rows, stopping at whichever budget binds first.

    The shared implementation for every dialect: what differs between them is
    how a cursor is obtained and whether it streams, never what a budget means.
    """
    budget = BudgetedRows(max_rows)
    for row in cursor:
        if budget.add(list(row)):
            break
    return budget.finish()
