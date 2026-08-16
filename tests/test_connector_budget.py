"""Tests for the connector result-set budget (W-5).

This is an availability fix, not an optimisation. Connectors called
``fetchall()`` and built the entire result in Python *before* any row cap was
applied — and the caps live two layers up. One ``SELECT *`` over a large table
therefore materialised every row in the worker and then discarded almost all of
them; five million rows at ten columns is roughly 3.5 GB of Python objects.
"""

from __future__ import annotations

import sys

import pytest
from nlqueries import config
from nlqueries.connectors._budget import (
    BYTE_BUDGET,
    ROW_BUDGET,
    BudgetedRows,
    collect,
    effective_row_budget,
)


def _rows(count: int, width: int = 2):
    for index in range(count):
        yield [index] * width


# ---------------------------------------------------------------------------
# The row budget
# ---------------------------------------------------------------------------


def test_a_result_inside_the_budget_is_not_marked_truncated(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 100)
    rows, truncated, reason = collect(_rows(30), None)

    assert len(rows) == 30
    assert truncated is False and reason is None


def test_a_large_result_stops_at_the_budget(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 1_000)
    rows, truncated, reason = collect(_rows(1_000_000), None)

    assert len(rows) == 1_000
    assert truncated is True and reason == ROW_BUDGET


def test_a_result_exactly_at_the_budget_is_not_called_truncated(monkeypatch) -> None:
    """Off by one here is not cosmetic: it would tell every user of a
    thousand-row table that their answer was incomplete."""
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 50)
    rows, truncated, _ = collect(_rows(50), None)

    assert len(rows) == 50 and truncated is False


def test_a_caller_may_ask_for_less_but_never_for_more(monkeypatch) -> None:
    """Otherwise one request could opt out of the protection entirely."""
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 100)

    assert effective_row_budget(10) == 10
    assert effective_row_budget(10_000) == 100, "a caller raised its own ceiling"
    assert effective_row_budget(None) == 100
    assert effective_row_budget(0) == 100
    assert effective_row_budget(-5) == 100


# ---------------------------------------------------------------------------
# The byte budget
# ---------------------------------------------------------------------------


def test_wide_rows_hit_the_byte_budget_before_the_row_budget(monkeypatch) -> None:
    """Rows are a poor proxy for memory: ten thousand rows of two integers and
    ten thousand rows carrying a document each differ by orders of magnitude."""
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 100_000)
    monkeypatch.setattr(config, "CONNECTOR_MAX_RESULT_BYTES", 1_000_000)

    blob = "x" * 100_000  # ~100 KB per row

    def _wide():
        for index in range(100_000):
            yield [index, blob]

    rows, truncated, reason = collect(_wide(), None)

    assert truncated is True and reason == BYTE_BUDGET
    assert len(rows) < 1_000, f"kept {len(rows)} wide rows before noticing"


def test_narrow_rows_are_not_stopped_by_the_byte_budget(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 5_000)
    monkeypatch.setattr(config, "CONNECTOR_MAX_RESULT_BYTES", 64 * 1024 * 1024)

    rows, truncated, reason = collect(_rows(5_000), None)

    assert len(rows) == 5_000
    assert reason != BYTE_BUDGET


# ---------------------------------------------------------------------------
# Memory, which is the whole point
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_million_row_result_does_not_grow_the_process(monkeypatch) -> None:
    """The failure this exists to prevent, measured rather than argued.

    Without a budget the list below would hold a million rows; with one it holds
    ten thousand, and the difference has to show up in RSS or the budget is not
    doing anything.
    """
    tracemalloc = pytest.importorskip("tracemalloc")

    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 10_000)
    monkeypatch.setattr(config, "CONNECTOR_MAX_RESULT_BYTES", 64 * 1024 * 1024)

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    rows, truncated, _ = collect(_rows(1_000_000, width=10), None)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    assert len(rows) == 10_000 and truncated is True
    grew_mb = (after - before) / (1024 * 1024)
    assert grew_mb < 100, f"held {grew_mb:.1f} MB for a bounded read"


def test_the_generator_is_not_drained_past_the_budget(monkeypatch) -> None:
    """Stopping the *fetch* is the point. Reading every row and then discarding
    the excess would leave the memory problem exactly where it was — and keep
    the database sending."""
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 10)
    produced = 0

    def _counting():
        nonlocal produced
        for index in range(1_000_000):
            produced += 1
            yield [index]

    rows, truncated, _ = collect(_counting(), None)

    assert len(rows) == 10 and truncated is True
    # One row beyond the budget is read, because that is how truncation is
    # detected at all; a million is not.
    assert produced <= 12, f"pulled {produced} rows from the cursor for a 10-row budget"


# ---------------------------------------------------------------------------
# Accounting details
# ---------------------------------------------------------------------------


def test_the_size_estimate_samples_rather_than_measuring_everything(monkeypatch) -> None:
    """Deep-sizing every row would cost more than the fetch it protects."""
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 10_000)
    monkeypatch.setattr(config, "CONNECTOR_MAX_RESULT_BYTES", 64 * 1024 * 1024)

    sized = 0
    real_getsizeof = sys.getsizeof

    def _counting_getsizeof(obj, *args):
        nonlocal sized
        sized += 1
        return real_getsizeof(obj, *args)

    monkeypatch.setattr("nlqueries.connectors._budget.sys.getsizeof", _counting_getsizeof)
    collect(_rows(5_000, width=4), None)

    # 50 sampled rows × 4 values, not 5,000 × 4.
    assert sized <= 50 * 4, f"sized {sized} values"


def test_the_budget_reports_the_first_reason_it_hit(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONNECTOR_MAX_FETCH_ROWS", 5)
    monkeypatch.setattr(config, "CONNECTOR_MAX_RESULT_BYTES", 64 * 1024 * 1024)

    budget = BudgetedRows(None)
    for index in range(10):
        if budget.add([index]):
            break

    _, truncated, reason = budget.finish()
    assert truncated is True and reason == ROW_BUDGET
