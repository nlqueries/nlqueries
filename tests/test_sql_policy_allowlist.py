"""The anonymous-function allowlist answers to every spelling of a dialect.

It was keyed by the callers' spellings and read with the name it was handed, so
the two drifted: the CLI hands SQL Server down as ``tsql`` (the name sqlglot
parses) while the entry was ``mssql``, and ``postgresql`` -- what ``nlqueries
connect postgresql`` saves -- missed Postgres's entry altogether.
"""

from __future__ import annotations

from collections import Counter

import pytest
from nlqueries.cli.main import DIALECT_CHOICES
from nlqueries.sql_policy import (
    _DIALECT_ALIASES,
    ALLOWED_ANONYMOUS,
    _sqlglot_dialect,
    allowed_anonymous,
    evaluate,
)
from nlqueries.sql_policy_report import InventoryReport
from sqlglot.dialects.dialect import Dialect

AGE = "SELECT age(created_at) FROM orders"


@pytest.mark.parametrize("key", sorted(ALLOWED_ANONYMOUS))
def test_the_table_is_keyed_by_sqlglot_names(key: str) -> None:
    """A key in a caller's spelling (``mssql``) is a key no lookup reaches."""
    assert _sqlglot_dialect(key) == key
    Dialect.get_or_raise(key)


@pytest.mark.parametrize("choice", DIALECT_CHOICES)
def test_every_engine_the_cli_offers_has_an_entry(choice: str) -> None:
    """Even an empty one: absent and empty allow the same, but only an entry
    says somebody decided."""
    assert _sqlglot_dialect(choice) in ALLOWED_ANONYMOUS


@pytest.mark.parametrize(("spelling", "canonical"), sorted(_DIALECT_ALIASES.items()))
def test_each_spelling_reads_its_canonical_entry(spelling: str, canonical: str) -> None:
    assert allowed_anonymous(spelling) == allowed_anonymous(canonical)
    assert allowed_anonymous(spelling.upper()) == allowed_anonymous(canonical)


def test_postgresql_now_gets_postgres_s_functions() -> None:
    """The one decision this changes, and why the policy version moved."""
    assert evaluate(AGE, "postgres").allowed
    assert evaluate(AGE, "postgresql").allowed


def test_a_dialect_with_no_entry_still_allows_nothing() -> None:
    assert allowed_anonymous("oracle") == frozenset()
    assert not evaluate(AGE, "oracle").allowed


def test_the_inventory_reads_the_same_entry() -> None:
    """Its candidate list is "called, but not yet allowlisted"."""
    report = InventoryReport(dialect="postgresql", anonymous_counts=Counter({"age": 3, "foo": 1}))
    assert report.candidates == [("foo", 1)]
