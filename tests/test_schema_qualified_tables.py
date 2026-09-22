"""
tests.test_schema_qualified_tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Naming a table to the model the way the database will accept it.

The knowledge base recorded only the bare table name, so a prompt could not say
where a table lived. On a connection whose default schema already held
everything that was invisible; anywhere else the model produced SQL referring
to a table the server could not resolve.

All THREE renderers are covered on purpose, and the count is the point. Two
live in ``prompt_assembly`` -- ``compact`` is the default and ``verbose`` is
what an older deployment gets -- and the third is
``sql_generation._format_schema_for_prompt``, which its own comment calls "the
one most easily missed". It is not a side path: it feeds ``generate_sql`` and
the repair step, which runs exactly when the first attempt was wrong. The first
version of this change covered two of the three and said so in its
description.
"""

from __future__ import annotations

from typing import Any

from nlqueries.orchestrator.prompt_assembly import (
    _build_full_schema_section,
    _render_m_schema,
    _table_ref,
)
from nlqueries.orchestrator.sql_generation import _format_schema_for_prompt


def _kb(table: dict[str, Any]) -> dict[str, Any]:
    return {"db_name": "sales", "schema": {"tables": [table]}}


def _compact(kb: dict[str, Any]) -> str:
    return _render_m_schema(kb)


# --- the reference itself ----------------------------------------------------


def test_a_table_with_a_schema_is_named_by_both() -> None:
    assert _table_ref({"name": "orders", "schema": "sales"}) == "sales.orders"


def test_a_table_without_a_schema_keeps_its_bare_name() -> None:
    """Two cases want this, and both want the old behaviour.

    A connector that reports no schema at all -- DuckDB and SQLite -- and a
    knowledge base written before the field existed.
    """
    assert _table_ref({"name": "orders"}) == "orders"
    assert _table_ref({"name": "orders", "schema": ""}) == "orders"
    assert _table_ref({"name": "orders", "schema": None}) == "orders"


def test_whitespace_is_not_a_schema() -> None:
    # `"   .orders"` would be worse than the bare name: it is not a table
    # anywhere, where the bare name at least resolves on the default schema.
    assert _table_ref({"name": "orders", "schema": "   "}) == "orders"


# --- the renderers -----------------------------------------------------------


def test_compact_renders_the_qualified_name() -> None:
    rendered = _compact(_kb({"name": "orders", "schema": "sales", "columns": []}))
    assert "sales.orders" in rendered, rendered


def test_verbose_renders_the_qualified_name() -> None:
    """The renderer a `verbose` deployment gets, which must not be forgotten."""
    rendered = _build_full_schema_section(_kb({"name": "orders", "schema": "sales", "columns": []}))
    assert "### Table: sales.orders" in rendered, rendered


def test_a_kb_without_schemas_renders_exactly_as_before() -> None:
    """Backward compatibility, asserted rather than assumed.

    Every knowledge base generated before this change has no `schema` key, and
    those prompts must not move: a table that renders as `.orders` names
    nothing at all.
    """
    table = {"name": "orders", "columns": [{"name": "id", "type": "BIGINT"}]}
    for rendered in (_compact(_kb(table)), _build_full_schema_section(_kb(table))):
        assert "orders" in rendered
        assert ".orders" not in rendered, rendered


def test_the_sql_generation_renderer_qualifies_too() -> None:
    """The third renderer, and the one a reader is least likely to look for.

    `_format_schema_for_prompt` feeds `_build_sql_system_prompt`, which serves
    both `generate_sql` and `validate_and_repair`. Leaving it bare meant the
    main generation path still asked for SQL against `orders` while the other
    two renderers said `sales.orders`.
    """
    rendered = _format_schema_for_prompt(_kb({"name": "orders", "schema": "sales", "columns": []}))
    assert "Table: sales.orders" in rendered, rendered


def test_the_sql_generation_renderer_falls_back_too() -> None:
    table = {"name": "orders", "columns": [{"name": "id", "type": "BIGINT"}]}
    rendered = _format_schema_for_prompt(_kb(table))
    assert "Table: orders" in rendered
    assert ".orders" not in rendered, rendered
