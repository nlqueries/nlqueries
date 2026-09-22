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
    """For knowledge bases written before the field existed, and hand-edited
    entries.

    Not for connectors: every one reports a schema, since `TableSpec.schema` is
    a required `str`.
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


def test_the_hint_list_agrees_with_the_schema_block() -> None:
    """One prompt must not show two names for one table.

    `_format_dynamic_context` heads its list with names from the Qdrant
    payload, which `upsert_schema` writes unqualified -- so a prompt could
    offer `sales.orders` in the schema block and `orders` in the hint list
    directly above it.
    """
    from nlqueries.orchestrator.prompt_assembly import _format_dynamic_context

    kb = _kb({"name": "orders", "schema": "sales", "columns": []})
    out = _format_dynamic_context(kb, 5, ["orders"], [], [])
    assert "sales.orders" in out, out


def test_a_hit_the_kb_does_not_hold_keeps_its_bare_name() -> None:
    """Better a name the model can still match than one invented here."""
    from nlqueries.orchestrator.prompt_assembly import _format_dynamic_context

    kb = _kb({"name": "orders", "schema": "sales", "columns": []})
    out = _format_dynamic_context(kb, 5, ["legacy_audit"], [], [])
    assert "legacy_audit" in out


def test_a_duplicated_table_name_stays_bare_in_the_hint_list() -> None:
    """The regression qualifying the hint list introduced.

    A knowledge base holding both `public.orders` and `sales.orders` is the
    ordinary multi-schema case, and the Qdrant payload carries no schema -- so
    nothing here can say which was meant. Naming one would point the model at a
    definite schema that may be the wrong one, and a query against the wrong
    table returns a plausible answer rather than an error. Bare leaves the
    schema block to disambiguate, which is what it did before.
    """
    from nlqueries.orchestrator.prompt_assembly import _format_dynamic_context

    kb = {
        "db_name": "sales",
        "schema": {
            "tables": [
                {"name": "orders", "schema": "public", "columns": []},
                {"name": "orders", "schema": "sales", "columns": []},
            ]
        },
    }
    out = _format_dynamic_context(kb, 5, ["orders"], [], [])
    assert "public.orders" not in out, out
    assert "sales.orders" not in out, out
    assert "orders" in out
