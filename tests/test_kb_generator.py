"""Tests for nlqueries.knowledge.kb_generator (Task 4.3.1)."""

from __future__ import annotations

from pathlib import Path

import yaml
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.knowledge.kb_generator import generate_knowledge_base, save_knowledge_base
from nlqueries.processing.parameterizer import Placeholder, QueryCapsule

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_column(
    name: str, col_type: str = "VARCHAR", description: str | None = None
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type=col_type,
        nullable=True,
        is_primary_key=False,
        is_foreign_key=False,
        references=None,
        description=description,
    )


def _make_table(
    name: str,
    columns: list[ColumnSpec] | None = None,
    row_count: int | None = None,
    description: str | None = None,
) -> TableSpec:
    return TableSpec(
        name=name,
        schema="public",
        row_count=row_count,
        columns=columns or [_make_column("id", "INT")],
        description=description,
    )


def _make_schema(tables: list[TableSpec] | None = None) -> SchemaSpec:
    return SchemaSpec(
        database="testdb",
        tables=tables if tables is not None else [_make_table("orders")],
        extracted_at="2026-06-08T00:00:00+00:00",
    )


def _make_capsule(intent: str = "Get orders by status", frequency: int = 10) -> QueryCapsule:
    return QueryCapsule(
        template_sql="SELECT id FROM orders WHERE status = '[status:VARCHAR]'",
        placeholders=[Placeholder(name="status", type="VARCHAR")],
        tables=["orders"],
        columns=["id", "status"],
        frequency=frequency,
        auto_description="Query on orders filtering by status",
        intent=intent,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_output_has_required_top_level_keys():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert "schema" in kb
    assert "business_context" in kb
    assert "query_capsules" in kb


def test_schema_contains_tables_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert isinstance(kb["schema"]["tables"], list)


def test_business_context_has_glossary_and_rules():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    bc = kb["business_context"]
    assert bc["glossary"] == []
    assert bc["rules"] == []


def test_query_capsules_is_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert isinstance(kb["query_capsules"], list)


# ---------------------------------------------------------------------------
# Table and capsule count tests
# ---------------------------------------------------------------------------


def test_table_count_matches_schema():
    tables = [_make_table("orders"), _make_table("customers"), _make_table("products")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert len(kb["schema"]["tables"]) == 3


def test_capsule_count_matches_inputs():
    capsules = [
        _make_capsule(),
        _make_capsule("Get customers by id"),
        _make_capsule("List products"),
    ]
    kb = generate_knowledge_base(_make_schema(), capsules, agent_name="agent1")
    assert len(kb["query_capsules"]) == len(capsules)


def test_empty_capsules_produces_empty_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert kb["query_capsules"] == []


def test_empty_schema_tables_produces_empty_list():
    schema = _make_schema(tables=[])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert kb["schema"]["tables"] == []


# ---------------------------------------------------------------------------
# Table field mapping tests
# ---------------------------------------------------------------------------


def test_table_fields_present():
    schema = _make_schema([_make_table("orders", row_count=500)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    tbl = kb["schema"]["tables"][0]
    assert tbl["name"] == "orders"
    assert tbl["row_count"] == 500
    assert "description" in tbl
    assert "columns" in tbl


def test_column_fields_present():
    cols = [_make_column("id", "INT"), _make_column("status", "VARCHAR")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    col_names = {c["name"] for c in kb["schema"]["tables"][0]["columns"]}
    assert col_names == {"id", "status"}
    for col in kb["schema"]["tables"][0]["columns"]:
        assert "name" in col
        assert "type" in col
        assert "description" in col


def test_capsule_fields_present():
    cap = _make_capsule("How many orders are pending?", frequency=42)
    kb = generate_knowledge_base(_make_schema(), [cap], agent_name="agent1")
    entry = kb["query_capsules"][0]
    assert entry["intent"] == "How many orders are pending?"
    assert entry["template"] == cap.template_sql
    assert entry["frequency"] == 42


def test_row_count_none_preserved():
    schema = _make_schema([_make_table("orders", row_count=None)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert kb["schema"]["tables"][0]["row_count"] is None


# ---------------------------------------------------------------------------
# Merge / preserve manual description tests
# ---------------------------------------------------------------------------


def test_merging_preserves_manual_table_description():
    existing_kb = {
        "schema": {
            "tables": [
                {"name": "orders", "description": "Manual: customer order records", "columns": []}
            ]
        }
    }
    schema = _make_schema([_make_table("orders", description="auto description")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert kb["schema"]["tables"][0]["description"] == "Manual: customer order records"


def test_merging_preserves_manual_column_description():
    existing_kb = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "",
                    "columns": [{"name": "status", "description": "Manual: order lifecycle state"}],
                }
            ]
        }
    }
    cols = [_make_column("status", "VARCHAR", description="auto col desc")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    status_col = next(c for c in kb["schema"]["tables"][0]["columns"] if c["name"] == "status")
    assert status_col["description"] == "Manual: order lifecycle state"


def test_merging_falls_back_to_schema_description_when_existing_is_empty():
    existing_kb = {"schema": {"tables": [{"name": "orders", "description": "", "columns": []}]}}
    schema = _make_schema([_make_table("orders", description="Schema-level auto description")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert kb["schema"]["tables"][0]["description"] == "Schema-level auto description"


def test_merging_new_tables_not_in_existing_kb():
    """A table in schema but not in existing_kb should still appear with auto description."""
    existing_kb = {
        "schema": {"tables": [{"name": "orders", "description": "Manual", "columns": []}]}
    }
    tables = [_make_table("orders"), _make_table("products", description="Auto products desc")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert len(kb["schema"]["tables"]) == 2
    products_tbl = next(t for t in kb["schema"]["tables"] if t["name"] == "products")
    assert products_tbl["description"] == "Auto products desc"


def test_no_existing_kb_uses_schema_descriptions():
    cols = [_make_column("id", "INT", description="Primary key")]
    schema = _make_schema([_make_table("orders", columns=cols, description="Order records")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=None)
    assert kb["schema"]["tables"][0]["description"] == "Order records"
    id_col = kb["schema"]["tables"][0]["columns"][0]
    assert id_col["description"] == "Primary key"


# ---------------------------------------------------------------------------
# save_knowledge_base tests
# ---------------------------------------------------------------------------


def test_save_knowledge_base_writes_valid_yaml(tmp_path: Path):
    kb = generate_knowledge_base(_make_schema(), [_make_capsule()], agent_name="agent1")
    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))

    assert out.exists()
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "schema" in loaded
    assert "business_context" in loaded
    assert "query_capsules" in loaded


def test_save_knowledge_base_creates_parent_dirs(tmp_path: Path):
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    nested = tmp_path / "deep" / "nested" / "kb.yaml"
    save_knowledge_base(kb, str(nested))
    assert nested.exists()


def test_save_knowledge_base_round_trip(tmp_path: Path):
    capsules = [_make_capsule("How many orders?"), _make_capsule("List customers")]
    tables = [_make_table("orders", row_count=100), _make_table("customers")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, capsules, agent_name="agent1")

    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert len(loaded["schema"]["tables"]) == 2
    assert len(loaded["query_capsules"]) == 2
    assert loaded["query_capsules"][0]["intent"] == "How many orders?"


def test_save_knowledge_base_uses_yaml_settings(tmp_path: Path):
    kb = generate_knowledge_base(
        _make_schema([_make_table("orders", description="Unicode: café")]),
        [],
        agent_name="agent1",
    )
    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))
    raw = out.read_text(encoding="utf-8")
    # allow_unicode=True means non-ASCII chars are NOT escaped
    assert "café" in raw
    # default_flow_style=False means no inline dicts on single line
    assert "{" not in raw
