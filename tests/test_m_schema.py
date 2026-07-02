"""Tests for Phase 6B — M-Schema compact serialization.

Covers:
- generate_knowledge_base() v2 fields (PK/FK/references/samples/kb_version/db_name)
- PII column sample suppression
- _render_m_schema() output correctness
- assemble_prompt() schema-format switching via config.SCHEMA_FORMAT
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.knowledge.kb_generator import _is_pii_column, generate_knowledge_base
from nlqueries.orchestrator.prompt_assembly import _render_m_schema, assemble_prompt
from nlqueries.processing.parameterizer import QueryCapsule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(
    name: str,
    col_type: str = "TEXT",
    is_primary_key: bool = False,
    is_foreign_key: bool = False,
    references: str | None = None,
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type=col_type,
        nullable=True,
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
        references=references,
        description=None,
    )


def _table(name: str, columns: list[ColumnSpec], description: str = "") -> TableSpec:
    return TableSpec(
        name=name,
        schema="public",
        row_count=100,
        columns=columns,
        description=description,
    )


def _schema(*tables: TableSpec, database: str = "testdb") -> SchemaSpec:
    return SchemaSpec(
        database=database,
        tables=list(tables),
        extracted_at="2026-01-01T00:00:00+00:00",
    )


def _capsule() -> QueryCapsule:
    return QueryCapsule(
        template_sql="SELECT COUNT(*) FROM orders",
        placeholders=[],
        tables=["orders"],
        columns=[],
        frequency=5,
        auto_description="count orders",
        intent="Count all orders",
    )


def _kb_from_schema(*tables: TableSpec, **kwargs: Any) -> dict[str, Any]:
    return generate_knowledge_base(
        schema=_schema(*tables),
        capsules=[_capsule()],
        agent_name="agent1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# generate_knowledge_base() — v2 metadata fields
# ---------------------------------------------------------------------------


class TestGenerateKnowledgeBaseV2:
    def test_kb_version_is_2(self) -> None:
        kb = _kb_from_schema(_table("orders", [_col("id", "INTEGER")]))
        assert kb.get("kb_version") == 2

    def test_db_name_stored(self) -> None:
        schema = _schema(_table("orders", [_col("id")]), database="sales_db")
        kb = generate_knowledge_base(schema, [_capsule()], agent_name="a")
        assert kb.get("db_name") == "sales_db"

    def test_is_primary_key_stored(self) -> None:
        kb = _kb_from_schema(_table("orders", [_col("id", "INTEGER", is_primary_key=True)]))
        col = kb["schema"]["tables"][0]["columns"][0]
        assert col["is_primary_key"] is True

    def test_is_foreign_key_stored(self) -> None:
        kb = _kb_from_schema(
            _table(
                "orders",
                [_col("customer_id", "INTEGER", is_foreign_key=True, references="customers.id")],
            )
        )
        col = kb["schema"]["tables"][0]["columns"][0]
        assert col["is_foreign_key"] is True
        assert col["references"] == "customers.id"

    def test_non_fk_references_is_none(self) -> None:
        kb = _kb_from_schema(_table("orders", [_col("name", "TEXT")]))
        col = kb["schema"]["tables"][0]["columns"][0]
        assert col["references"] is None

    def test_foreign_keys_list_populated(self) -> None:
        kb = _kb_from_schema(
            _table(
                "orders",
                [_col("customer_id", "INTEGER", is_foreign_key=True, references="customers.id")],
            )
        )
        fks = kb["schema"]["foreign_keys"]
        assert len(fks) == 1
        assert fks[0]["from"] == "orders.customer_id"
        assert fks[0]["to"] == "customers.id"

    def test_foreign_keys_empty_when_no_fks(self) -> None:
        kb = _kb_from_schema(_table("orders", [_col("id", "INTEGER", is_primary_key=True)]))
        assert kb["schema"]["foreign_keys"] == []

    def test_column_samples_stored(self) -> None:
        samples = {"orders": {"status": ["pending", "shipped", "cancelled"]}}
        kb = _kb_from_schema(_table("orders", [_col("status", "TEXT")]), column_samples=samples)
        col = kb["schema"]["tables"][0]["columns"][0]
        assert col.get("samples") == ["pending", "shipped", "cancelled"]

    def test_samples_capped_at_five(self) -> None:
        samples = {"orders": {"status": ["a", "b", "c", "d", "e", "f"]}}
        kb = _kb_from_schema(_table("orders", [_col("status", "TEXT")]), column_samples=samples)
        col = kb["schema"]["tables"][0]["columns"][0]
        assert len(col["samples"]) == 5

    def test_pii_column_samples_suppressed(self) -> None:
        samples = {"users": {"email": ["a@b.com", "c@d.com"]}}
        kb = _kb_from_schema(_table("users", [_col("email", "TEXT")]), column_samples=samples)
        col = kb["schema"]["tables"][0]["columns"][0]
        assert "samples" not in col

    def test_no_samples_field_when_none_provided(self) -> None:
        kb = _kb_from_schema(_table("orders", [_col("status", "TEXT")]))
        col = kb["schema"]["tables"][0]["columns"][0]
        assert "samples" not in col

    def test_multiple_fks_across_tables(self) -> None:
        orders_table = _table(
            "orders",
            [_col("customer_id", "INTEGER", is_foreign_key=True, references="customers.id")],
        )
        items_table = _table(
            "order_items",
            [_col("order_id", "INTEGER", is_foreign_key=True, references="orders.id")],
        )
        kb = _kb_from_schema(orders_table, items_table)
        fks = kb["schema"]["foreign_keys"]
        assert len(fks) == 2


# ---------------------------------------------------------------------------
# _is_pii_column
# ---------------------------------------------------------------------------


class TestIsPiiColumn:
    def test_email_is_pii(self) -> None:
        assert _is_pii_column("user_email")

    def test_password_is_pii(self) -> None:
        assert _is_pii_column("password_hash")

    def test_status_is_not_pii(self) -> None:
        assert not _is_pii_column("status")

    def test_order_total_is_not_pii(self) -> None:
        assert not _is_pii_column("order_total")

    def test_ssn_is_pii(self) -> None:
        assert _is_pii_column("ssn")

    def test_credit_card_is_pii(self) -> None:
        assert _is_pii_column("credit_card_number")

    def test_token_is_pii(self) -> None:
        assert _is_pii_column("api_token")


# ---------------------------------------------------------------------------
# _render_m_schema()
# ---------------------------------------------------------------------------


class TestRenderMSchema:
    def _simple_kb(
        self,
        table_name: str = "orders",
        desc: str = "",
        columns: list[dict[str, Any]] | None = None,
        foreign_keys: list[dict[str, str]] | None = None,
        db_name: str = "testdb",
    ) -> dict[str, Any]:
        cols = columns or [
            {
                "name": "id",
                "type": "INTEGER",
                "is_primary_key": False,
                "is_foreign_key": False,
                "references": None,
            }
        ]
        return {
            "kb_version": 2,
            "db_name": db_name,
            "schema": {
                "tables": [{"name": table_name, "description": desc, "columns": cols}],
                "foreign_keys": foreign_keys or [],
            },
        }

    def test_db_id_line_present(self) -> None:
        kb = self._simple_kb(db_name="sales")
        result = _render_m_schema(kb)
        assert "【DB_ID】 sales" in result

    def test_table_header_present(self) -> None:
        kb = self._simple_kb(table_name="orders")
        result = _render_m_schema(kb)
        assert "【Table】 orders" in result

    def test_table_description_in_header(self) -> None:
        kb = self._simple_kb(desc="one row per order")
        result = _render_m_schema(kb)
        assert "— one row per order" in result

    def test_pk_flag_present(self) -> None:
        kb = self._simple_kb(
            columns=[
                {
                    "name": "id",
                    "type": "BIGINT",
                    "is_primary_key": True,
                    "is_foreign_key": False,
                    "references": None,
                }
            ]
        )
        result = _render_m_schema(kb)
        assert "PK" in result
        assert "(id:BIGINT, PK)" in result

    def test_fk_flag_present(self) -> None:
        kb = self._simple_kb(
            columns=[
                {
                    "name": "customer_id",
                    "type": "BIGINT",
                    "is_primary_key": False,
                    "is_foreign_key": True,
                    "references": "customers.id",
                }
            ]
        )
        result = _render_m_schema(kb)
        assert "FK->customers.id" in result

    def test_samples_in_output(self) -> None:
        kb = self._simple_kb(
            columns=[
                {
                    "name": "status",
                    "type": "TEXT",
                    "is_primary_key": False,
                    "is_foreign_key": False,
                    "references": None,
                    "samples": ["pending", "shipped"],
                }
            ]
        )
        result = _render_m_schema(kb)
        assert "samples:" in result
        assert "'pending'" in result
        assert "'shipped'" in result

    def test_foreign_keys_section_present(self) -> None:
        kb = self._simple_kb(foreign_keys=[{"from": "orders.customer_id", "to": "customers.id"}])
        result = _render_m_schema(kb)
        assert "【Foreign keys】" in result
        assert "orders.customer_id = customers.id" in result

    def test_no_foreign_keys_section_when_empty(self) -> None:
        kb = self._simple_kb(foreign_keys=[])
        result = _render_m_schema(kb)
        assert "【Foreign keys】" not in result

    def test_empty_tables_returns_empty_string(self) -> None:
        kb = {"db_name": "db", "schema": {"tables": [], "foreign_keys": []}}
        assert _render_m_schema(kb) == ""

    def test_v1_kb_no_pk_fk_fields_renders_gracefully(self) -> None:
        """v1 KBs lack is_primary_key/is_foreign_key — must not crash."""
        kb = {
            "schema": {
                "tables": [
                    {
                        "name": "orders",
                        "columns": [{"name": "id", "type": "INTEGER"}],
                    }
                ]
            }
        }
        result = _render_m_schema(kb)
        assert "【Table】 orders" in result
        assert "(id:INTEGER)" in result

    def test_column_without_flags_rendered_cleanly(self) -> None:
        kb = self._simple_kb(
            columns=[
                {
                    "name": "name",
                    "type": "TEXT",
                    "is_primary_key": False,
                    "is_foreign_key": False,
                    "references": None,
                }
            ]
        )
        result = _render_m_schema(kb)
        assert "(name:TEXT)" in result
        assert "PK" not in result
        assert "FK" not in result


# ---------------------------------------------------------------------------
# assemble_prompt() — schema format switching
# ---------------------------------------------------------------------------


class TestAssemblePromptSchemaFormat:
    def _kb(self) -> dict[str, Any]:
        return {
            "kb_version": 2,
            "db_name": "sales",
            "schema": {
                "tables": [
                    {
                        "name": "orders",
                        "description": "One row per order",
                        "row_count": 5000,
                        "columns": [
                            {
                                "name": "id",
                                "type": "INTEGER",
                                "description": "PK",
                                "is_primary_key": True,
                                "is_foreign_key": False,
                                "references": None,
                            },
                            {
                                "name": "status",
                                "type": "TEXT",
                                "description": "",
                                "is_primary_key": False,
                                "is_foreign_key": False,
                                "references": None,
                                "samples": ["pending", "shipped"],
                            },
                        ],
                    }
                ],
                "foreign_keys": [],
            },
            "business_context": {"glossary": [], "rules": []},
            "query_capsules": [],
        }

    def test_compact_format_uses_m_schema_markers(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "compact"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "【DB_ID】" in prompt.static_system
        assert "【Table】" in prompt.static_system

    def test_compact_format_is_default(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "compact"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "【Table】 orders" in prompt.static_system

    def test_verbose_format_uses_markdown_headers(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "verbose"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "### Table: orders" in prompt.static_system
        assert "【DB_ID】" not in prompt.static_system

    def test_compact_includes_pk_flag(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "compact"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "PK" in prompt.static_system

    def test_compact_includes_sample_values(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "compact"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "pending" in prompt.static_system

    def test_verbose_still_has_all_columns(self) -> None:
        with patch("nlqueries.config.SCHEMA_FORMAT", "verbose"):
            prompt = assemble_prompt("How many orders?", self._kb())
        assert "status" in prompt.static_system
        assert "id" in prompt.static_system
