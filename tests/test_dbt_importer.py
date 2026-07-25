"""
tests.test_dbt_importer
~~~~~~~~~~~~~~~~~~~~~~~~~
dbt artifact parsing + the merge-precedence it feeds (gap-bridging Block C).

Covers the parser (:mod:`nlqueries.knowledge.dbt_importer`) against the fixture
project, and the ``manual > dbt > schema > llm`` precedence in
``generate_knowledge_base`` — the rule that lets a dbt sync ground the KB without
ever overwriting a human edit.
"""

from __future__ import annotations

from pathlib import Path

from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.knowledge.dbt_importer import (
    load_dbt_artifacts,
    parse_dbt_docs,
    parse_dbt_metrics,
)
from nlqueries.knowledge.kb_generator import generate_knowledge_base

FIXTURES = Path(__file__).parent / "fixtures" / "dbt"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_docs_are_read_only_for_models() -> None:
    docs = parse_dbt_docs(FIXTURES / "manifest.json")
    # orders + customers are models; the seed and test nodes are ignored.
    assert set(docs) == {"orders", "customers"}
    assert docs["orders"].description == "One row per confirmed customer order."


def test_the_model_alias_is_used_as_the_table_name() -> None:
    docs = parse_dbt_docs(FIXTURES / "manifest.json")
    # model name is "customers_raw", alias is "customers" — the alias wins.
    assert "customers" in docs
    assert "customers_raw" not in docs


def test_blank_column_descriptions_are_dropped() -> None:
    cols = parse_dbt_docs(FIXTURES / "manifest.json")["orders"].columns
    assert cols["order_id"] == "Surrogate key for the order."
    assert "amount" not in cols  # empty description
    assert "status" not in cols  # whitespace-only description


def test_simple_metrics_resolve_to_agg_and_column() -> None:
    metrics = {m.name: m for m in parse_dbt_metrics(FIXTURES / "semantic_manifest.json")}
    assert (metrics["revenue"].agg, metrics["revenue"].column) == ("sum", "amount")
    assert (metrics["total_orders"].agg, metrics["total_orders"].column) == ("count", "order_id")


def test_a_derived_metric_keeps_its_expression_and_no_column() -> None:
    metrics = {m.name: m for m in parse_dbt_metrics(FIXTURES / "semantic_manifest.json")}
    derived = metrics["revenue_growth"]
    assert derived.column == ""
    assert derived.expression == "revenue / revenue_prev"


def test_load_artifacts_from_a_directory() -> None:
    docs, metrics = load_dbt_artifacts(FIXTURES)
    assert "orders" in docs
    assert len(metrics) == 3


def test_a_malformed_artifact_degrades_to_empty(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert parse_dbt_docs(bad) == {}
    assert parse_dbt_metrics(bad) == []


# ---------------------------------------------------------------------------
# Merge precedence in generate_knowledge_base
# ---------------------------------------------------------------------------


def _col(name: str, description: str = "") -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type="text",
        nullable=True,
        is_primary_key=False,
        is_foreign_key=False,
        references=None,
        description=description,
    )


def _schema() -> SchemaSpec:
    return SchemaSpec(
        database="db",
        tables=[
            TableSpec(
                name="orders",
                schema="public",
                row_count=10,
                columns=[
                    _col("order_id"),
                    _col("amount", "schema-desc for amount"),
                    _col("notes"),
                ],
                description=None,
            )
        ],
        extracted_at="2026-07-25T00:00:00Z",
    )


def _cols(kb: dict, table: str = "orders") -> dict[str, dict]:
    t = next(t for t in kb["schema"]["tables"] if t["name"] == table)
    return {c["name"]: c for c in t["columns"]}


def test_dbt_beats_schema_and_llm() -> None:
    docs = parse_dbt_docs(FIXTURES / "manifest.json")
    kb = generate_knowledge_base(
        _schema(),
        [],
        "a",
        dbt_docs=docs,
        llm_column_descriptions={"orders": {"order_id": "llm guess", "notes": "llm note"}},
    )
    cols = _cols(kb)
    # order_id: dbt doc present → dbt wins over the llm guess.
    assert cols["order_id"]["description"] == "Surrogate key for the order."
    assert cols["order_id"]["description_source"] == "dbt"
    # amount: no dbt doc, has a schema description → schema wins over nothing.
    assert cols["amount"]["description_source"] == "schema"
    # notes: only an llm guess → llm.
    assert cols["notes"]["description"] == "llm note"
    assert cols["notes"]["description_source"] == "llm"


def test_a_manual_edit_is_never_overwritten_by_dbt() -> None:
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "",
                    "columns": [
                        {
                            "name": "order_id",
                            "description": "Our hand-written definition.",
                            "description_source": "manual",
                        }
                    ],
                }
            ]
        }
    }
    kb = generate_knowledge_base(
        _schema(),
        [],
        "a",
        existing_kb=existing,
        dbt_docs=parse_dbt_docs(FIXTURES / "manifest.json"),
    )
    order_id = _cols(kb)["order_id"]
    assert order_id["description"] == "Our hand-written definition."
    assert order_id["description_source"] == "manual"


def test_a_legacy_description_without_a_source_is_treated_as_manual() -> None:
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "order_id", "description": "Legacy note, no source recorded."}
                    ],
                }
            ]
        }
    }
    kb = generate_knowledge_base(
        _schema(),
        [],
        "a",
        existing_kb=existing,
        dbt_docs=parse_dbt_docs(FIXTURES / "manifest.json"),
    )
    order_id = _cols(kb)["order_id"]
    assert order_id["description"] == "Legacy note, no source recorded."
    assert order_id["description_source"] == "manual"


def test_a_prior_dbt_field_is_updated_by_a_new_dbt_sync() -> None:
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {
                            "name": "order_id",
                            "description": "Old dbt text.",
                            "description_source": "dbt",
                        }
                    ],
                }
            ]
        }
    }
    kb = generate_knowledge_base(
        _schema(),
        [],
        "a",
        existing_kb=existing,
        dbt_docs=parse_dbt_docs(FIXTURES / "manifest.json"),
    )
    order_id = _cols(kb)["order_id"]
    # dbt-sourced → a fresh dbt doc overwrites it.
    assert order_id["description"] == "Surrogate key for the order."
    assert order_id["description_source"] == "dbt"


def test_no_dbt_docs_leaves_behaviour_unchanged() -> None:
    kb = generate_knowledge_base(_schema(), [], "a")
    cols = _cols(kb)
    assert cols["amount"]["description"] == "schema-desc for amount"
    assert cols["amount"]["description_source"] == "schema"
    assert cols["order_id"]["description"] == ""
    assert cols["order_id"]["description_source"] == ""
