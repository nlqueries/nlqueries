# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nlqueries.connectors.base import SchemaSpec
from nlqueries.processing.parameterizer import QueryCapsule


def generate_knowledge_base(
    schema: SchemaSpec,
    capsules: list[QueryCapsule],
    agent_name: str,
    existing_kb: dict[str, Any] | None = None,
    embed: bool = False,
) -> dict[str, Any]:
    """Build a structured knowledge-base dict from a schema and query capsules.

    When *existing_kb* is supplied, manually written table and column
    descriptions found there take precedence over the auto-generated ones
    from *schema*, preserving human edits across regenerations.

    When *embed* is ``True``, table and column descriptions are also upserted
    into the Qdrant collection ``agent_{agent_name}_schema`` (Qdrant must be
    reachable).
    """
    existing_table_descs: dict[str, str] = {}
    existing_col_descs: dict[str, dict[str, str]] = {}

    if existing_kb:
        for tbl in existing_kb.get("schema", {}).get("tables", []):
            tname = tbl.get("name", "")
            if tbl.get("description"):
                existing_table_descs[tname] = tbl["description"]
            col_descs: dict[str, str] = {}
            for col in tbl.get("columns", []):
                cname = col.get("name", "")
                if col.get("description"):
                    col_descs[cname] = col["description"]
            if col_descs:
                existing_col_descs[tname] = col_descs

    tables = []
    for table in schema.tables:
        table_desc = existing_table_descs.get(table.name) or table.description or ""
        columns = []
        for col in table.columns:
            col_desc = existing_col_descs.get(table.name, {}).get(col.name) or col.description or ""
            columns.append(
                {
                    "name": col.name,
                    "type": col.type,
                    "description": col_desc,
                }
            )
        tables.append(
            {
                "name": table.name,
                "description": table_desc,
                "row_count": table.row_count,
                "columns": columns,
            }
        )

    query_capsules = [
        {
            "intent": cap.intent,
            "template": cap.template_sql,
            "frequency": cap.frequency,
        }
        for cap in capsules
    ]

    kb: dict[str, Any] = {
        "schema": {"tables": tables},
        "business_context": {
            "glossary": [],
            "rules": [],
        },
        "query_capsules": query_capsules,
    }

    if embed and schema.tables:
        from nlqueries.embeddings.qdrant_store import ensure_collection, upsert_schema

        collection = f"agent_{agent_name}_schema"
        ensure_collection(collection)
        upsert_schema(collection, schema, agent_id=agent_name)

    return kb


def save_knowledge_base(kb: dict[str, Any], path: str) -> None:
    """Write *kb* to *path* as YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(kb, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
