# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.processing.parameterizer import QueryCapsule

# Column names that strongly indicate PII — sample values are never included.
# No word boundaries: underscore-delimited names like `user_email` must match.
_PII_COLUMN_RE = re.compile(
    r"password|passwd|secret|token|hash|salt|ssn|"
    r"credit.?card|card.?num|cvv|dob|birth|email|phone|address",
    re.IGNORECASE,
)


def _is_pii_column(col_name: str) -> bool:
    return bool(_PII_COLUMN_RE.search(col_name))

# Column name suffixes that indicate surrogate/technical keys with no business meaning.
_SKIP_SUFFIXES = ("_id", "_key", "_uuid", "_hash", "_token", "_code", "_pk", "_fk")

# LLM descriptions matching these phrases are treated as uninformative and dropped.
_GENERIC_PHRASES = (
    "stores data",
    "contains value",
    "represents the",
    "holds the",
    "this column",
    "the column",
)

_MAX_DESC_WORDS = 15


def _should_skip_column(col: ColumnSpec) -> bool:
    """Return True for columns whose descriptions cannot be inferred from data."""
    if col.is_primary_key or col.is_foreign_key:
        return True
    return col.name.lower().endswith(_SKIP_SUFFIXES)


def _is_valid_description(desc: str) -> bool:
    """Return True if *desc* is a short, specific business description."""
    if not desc or not desc.strip():
        return False
    if len(desc.split()) > _MAX_DESC_WORDS:
        return False
    desc_lower = desc.lower()
    return not any(phrase in desc_lower for phrase in _GENERIC_PHRASES)


def describe_columns(
    table: TableSpec,
    sample_rows: list[list[Any]],
    col_names: list[str],
    llm: Any,
) -> dict[str, str]:
    """Ask the LLM to generate short descriptions for eligible columns in *table*.

    Columns that are primary keys, foreign keys, or have key-like name suffixes
    are skipped before the LLM call.  One LLM call is made per table with all
    eligible columns in a markdown table so the model has cross-column context.

    Args:
        table:       The ``TableSpec`` whose columns to describe.
        sample_rows: Raw rows returned by ``connector.execute_query()``.
        col_names:   Column names matching the order of values in *sample_rows*.
        llm:         An ``LLMClient`` instance (``complete`` method used).

    Returns:
        ``{col_name: description}`` for columns where a meaningful description
        was inferred.  Columns the LLM marked as opaque (empty string) or whose
        descriptions failed validation are omitted.
    """
    eligible = [col for col in table.columns if not _should_skip_column(col)]
    if not eligible:
        return {}

    col_idx = {name: i for i, name in enumerate(col_names)}

    # Collect up to 3 non-null sample values per eligible column.
    samples: dict[str, list[str]] = {}
    for col in eligible:
        idx = col_idx.get(col.name)
        if idx is not None:
            vals = [
                str(row[idx]) for row in sample_rows if idx < len(row) and row[idx] is not None
            ][:3]
            samples[col.name] = vals

    lines = [
        f"Table: {table.name}",
        "",
        "Describe each column in ≤ 10 words based on its name, type, and sample values.",
        'Reply ONLY with a JSON object mapping column names to descriptions. Use "" if you',
        "cannot infer business meaning (e.g. the values are opaque IDs or encoded strings).",
        "",
        "| column | type | sample values |",
        "|--------|------|---------------|",
    ]
    for col in eligible:
        vals_str = ", ".join(samples.get(col.name, [])[:3]) or "(no samples)"
        lines.append(f"| {col.name} | {col.type} | {vals_str} |")

    system = (
        "You are a database analyst writing brief business descriptions for database columns. "
        "Reply ONLY with a valid JSON object — no markdown fences, no explanation."
    )
    user = "\n".join(lines)

    try:
        raw = llm.complete(system, user, max_tokens=512)
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not match:
            return {}
        parsed: dict[str, Any] = json.loads(match.group())
    except Exception:  # noqa: BLE001
        return {}

    return {
        col.name: str(parsed[col.name]).strip()
        for col in eligible
        if col.name in parsed and _is_valid_description(str(parsed[col.name]).strip())
    }


def generate_knowledge_base(
    schema: SchemaSpec,
    capsules: list[QueryCapsule],
    agent_name: str,
    existing_kb: dict[str, Any] | None = None,
    embed: bool = False,
    llm_column_descriptions: dict[str, dict[str, str]] | None = None,
    column_samples: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Build a structured knowledge-base dict from a schema and query capsules.

    When *existing_kb* is supplied, manually written table and column
    descriptions found there take precedence over all other sources,
    preserving human edits across regenerations.

    When *llm_column_descriptions* is supplied (a ``{table_name: {col_name: desc}}``
    mapping produced by :func:`describe_columns`), those descriptions fill columns
    that have no existing manual description and no description from the DB schema.

    When *embed* is ``True``, table and column descriptions are also upserted
    into the Qdrant collection ``agent_{agent_name}_schema`` (Qdrant must be
    reachable).

    When *column_samples* is supplied (``{table: {col: [val, ...]}}``), up to
    five non-PII sample values are stored per column for use by the M-Schema
    renderer.  PII columns (name matches :data:`_PII_COLUMN_RE`) are silently
    skipped.
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

    llm_descs = llm_column_descriptions or {}
    col_samples_map = column_samples or {}

    # Collect foreign-key relationships for the M-Schema FK section.
    foreign_keys: list[dict[str, str]] = []

    tables = []
    for table in schema.tables:
        table_desc = existing_table_descs.get(table.name) or table.description or ""
        columns = []
        for col in table.columns:
            col_desc = (
                existing_col_descs.get(table.name, {}).get(col.name)
                or col.description
                or llm_descs.get(table.name, {}).get(col.name)
                or ""
            )
            col_dict: dict[str, Any] = {
                "name": col.name,
                "type": col.type,
                "description": col_desc,
                "is_primary_key": col.is_primary_key,
                "is_foreign_key": col.is_foreign_key,
                "references": col.references,
            }
            # Sample values: store up to 5 for non-PII columns.
            if not _is_pii_column(col.name):
                raw_samples = col_samples_map.get(table.name, {}).get(col.name, [])
                if raw_samples:
                    col_dict["samples"] = [str(s) for s in raw_samples[:5]]

            if col.is_foreign_key and col.references:
                foreign_keys.append(
                    {"from": f"{table.name}.{col.name}", "to": col.references}
                )

            columns.append(col_dict)
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
        "kb_version": 2,
        "db_name": schema.database,
        "schema": {
            "tables": tables,
            "foreign_keys": foreign_keys,
        },
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
