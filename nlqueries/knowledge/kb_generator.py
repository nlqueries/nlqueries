# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.knowledge.concept_hierarchy import build_glossary_hierarchy
from nlqueries.processing.parameterizer import QueryCapsule

logger = logging.getLogger(__name__)

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

# Output budget for one describe-columns reply. It used to be a flat 512 tokens
# no matter how many columns were being described, which silently capped the
# feature at roughly twenty columns: a 34-column fact table got a reply cut off
# mid-word, the JSON never closed, and every description was discarded. The
# table came back "0 descriptions written" after a charged LLM call.
#
# A description is capped at fifteen words, so forty tokens per column is
# generous — observed replies run to about sixteen — and the ceiling keeps a
# pathologically wide table from asking for an unbounded completion.
_TOKENS_PER_COLUMN = 40
_MIN_DESCRIPTION_TOKENS = 512
_MAX_DESCRIPTION_TOKENS = 8192

#: One `"column": "description"` pair. Used to recover a reply that stopped
#: mid-object, where json.loads has nothing it can work with.
_JSON_PAIR_RE = re.compile(r'"([^"\\]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _description_token_budget(column_count: int) -> int:
    """Output tokens to allow for describing *column_count* columns."""
    scaled = 64 + _TOKENS_PER_COLUMN * column_count
    return max(_MIN_DESCRIPTION_TOKENS, min(_MAX_DESCRIPTION_TOKENS, scaled))


def _parse_descriptions(raw: str) -> tuple[dict[str, str], bool]:
    """Pull `{column: description}` out of an LLM reply.

    Returns ``(descriptions, recovered)``, where *recovered* means the reply was
    not valid JSON and the pairs were salvaged from it. A reply truncated by the
    token limit is well-formed right up to the byte it stops on; throwing all of
    it away for want of the closing brace is how a wide table produced nothing.
    """
    start = raw.find("{")
    if start == -1:
        return {}, False

    end = raw.rfind("}")
    if end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}, False

    return {k: v for k, v in _JSON_PAIR_RE.findall(raw[start:])}, True


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

    budget = _description_token_budget(len(eligible))
    try:
        raw = llm.complete(system, user, max_tokens=budget)
    except Exception:  # noqa: BLE001
        logger.warning(
            "describe_columns: the LLM call for %r failed; no descriptions written.",
            table.name,
            exc_info=True,
        )
        return {}

    parsed, recovered = _parse_descriptions(raw)
    if not parsed:
        # Say so. This returned an empty dict silently, so a caller reporting
        # what it wrote could only report zero, with nothing to explain it.
        logger.warning(
            "describe_columns: no usable JSON in the reply for %r "
            "(%d columns, %d-token budget, %d-char reply); no descriptions written.",
            table.name,
            len(eligible),
            budget,
            len(raw),
        )
        return {}
    if recovered:
        logger.warning(
            "describe_columns: the reply for %r was not valid JSON (likely cut off at "
            "the %d-token budget); recovered %d of %d columns.",
            table.name,
            budget,
            len(parsed),
            len(eligible),
        )

    return {
        col.name: str(parsed[col.name]).strip()
        for col in eligible
        if col.name in parsed and _is_valid_description(str(parsed[col.name]).strip())
    }


#: Provenance of a table/column description, ranked so a re-sync knows what it
#: may overwrite (Block C). ``manual`` is a human edit and is never overwritten;
#: a legacy description with no recorded source is treated as ``manual`` so an
#: upgrade or a first dbt sync never clobbers existing content.
SOURCE_MANUAL = "manual"
SOURCE_DBT = "dbt"
SOURCE_SCHEMA = "schema"
SOURCE_LLM = "llm"


def _resolve_description(
    *,
    existing_desc: str,
    existing_source: str,
    dbt_desc: str,
    schema_desc: str,
    llm_desc: str,
) -> tuple[str, str]:
    """Pick a description + its source under the Block C precedence.

    ``manual edit > dbt doc > db schema > LLM-generated``. A manual (or
    legacy-untracked) existing description wins outright. Otherwise the highest
    *fresh* source available is taken, so a dbt sync updates dbt/schema/llm
    fields but a dropped dbt doc falls back rather than sticking. When nothing
    fresh is available the prior value is preserved.
    """
    if existing_desc and existing_source in ("", SOURCE_MANUAL):
        return existing_desc, SOURCE_MANUAL
    if dbt_desc:
        return dbt_desc, SOURCE_DBT
    if schema_desc:
        return schema_desc, SOURCE_SCHEMA
    if llm_desc:
        return llm_desc, SOURCE_LLM
    if existing_desc:
        return existing_desc, existing_source or SOURCE_MANUAL
    return "", ""


def generate_knowledge_base(
    schema: SchemaSpec,
    capsules: list[QueryCapsule],
    agent_name: str,
    existing_kb: dict[str, Any] | None = None,
    embed: bool = False,
    llm_column_descriptions: dict[str, dict[str, str]] | None = None,
    column_samples: dict[str, dict[str, list[str]]] | None = None,
    dbt_docs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured knowledge-base dict from a schema and query capsules.

    When *existing_kb* is supplied, manually written table and column
    descriptions found there take precedence over all other sources,
    preserving human edits across regenerations.

    When *llm_column_descriptions* is supplied (a ``{table_name: {col_name: desc}}``
    mapping produced by :func:`describe_columns`), those descriptions fill columns
    that have no existing manual description and no description from the DB schema.

    When *dbt_docs* is supplied (``{table_name: DbtTableDoc}`` from
    :func:`nlqueries.knowledge.dbt_importer.parse_dbt_docs`), dbt model/column
    docs are merged in with source ``dbt`` — above the DB schema and LLM
    descriptions but below a human edit (Block C). Each description records its
    provenance in a ``description_source`` field so a later dbt sync updates only
    dbt-sourced fields.

    When *embed* is ``True``, table and column descriptions are also upserted
    into the Qdrant collection ``agent_{agent_name}_schema`` (Qdrant must be
    reachable).

    When *column_samples* is supplied (``{table: {col: [val, ...]}}``), up to
    five non-PII sample values are stored per column for use by the M-Schema
    renderer.  PII columns (name matches :data:`_PII_COLUMN_RE`) are silently
    skipped.
    """
    existing_table_descs: dict[str, str] = {}
    existing_table_srcs: dict[str, str] = {}
    existing_col_descs: dict[str, dict[str, str]] = {}
    existing_col_srcs: dict[str, dict[str, str]] = {}

    if existing_kb:
        for tbl in existing_kb.get("schema", {}).get("tables", []):
            tname = tbl.get("name", "")
            if tbl.get("description"):
                existing_table_descs[tname] = tbl["description"]
                existing_table_srcs[tname] = str(tbl.get("description_source") or "")
            col_descs: dict[str, str] = {}
            col_srcs: dict[str, str] = {}
            for col in tbl.get("columns", []):
                cname = col.get("name", "")
                if col.get("description"):
                    col_descs[cname] = col["description"]
                    col_srcs[cname] = str(col.get("description_source") or "")
            if col_descs:
                existing_col_descs[tname] = col_descs
                existing_col_srcs[tname] = col_srcs

    llm_descs = llm_column_descriptions or {}
    col_samples_map = column_samples or {}
    dbt = dbt_docs or {}

    # Collect foreign-key relationships for the M-Schema FK section.
    foreign_keys: list[dict[str, str]] = []

    tables = []
    for table in schema.tables:
        dbt_table = dbt.get(table.name)
        dbt_table_desc = getattr(dbt_table, "description", "") if dbt_table is not None else ""
        dbt_cols = getattr(dbt_table, "columns", {}) if dbt_table is not None else {}
        table_desc, table_src = _resolve_description(
            existing_desc=existing_table_descs.get(table.name, ""),
            existing_source=existing_table_srcs.get(table.name, ""),
            dbt_desc=dbt_table_desc,
            schema_desc=table.description or "",
            llm_desc="",
        )
        columns = []
        for col in table.columns:
            col_desc, col_src = _resolve_description(
                existing_desc=existing_col_descs.get(table.name, {}).get(col.name, ""),
                existing_source=existing_col_srcs.get(table.name, {}).get(col.name, ""),
                dbt_desc=dbt_cols.get(col.name, "") if isinstance(dbt_cols, dict) else "",
                schema_desc=col.description or "",
                llm_desc=llm_descs.get(table.name, {}).get(col.name, "") or "",
            )
            col_dict: dict[str, Any] = {
                "name": col.name,
                "type": col.type,
                "description": col_desc,
                "description_source": col_src,
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
                foreign_keys.append({"from": f"{table.name}.{col.name}", "to": col.references})

            columns.append(col_dict)
        tables.append(
            {
                "name": table.name,
                "description": table_desc,
                "description_source": table_src,
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

    # Preserve hand-authored business context (glossary + rules, incl. any
    # glossary `parent:` hierarchy) across regenerations — mirroring how manual
    # table/column descriptions are preserved above. A cycle in the preserved
    # glossary hierarchy is rejected here (CG-2.1); a flat glossary is unaffected.
    business_context: dict[str, Any] = {"glossary": [], "rules": []}
    if existing_kb and isinstance(existing_kb.get("business_context"), dict):
        prev_bc = existing_kb["business_context"]
        business_context = {
            "glossary": prev_bc.get("glossary") or [],
            "rules": prev_bc.get("rules") or [],
            **{k: v for k, v in prev_bc.items() if k not in ("glossary", "rules")},
        }
        build_glossary_hierarchy(business_context["glossary"], validate=True)

    kb: dict[str, Any] = {
        "kb_version": 2,
        "db_name": schema.database,
        "schema": {
            "tables": tables,
            "foreign_keys": foreign_keys,
        },
        "business_context": business_context,
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
