"""
nlqueries.knowledge.dbt_importer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parse dbt artifacts into knowledge-base documentation (gap-bridging Block C).

dbt projects already carry the curation NLQueries wants: model and column
descriptions, and — in projects on the dbt Semantic Layer — metric definitions.
This reads those out of the standard artifacts with the **stdlib json only** (no
``dbt-core`` dependency), so ``generate_knowledge_base`` can ground the KB in the
customer's own dbt docs instead of, or ahead of, LLM-guessed descriptions.

Two things are extracted:

* :func:`parse_dbt_docs` — per-model and per-column descriptions from
  ``manifest.json`` (the source of truth for docs). Only ``resource_type ==
  "model"`` nodes are considered; seeds, tests and sources are ignored. The
  table name is the model's physical relation name (its ``alias`` when set),
  which is what the live-schema KB keys on.
* :func:`parse_dbt_metrics` — metric definitions from a Semantic Layer
  ``semantic_manifest.json``, resolved to a simple ``(name, agg, column)`` shape
  that enterprise maps to Nexus Beacons. Derived metrics (an expression over
  other metrics, no single column) are surfaced without a column so the caller
  can record them as review-only rather than guess a measure.

Merge precedence (Block C decision, applied in ``kb_generator``):
``manual edit > dbt doc > db schema > LLM-generated``. This module only reads
dbt; the ranking and the per-field ``description_source`` marker live in
``generate_knowledge_base``.

Pure and dependency-free: dicts/paths in, dataclasses out. Malformed artifacts
degrade to empty results rather than raising, so a bad file never breaks a build.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = "model"

#: Longest description kept from dbt, for a model or a column. dbt docs are
#: written for a docs site and can run to pages; the knowledge base wants what
#: the table or column is, and every prompt renders it again.
MAX_DESCRIPTION_CHARS = 2000


@dataclass
class DbtTableDoc:
    """dbt documentation for one model → one table in the KB."""

    description: str = ""
    #: column name → description (only non-empty descriptions are kept)
    columns: dict[str, str] = field(default_factory=dict)
    #: the model's physical table name (its alias when set)
    name: str = ""
    #: the schema dbt builds the model into, or "" when the manifest has none
    schema: str = ""


@dataclass
class DbtMetric:
    """A dbt Semantic Layer metric, resolved toward a Beacon.

    ``agg`` / ``column`` are populated for a *simple* metric backed by a single
    measure; a derived metric (an expression over other metrics) leaves them
    empty and carries its ``expression`` instead, so the caller can flag it for
    review rather than invent a column.
    """

    name: str
    label: str = ""
    agg: str = ""
    column: str = ""
    expression: str = ""
    entity: str = ""


def _load(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("Could not read dbt artifact %s.", path, exc_info=True)
        return {}


def _model_table_name(node: dict[str, Any]) -> str:
    """Physical table name for a model node: alias if set, else its name.

    dbt's ``alias`` is exactly the relation identifier the warehouse (and thus
    the live-schema KB) uses, so it wins over the model's logical ``name``.
    """
    alias = str(node.get("alias") or "").strip()
    if alias:
        return alias
    return str(node.get("name") or "").strip()


#: Appended to a description that was cut, inside the cap, so the model and a
#: reader of the knowledge base can tell a cut description from a whole one.
_CUT_MARKER = "..."


def _description(raw: Any) -> tuple[str, bool]:
    """A dbt description, stripped and cut to :data:`MAX_DESCRIPTION_CHARS`.

    Paired with whether it was cut. A cut description ends with
    :data:`_CUT_MARKER`, counted within the cap.
    """
    text = str(raw or "").strip()
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text, False
    return text[: MAX_DESCRIPTION_CHARS - len(_CUT_MARKER)].rstrip() + _CUT_MARKER, True


def parse_dbt_models(manifest: dict[str, Any] | Path | str) -> list[DbtTableDoc]:
    """Every documented model in a dbt ``manifest.json``, with its name and schema.

    Unlike :func:`parse_dbt_docs` this keeps two models that share a table name
    in different schemas apart, which is what :func:`match_dbt_docs` needs to
    give each knowledge-base table the docs of its own model.
    """
    data = manifest if isinstance(manifest, dict) else _load(manifest)
    models: list[DbtTableDoc] = []

    for node in (data.get("nodes") or {}).values():
        if not isinstance(node, dict) or node.get("resource_type") != _MODEL:
            continue
        table = _model_table_name(node)
        if not table:
            continue
        description, cut = _description(node.get("description"))
        columns: dict[str, str] = {}
        for col_name, col in (node.get("columns") or {}).items():
            if isinstance(col, dict):
                desc, col_cut = _description(col.get("description"))
                cut = cut or col_cut
                if desc:
                    columns[str(col_name)] = desc
        schema = str(node.get("schema") or "").strip()
        if cut:
            # Once per model, not per column: a model documented at length
            # usually is so throughout, and one line names it.
            logger.warning(
                "dbt model %s has descriptions longer than %d characters; they are "
                "cut and end with %r.",
                f"{schema}.{table}" if schema else table,
                MAX_DESCRIPTION_CHARS,
                _CUT_MARKER,
            )
        if description or columns:
            models.append(
                DbtTableDoc(description=description, columns=columns, name=table, schema=schema)
            )
    return models


def parse_dbt_docs(manifest: dict[str, Any] | Path | str) -> dict[str, DbtTableDoc]:
    """Model + column descriptions from a dbt ``manifest.json``.

    Accepts a parsed dict or a path. Returns ``{table_name: DbtTableDoc}`` for
    every model that carries at least one non-empty description (a model with no
    docs at all is omitted, so the result is exactly the docs dbt actually has).
    Whitespace-only descriptions are treated as empty, and each description is
    cut to :data:`MAX_DESCRIPTION_CHARS`. Keyed by table name alone: where two
    schemas hold a model of the same name, use :func:`parse_dbt_models`.
    """
    return {doc.name: doc for doc in parse_dbt_models(manifest)}


def match_dbt_docs(models: list[DbtTableDoc], tables: list[Any]) -> list[DbtTableDoc | None]:
    """The dbt docs for each knowledge-base table in *tables*, in order, or ``None``.

    A table that records its schema takes the model built into that schema with
    its name. Otherwise the bare name decides, as it did before schemas were compared, but
    only when one model has that name and no other table already took it by
    schema: ``staging.orders`` must not take the docs of ``analytics.orders``
    because both are called ``orders``. Both comparisons ignore letter case, as
    warehouses fold identifiers: Snowflake reports ``ORDERS`` for dbt's ``orders``.
    """
    by_relation: dict[tuple[str, str], DbtTableDoc] = {}
    by_name: dict[str, list[DbtTableDoc]] = {}
    for model in models:
        by_relation.setdefault((model.schema.lower(), model.name.lower()), model)
        by_name.setdefault(model.name.lower(), []).append(model)

    qualified: list[DbtTableDoc | None] = []
    for table in tables:
        if not isinstance(table, dict):
            qualified.append(None)
            continue
        schema = str(table.get("schema") or "").strip().lower()
        name = str(table.get("name") or "").lower()
        qualified.append(by_relation.get((schema, name)) if schema and name else None)

    claimed = {id(found) for found in qualified if found is not None}
    matched: list[DbtTableDoc | None] = []
    for table, found in zip(tables, qualified, strict=True):
        if found is not None or not isinstance(table, dict):
            matched.append(found)
            continue
        candidates = [
            m for m in by_name.get(str(table.get("name") or "").lower(), []) if id(m) not in claimed
        ]
        matched.append(candidates[0] if len(candidates) == 1 else None)
    return matched


def _measure_ref(type_params: dict[str, Any]) -> str:
    """The measure name a simple metric points at (``measure`` may be a str or {name})."""
    measure = type_params.get("measure")
    if isinstance(measure, dict):
        return str(measure.get("name") or "").strip()
    return str(measure or "").strip()


def parse_dbt_metrics(
    semantic_manifest: dict[str, Any] | Path | str,
) -> list[DbtMetric]:
    """Metric definitions from a dbt ``semantic_manifest.json``.

    Resolves each simple metric through its measure to an ``(agg, column)`` pair
    using the semantic models' measure definitions. A metric whose measure can't
    be resolved, or a derived metric, is returned with empty ``agg``/``column``
    and its ``expression`` set — never dropped, so the caller can surface it for
    review. Best-effort: a malformed file yields ``[]``.
    """
    data = semantic_manifest if isinstance(semantic_manifest, dict) else _load(semantic_manifest)

    # Build measure name → (agg, expr) across all semantic models.
    measures: dict[str, tuple[str, str]] = {}
    for model in data.get("semantic_models") or []:
        if not isinstance(model, dict):
            continue
        for measure in model.get("measures") or []:
            if isinstance(measure, dict) and measure.get("name"):
                measures[str(measure["name"])] = (
                    str(measure.get("agg") or "").strip(),
                    str(measure.get("expr") or "").strip(),
                )

    metrics: list[DbtMetric] = []
    for raw in data.get("metrics") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        name = str(raw["name"])
        label = str(raw.get("label") or "").strip()
        type_params = raw.get("type_params") if isinstance(raw.get("type_params"), dict) else {}
        measure_name = _measure_ref(type_params or {})
        if measure_name and measure_name in measures:
            agg, column = measures[measure_name]
            metrics.append(DbtMetric(name=name, label=label, agg=agg, column=column))
        else:
            # Derived / unresolved — carry the expression for review, no column.
            expr = str((type_params or {}).get("expr") or "").strip()
            metrics.append(DbtMetric(name=name, label=label, expression=expr))
    return metrics


def load_dbt_artifacts(path: Path | str) -> tuple[dict[str, DbtTableDoc], list[DbtMetric]]:
    """Load docs + metrics from a dbt artifacts directory or a single manifest file.

    *path* may be a directory containing ``manifest.json`` and (optionally)
    ``semantic_manifest.json``, or a path to a manifest file directly. Missing
    files simply yield empty results — a docs-only project (no semantic layer)
    returns metrics ``[]``, and vice versa.
    """
    docs = {doc.name: doc for doc in load_dbt_models(path)}
    return docs, load_dbt_metrics(path)


def load_dbt_metrics(path: Path | str) -> list[DbtMetric]:
    """The metrics of a dbt artifacts directory's ``semantic_manifest.json``;
    ``[]`` for a manifest file or a directory without one."""
    p = Path(path)
    sem_path = p / "semantic_manifest.json"
    return parse_dbt_metrics(sem_path) if p.is_dir() and sem_path.exists() else []


def load_dbt_models(path: Path | str) -> list[DbtTableDoc]:
    """The documented models (see :func:`parse_dbt_models`) of a dbt artifacts
    directory or a single manifest file; ``[]`` when there is no manifest."""
    p = Path(path)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    return parse_dbt_models(manifest_path) if manifest_path.exists() else []
