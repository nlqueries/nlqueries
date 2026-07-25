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


@dataclass
class DbtTableDoc:
    """dbt documentation for one model → one table in the KB."""

    description: str = ""
    #: column name → description (only non-empty descriptions are kept)
    columns: dict[str, str] = field(default_factory=dict)


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


def parse_dbt_docs(manifest: dict[str, Any] | Path | str) -> dict[str, DbtTableDoc]:
    """Model + column descriptions from a dbt ``manifest.json``.

    Accepts a parsed dict or a path. Returns ``{table_name: DbtTableDoc}`` for
    every model that carries at least one non-empty description (a model with no
    docs at all is omitted, so the result is exactly the docs dbt actually has).
    Whitespace-only descriptions are treated as empty.
    """
    data = manifest if isinstance(manifest, dict) else _load(manifest)
    docs: dict[str, DbtTableDoc] = {}

    for node in (data.get("nodes") or {}).values():
        if not isinstance(node, dict) or node.get("resource_type") != _MODEL:
            continue
        table = _model_table_name(node)
        if not table:
            continue
        description = str(node.get("description") or "").strip()
        columns: dict[str, str] = {}
        for col_name, col in (node.get("columns") or {}).items():
            if isinstance(col, dict):
                desc = str(col.get("description") or "").strip()
                if desc:
                    columns[str(col_name)] = desc
        if description or columns:
            docs[table] = DbtTableDoc(description=description, columns=columns)
    return docs


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
    p = Path(path)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    docs = parse_dbt_docs(manifest_path) if manifest_path.exists() else {}

    metrics: list[DbtMetric] = []
    if p.is_dir():
        sem_path = p / "semantic_manifest.json"
        if sem_path.exists():
            metrics = parse_dbt_metrics(sem_path)
    return docs, metrics
