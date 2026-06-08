"""
nlqueries.processing.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full Query History Processor pipeline: extract → filter → cluster → parameterize.

Public API
----------
``process_query_history(connector, schema, days, min_executions, annotate)``
    Run all pipeline stages and return a list of ``QueryCapsule`` objects.
    When ``annotate=True`` an LLM client is obtained via ``get_llm_client()``
    and each capsule's ``intent`` field is filled before returning.

``save_capsules(capsules, connector_id)``
    Serialise capsules to ``~/.nlqueries/capsules/{connector_id}.json``
    and return the output path.

``load_capsules(connector_id)``
    Deserialise and return capsules previously saved by ``save_capsules()``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from nlqueries.config import CAPSULES_DIR
from nlqueries.connectors.base import DatabaseConnector, SchemaSpec
from nlqueries.processing.parameterizer import Placeholder, QueryCapsule, parameterize_clusters
from nlqueries.processing.query_clusterer import cluster_queries
from nlqueries.processing.query_filter import filter_and_deduplicate


def process_query_history(
    connector: DatabaseConnector,
    schema: SchemaSpec | None = None,
    days: int = 90,
    min_executions: int = 1,
    annotate: bool = False,
) -> list[QueryCapsule]:
    """Run the full Query History Processor pipeline.

    Stages
    ------
    1. **Extract** — ``connector.extract_query_history(days)``
    2. **Filter + normalize** — ``filter_and_deduplicate(records, min_executions)``
    3. **Cluster** — ``cluster_queries(normalized)``
    4. **Parameterize** — ``parameterize_clusters(clusters, schema)``
    5. **Annotate** (optional) — fill ``QueryCapsule.intent`` via LLM

    Args:
        connector:       A connected ``DatabaseConnector`` instance.
        schema:          Optional ``SchemaSpec`` used to refine placeholder types.
                         When ``None`` string literals default to ``VARCHAR``.
        days:            Number of days of query history to process.
        min_executions:  Minimum execution count; lower-frequency queries are dropped.
        annotate:        When ``True``, call the LLM annotator (via ``get_llm_client()``)
                         to fill ``capsule.intent`` for every capsule before returning.

    Returns:
        ``list[QueryCapsule]`` sorted by frequency descending, capped at 1 000.
    """
    records = connector.extract_query_history(days=days)
    normalized = filter_and_deduplicate(records, min_executions=min_executions)
    clusters = cluster_queries(normalized)
    capsules = parameterize_clusters(clusters, schema=schema)
    if annotate and capsules:
        from nlqueries.llm import get_llm_client  # deferred — LLM deps optional at import time
        from nlqueries.processing.intent_annotator import annotate_capsules

        llm = get_llm_client()
        annotate_capsules(capsules, llm)
    return capsules


def save_capsules(capsules: list[QueryCapsule], connector_id: str) -> Path:
    """Serialise *capsules* to ``CAPSULES_DIR/{connector_id}.json``.

    The connector_id is sanitised so it is safe to use as a filename
    (colons and slashes are replaced with underscores).

    Args:
        capsules:     Capsules produced by ``process_query_history()``.
        connector_id: Identifier used as the output filename stem.

    Returns:
        Absolute ``Path`` of the written file.
    """
    CAPSULES_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w.-]", "_", connector_id)
    out_path = CAPSULES_DIR / f"{safe_id}.json"
    data = [dataclasses.asdict(c) for c in capsules]
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_capsules(connector_id: str) -> list[QueryCapsule]:
    """Deserialise capsules previously saved by ``save_capsules()``.

    Args:
        connector_id: The same identifier used when calling ``save_capsules()``.

    Returns:
        ``list[QueryCapsule]`` with the persisted intent values (may be empty
        strings if annotation has not been run yet).

    Raises:
        FileNotFoundError: When no saved capsules exist for *connector_id*.
    """
    safe_id = re.sub(r"[^\w.-]", "_", connector_id)
    path = CAPSULES_DIR / f"{safe_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved capsules for connector '{connector_id}'. "
            "Run 'nlqueries process-history' first."
        )
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [
        QueryCapsule(
            template_sql=item["template_sql"],
            placeholders=[Placeholder(**p) for p in item.get("placeholders", [])],
            tables=item["tables"],
            columns=item["columns"],
            frequency=item["frequency"],
            auto_description=item["auto_description"],
            intent=item.get("intent", ""),
        )
        for item in data
    ]
