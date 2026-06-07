"""
nlqueries.processing.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full Query History Processor pipeline: extract → filter → cluster → parameterize.

Public API
----------
``process_query_history(connector, schema, days, min_executions)``
    Run all four stages and return a list of ``QueryCapsule`` objects.

``save_capsules(capsules, connector_id)``
    Serialise capsules to ``~/.nlqueries/capsules/{connector_id}.json``
    and return the output path.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from nlqueries.config import CAPSULES_DIR
from nlqueries.connectors.base import DatabaseConnector, SchemaSpec
from nlqueries.processing.parameterizer import QueryCapsule, parameterize_clusters
from nlqueries.processing.query_clusterer import cluster_queries
from nlqueries.processing.query_filter import filter_and_deduplicate


def process_query_history(
    connector: DatabaseConnector,
    schema: SchemaSpec | None = None,
    days: int = 90,
    min_executions: int = 1,
) -> list[QueryCapsule]:
    """Run the full Query History Processor pipeline.

    Stages
    ------
    1. **Extract** — ``connector.extract_query_history(days)``
    2. **Filter + normalize** — ``filter_and_deduplicate(records, min_executions)``
    3. **Cluster** — ``cluster_queries(normalized)``
    4. **Parameterize** — ``parameterize_clusters(clusters, schema)``

    Args:
        connector:       A connected ``DatabaseConnector`` instance.
        schema:          Optional ``SchemaSpec`` used to refine placeholder types.
                         When ``None`` string literals default to ``VARCHAR``.
        days:            Number of days of query history to process.
        min_executions:  Minimum execution count; lower-frequency queries are dropped.

    Returns:
        ``list[QueryCapsule]`` sorted by frequency descending, capped at 1 000.
        The ``intent`` field of every capsule is empty — it is filled by the
        LLM annotator in a later stage (Sprint 4).
    """
    records = connector.extract_query_history(days=days)
    normalized = filter_and_deduplicate(records, min_executions=min_executions)
    clusters = cluster_queries(normalized)
    return parameterize_clusters(clusters, schema=schema)


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
