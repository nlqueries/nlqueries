"""
nlqueries.processing.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full Query History Processor pipeline: extract → filter → cluster → parameterize.

Public API
----------
``process_query_history(connector, schema, days, min_executions, annotate, embed)``
    Run all pipeline stages and return a list of ``QueryCapsule`` objects.
    When ``annotate=True`` an LLM client is obtained via ``get_llm_client()``
    and each capsule's ``intent`` field is filled before returning.
    When ``embed=True`` the capsules are upserted into the Qdrant vector store.

``save_capsules(capsules, connector_id)``
    Serialise capsules to ``~/.nlqueries/capsules/{connector_id}.json``
    and return the output path.

``load_capsules(connector_id)``
    Deserialise and return capsules previously saved by ``save_capsules()``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from nlqueries.config import CAPSULES_DIR
from nlqueries.connectors.base import DatabaseConnector, SchemaSpec
from nlqueries.processing.parameterizer import Placeholder, QueryCapsule, parameterize_clusters
from nlqueries.processing.query_clusterer import cluster_queries
from nlqueries.processing.query_filter import filter_and_deduplicate
from nlqueries.state_files import agent_storage_id, write_private


def process_query_history(
    connector: DatabaseConnector,
    schema: SchemaSpec | None = None,
    days: int = 90,
    min_executions: int = 1,
    limit: int = 500,
    annotate: bool = False,
    embed: bool = False,
    _filter_stats: dict[str, int] | None = None,
    *,
    agent_id: str | None = None,
) -> list[QueryCapsule]:
    """Run the full Query History Processor pipeline.

    Stages
    ------
    1. **Extract** — ``connector.extract_query_history(days)``
    2. **Filter + normalize** — ``filter_and_deduplicate(records, min_executions)``
    3. **Cluster** — ``cluster_queries(normalized)``
    4. **Parameterize** — ``parameterize_clusters(clusters, schema)``
    5. **Annotate** (optional) — fill ``QueryCapsule.intent`` via LLM
    6. **Embed** (optional) — make them *agent_id*'s signed capsules in its
       schema collection, where its questions search

    Args:
        connector:       A connected ``DatabaseConnector`` instance.
        schema:          Optional ``SchemaSpec`` used to refine placeholder types.
                         When ``None`` string literals default to ``VARCHAR``.
        days:            Number of days of query history to process.
        min_executions:  Minimum execution count; lower-frequency queries are dropped.
        limit:           Maximum number of useful queries to return after filtering.
        annotate:        When ``True``, call the LLM annotator (via ``get_llm_client()``)
                         to fill ``capsule.intent`` for every capsule before returning.
        embed:           When ``True``, replace *agent_id*'s capsules in its schema
                         collection (``agent_{id}_schema``) with these, signed, after
                         the annotation step (Qdrant is optional; defaults to
                         ``False``). Requires *agent_id*.
        _filter_stats:   Optional dict populated in-place with per-reason drop counts
                         from the filter stage (see ``filter_and_deduplicate``).
        agent_id:        The agent the capsules are for; required with *embed*.

    Raises:
        ValueError: *embed* without *agent_id*.

    Returns:
        ``list[QueryCapsule]`` sorted by frequency descending, capped at 1 000.
    """
    # Fetch more rows than needed so the cap can be applied *after* filtering
    # out DDL/system statements (SET, ALTER, pg_* queries, etc.).  A 3× factor
    # gives the filter stage enough headroom without fetching excessive data.
    if embed and not agent_id:
        # Capsules embedded for no agent went to a collection no question
        # searches; refusing is better than doing that silently.
        raise ValueError("embed=True needs agent_id: capsules are stored for one agent")
    raw_limit = min(limit * 3, 10_000)
    records = connector.extract_query_history(days=days, limit=raw_limit)
    normalized = filter_and_deduplicate(
        records, min_executions=min_executions, _stats=_filter_stats
    )
    if _filter_stats is not None:
        _filter_stats["useful"] = len(normalized)
    normalized = normalized[:limit]
    clusters = cluster_queries(normalized)
    capsules = parameterize_clusters(clusters, schema=schema)
    if annotate and capsules:
        from nlqueries.llm import get_llm_client  # deferred — LLM deps optional at import time
        from nlqueries.processing.intent_annotator import annotate_capsules

        llm = get_llm_client()
        annotate_capsules(capsules, llm)
    if embed and capsules and agent_id:
        from nlqueries.embeddings.qdrant_store import (  # deferred — Qdrant optional
            replace_capsules,
            schema_collection,
        )

        replace_capsules(schema_collection(agent_id), capsules, agent_id)
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
    safe_id = agent_storage_id(connector_id)
    out_path = CAPSULES_DIR / f"{safe_id}.json"
    data = [dataclasses.asdict(c) for c in capsules]
    return write_private(out_path, json.dumps(data, indent=2, ensure_ascii=False))


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
    safe_id = agent_storage_id(connector_id)
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
