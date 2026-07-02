"""
nlqueries.feedback.promoter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Promotes positively-rated feedback into the ``agent_{id}_verified`` Qdrant
collection so that future prompt assemblies can blend user-confirmed correct
examples into the dynamic context.

Public API
----------
``promote_feedback(agent_id) -> int``
    Load feedback for *agent_id*, filter for positive ratings, validate each
    SQL against the current KB schema, and upsert qualifying pairs to
    ``agent_{safe_id}_verified``.  Returns the count of newly upserted points.

The function is idempotent — the same (question, sql) pair always maps to the
same Qdrant point ID, so repeated calls produce the same collection state.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")

VERIFIED_COLLECTION_PREFIX = "agent_"
VERIFIED_COLLECTION_SUFFIX = "_verified"
VERIFIED_VECTOR_SIZE = 384


def _safe_agent_id(agent_id: str) -> str:
    return _SAFE_ID_RE.sub("_", agent_id)


def _verified_collection(agent_id: str) -> str:
    return f"{VERIFIED_COLLECTION_PREFIX}{_safe_agent_id(agent_id)}{VERIFIED_COLLECTION_SUFFIX}"


def _pair_point_id(question: str, sql: str) -> int:
    """Deterministic unsigned-64-bit Qdrant point ID for a (question, sql) pair."""
    digest = hashlib.sha256(f"{question}\x00{sql}".encode()).hexdigest()
    return int(digest[:16], 16)


def _load_kb(agent_id: str) -> dict[str, Any]:
    """Load the YAML knowledge base for *agent_id*, returning {} on failure."""
    try:
        import yaml  # noqa: PLC0415

        from nlqueries import config  # noqa: PLC0415

        safe_id = re.sub(r"[^\w.-]", "_", agent_id)
        kb_path = config.KB_PATH / f"{safe_id}.yaml"
        if kb_path.exists():
            return yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _sql_references_known_tables(sql: str, knowledge_base: dict[str, Any]) -> bool:
    """Return True when every table in *sql* appears in *knowledge_base*.

    Uses sqlglot for table extraction; returns True (permissive) when the KB
    has no schema or when sqlglot cannot parse the statement.
    """
    try:
        import sqlglot  # noqa: PLC0415
        import sqlglot.expressions as exp  # noqa: PLC0415
    except ImportError:
        return True

    schema_tables = {
        t.get("name", "").lower()
        for t in knowledge_base.get("schema", {}).get("tables", [])
        if t.get("name")
    }
    if not schema_tables:
        return True  # no schema to validate against

    try:
        stmt = sqlglot.parse_one(sql)
        if stmt is None:
            return False
        cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE) if cte.alias}
        ast_tables = {
            t.name.lower()
            for t in stmt.find_all(exp.Table)
            if t.name and t.name.lower() not in cte_names
        }
        return not (ast_tables - schema_tables)
    except Exception:  # noqa: BLE001
        return True  # parse error → be permissive


def promote_feedback(agent_id: str) -> int:
    """Promote positively-rated feedback pairs to the verified Qdrant collection.

    Only ``rating == "up"`` records with a non-empty SQL (``corrected_sql``
    preferred over ``generated_sql``) are considered.  Each SQL is validated
    against the current KB schema; pairs referencing removed tables are silently
    dropped.

    The upsert is idempotent — repeated calls with the same data produce the
    same collection state.

    Args:
        agent_id: The agent whose feedback file and KB to use.

    Returns:
        Number of (question, sql) pairs upserted in this call.
    """
    from qdrant_client.models import PointStruct  # noqa: PLC0415

    from nlqueries.embeddings.embedder import embed_text  # noqa: PLC0415
    from nlqueries.embeddings.qdrant_store import ensure_collection  # noqa: PLC0415
    from nlqueries.feedback.store import load_feedback  # noqa: PLC0415

    records = load_feedback(agent_id)
    positive = [r for r in records if r.rating == "up"]
    if not positive:
        return 0

    knowledge_base = _load_kb(agent_id)
    collection = _verified_collection(agent_id)

    # Dedup: (normalized_sql → first record seen)
    seen_sql: dict[str, bool] = {}
    candidates: list[tuple[str, str]] = []
    for rec in positive:
        sql = rec.corrected_sql or rec.generated_sql
        if not sql or not sql.strip():
            continue
        norm = sql.strip().lower()
        if norm in seen_sql:
            continue
        seen_sql[norm] = True
        if not _sql_references_known_tables(sql, knowledge_base):
            _log.debug("promote_feedback: skipping SQL referencing unknown tables: %.60s", sql)
            continue
        candidates.append((rec.question, sql))

    if not candidates:
        return 0

    ensure_collection(collection, VERIFIED_VECTOR_SIZE)

    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        from nlqueries import config as _cfg  # noqa: PLC0415

        client = QdrantClient(url=_cfg.QDRANT_URL)
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_feedback: Qdrant unavailable — %s", exc)
        return 0

    points: list[PointStruct] = []
    for question, sql in candidates:
        try:
            vector = embed_text(question)
        except Exception as exc:  # noqa: BLE001
            _log.warning("promote_feedback: embed failed for question %.60s — %s", question, exc)
            continue
        point_id = _pair_point_id(question, sql)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "question": question,
                    "sql": sql,
                    "agent_id": agent_id,
                    "verified": True,
                },
            )
        )

    if not points:
        return 0

    try:
        client.upsert(collection_name=collection, points=points)
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_feedback: upsert failed — %s", exc)
        return 0

    return len(points)
