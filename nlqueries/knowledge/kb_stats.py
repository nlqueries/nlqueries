"""
nlqueries.knowledge.kb_stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Knowledge-base coverage and quality metrics.

``compute_kb_stats`` reads the YAML knowledge base, optionally compares it
against the live database schema, and returns a populated :class:`KBStats`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from nlqueries.connectors.base import DatabaseConnector

# Columns whose names are highly generic — the LLM is most likely to produce
# wrong SQL when these columns have no description.
_AMBIGUOUS_NAMES = frozenset(
    {"status", "type", "code", "value", "data", "flag", "name", "key", "text", "active"}
)

_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
_FROM_RE = re.compile(r"\bFROM\s+[\"'`]?(\w+)[\"'`]?", re.IGNORECASE)
_JOIN_TABLE_RE = re.compile(r"\bJOIN\s+[\"'`]?(\w+)[\"'`]?", re.IGNORECASE)


@dataclass
class TableCoverage:
    """Per-table coverage detail for ``--verbose`` output."""

    name: str
    row_count: int | None
    column_count: int
    columns_with_desc: int
    has_table_desc: bool


@dataclass
class KBStats:
    """Coverage and quality metrics computed from a knowledge-base YAML."""

    agent_id: str
    kb_path: Path
    kb_mtime: float | None = None

    # --- Schema coverage (KB side) ---
    kb_tables: int = 0
    kb_tables_with_desc: int = 0
    kb_columns: int = 0
    kb_columns_with_desc: int = 0
    kb_tables_with_samples: int = 0
    ambiguous_columns: int = 0

    # --- Schema coverage (live DB) — None when connector unavailable ---
    db_tables: int | None = None
    db_columns: int | None = None

    # --- Query coverage ---
    capsule_count: int = 0
    capsule_with_intent: int = 0
    joins_in_capsules: int = 0

    # --- Join coverage (requires live DB for FK declarations) ---
    fk_joins: int | None = None
    fk_joins_seen: int | None = None  # FK pairs that appear in at least one capsule

    # --- Quality signals ---
    feedback_total: int = 0
    feedback_thumbs_up: int = 0
    feedback_thumbs_down: int = 0
    feedback_corrections: int = 0
    cache_entries: int | None = None  # None when Qdrant unreachable

    # --- Per-table breakdown (always populated; used by --verbose) ---
    table_details: list[TableCoverage] = field(default_factory=list)


def _capsule_join_pairs(capsules: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Return sorted (left, right) table-name pairs for every JOIN in every capsule."""
    pairs: set[tuple[str, str]] = set()
    for cap in capsules:
        sql = (cap.get("template") or "").lower()
        from_m = _FROM_RE.search(sql)
        if not from_m:
            continue
        left = from_m.group(1)
        for m in _JOIN_TABLE_RE.finditer(sql):
            right = m.group(1)
            if right != left:
                pairs.add((min(left, right), max(left, right)))
    return pairs


def compute_kb_stats(
    agent_id: str,
    kb_path: Path,
    connector: DatabaseConnector | None = None,
) -> KBStats:
    """Compute :class:`KBStats` for *agent_id*.

    Parameters
    ----------
    agent_id:
        Connector ID that was used when the KB was exported.
    kb_path:
        Absolute path to the ``.yaml`` knowledge-base file.
    connector:
        An *already-connected* :class:`~nlqueries.connectors.base.DatabaseConnector`.
        When ``None``, live-database comparisons are skipped and reported as ``None``.
    """
    stats = KBStats(agent_id=agent_id, kb_path=kb_path)

    if not kb_path.exists():
        return stats

    stats.kb_mtime = os.path.getmtime(kb_path)
    kb: dict[str, Any] = yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}

    # -------------------------------------------------------------------------
    # Schema coverage — KB side
    # -------------------------------------------------------------------------
    tables: list[dict[str, Any]] = kb.get("schema", {}).get("tables", []) or []
    stats.kb_tables = len(tables)

    for tbl in tables:
        tname = tbl.get("name", "")
        has_tbl_desc = bool((tbl.get("description") or "").strip())
        if has_tbl_desc:
            stats.kb_tables_with_desc += 1
        if tbl.get("sample_rows"):
            stats.kb_tables_with_samples += 1

        cols: list[dict[str, Any]] = tbl.get("columns", []) or []
        stats.kb_columns += len(cols)

        cols_with_desc = 0
        for col in cols:
            cname = (col.get("name") or "").lower()
            has_desc = bool((col.get("description") or "").strip())
            if has_desc:
                stats.kb_columns_with_desc += 1
                cols_with_desc += 1
            if cname in _AMBIGUOUS_NAMES and not has_desc:
                stats.ambiguous_columns += 1

        stats.table_details.append(
            TableCoverage(
                name=tname,
                row_count=tbl.get("row_count"),
                column_count=len(cols),
                columns_with_desc=cols_with_desc,
                has_table_desc=has_tbl_desc,
            )
        )

    # -------------------------------------------------------------------------
    # Query coverage
    # -------------------------------------------------------------------------
    capsules: list[dict[str, Any]] = kb.get("query_capsules", []) or []
    stats.capsule_count = len(capsules)
    stats.capsule_with_intent = sum(1 for cap in capsules if (cap.get("intent") or "").strip())
    stats.joins_in_capsules = sum(
        len(_JOIN_RE.findall(cap.get("template") or "")) for cap in capsules
    )

    # -------------------------------------------------------------------------
    # Live DB counts + FK join analysis (optional)
    # -------------------------------------------------------------------------
    if connector is not None:
        try:
            live_spec = connector.extract_schema()
            stats.db_tables = len(live_spec.tables)
            stats.db_columns = sum(len(t.columns) for t in live_spec.tables)

            fk_pairs: set[tuple[str, str]] = set()
            for live_tbl in live_spec.tables:
                for live_col in live_tbl.columns:
                    if live_col.references:
                        ref_table = live_col.references.split(".")[0].lower()
                        left = live_tbl.name.lower()
                        fk_pairs.add((min(left, ref_table), max(left, ref_table)))
            stats.fk_joins = len(fk_pairs)
            stats.fk_joins_seen = len(fk_pairs & _capsule_join_pairs(capsules))
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------------
    try:
        from nlqueries.feedback.store import load_feedback  # noqa: PLC0415

        feedback = load_feedback(agent_id)
        stats.feedback_total = len(feedback)
        stats.feedback_thumbs_up = sum(1 for f in feedback if f.rating == "up")
        stats.feedback_thumbs_down = sum(1 for f in feedback if f.rating == "down")
        stats.feedback_corrections = sum(
            1 for f in feedback if f.rating == "down" and f.corrected_sql
        )
    except Exception:  # noqa: BLE001
        pass

    # -------------------------------------------------------------------------
    # Cache entry count (best-effort Qdrant probe)
    # -------------------------------------------------------------------------
    try:
        import json as _json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        from nlqueries.cache.semantic_cache import (  # noqa: PLC0415
            _SAFE_ID_RE,
            CACHE_COLLECTION_PREFIX,
        )
        from nlqueries.config import QDRANT_URL  # noqa: PLC0415

        safe_id = _SAFE_ID_RE.sub("_", agent_id)
        collection = f"{CACHE_COLLECTION_PREFIX}{safe_id}"
        with urllib.request.urlopen(f"{QDRANT_URL}/collections/{collection}", timeout=3) as resp:
            data = _json.loads(resp.read())
            stats.cache_entries = data.get("result", {}).get("vectors_count", 0)
    except Exception:  # noqa: BLE001
        pass

    return stats
