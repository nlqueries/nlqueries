"""
nlqueries.analysis.query_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Slow query detection and SQL query plan analysis.

After executing a SQL query, ``analyze_query()`` checks whether it ran slowly
and — for Postgres dialects — captures the EXPLAIN plan so developers can
diagnose performance issues.

Public API
----------
``QueryPlanEntry``
    A single node from the parsed EXPLAIN plan tree.
``QueryAnalysis``
    Full analysis result for one SQL execution.
``analyze_query``
    Analyse a completed SQL execution (timing + optional EXPLAIN).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from nlqueries.connectors.base import DatabaseConnector

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD_MS = 2000  # queries slower than this are "slow"

_EXPLAIN_SUPPORTED_DIALECTS = frozenset({"postgres", "postgresql"})

_LARGE_TABLE_ROW_THRESHOLD = 10_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueryPlanEntry:
    """A single node from the parsed EXPLAIN (FORMAT JSON) plan tree."""

    node_type: str
    estimated_rows: int | None
    actual_rows: int | None
    actual_time_ms: float | None
    cost: float | None
    details: dict[str, Any]


@dataclass
class QueryAnalysis:
    """Result of analysing a completed SQL query execution."""

    sql: str
    dialect: str
    execution_time_ms: int
    is_slow: bool
    plan: list[QueryPlanEntry] | None  # None if EXPLAIN failed or dialect unsupported
    warnings: list[str]  # e.g. ["Sequential scan on large table: orders"]
    recommendation: str | None  # LLM-generated suggestion when is_slow=True


# ---------------------------------------------------------------------------
# EXPLAIN JSON parsing
# ---------------------------------------------------------------------------

_PLAN_SKIP_KEYS = frozenset(
    {"Node Type", "Plan Rows", "Actual Rows", "Actual Total Time", "Total Cost", "Plans"}
)


def _parse_plan_node(node: dict[str, Any]) -> list[QueryPlanEntry]:
    """Recursively parse an EXPLAIN JSON plan node into a flat list."""
    entries: list[QueryPlanEntry] = []

    raw_estimated = node.get("Plan Rows")
    raw_actual = node.get("Actual Rows")
    raw_time = node.get("Actual Total Time")
    raw_cost = node.get("Total Cost")

    details: dict[str, Any] = {k: v for k, v in node.items() if k not in _PLAN_SKIP_KEYS}

    entries.append(
        QueryPlanEntry(
            node_type=node.get("Node Type", "Unknown"),
            estimated_rows=int(raw_estimated) if raw_estimated is not None else None,
            actual_rows=int(raw_actual) if raw_actual is not None else None,
            actual_time_ms=float(raw_time) if raw_time is not None else None,
            cost=float(raw_cost) if raw_cost is not None else None,
            details=details,
        )
    )

    for child in node.get("Plans", []):
        entries.extend(_parse_plan_node(child))

    return entries


def _extract_plan(rows: list[list[Any]]) -> list[QueryPlanEntry] | None:
    """Parse EXPLAIN (ANALYZE, FORMAT JSON) output rows into a QueryPlanEntry list."""
    if not rows:
        return None
    try:
        raw = rows[0][0]
        if isinstance(raw, list):
            plan_data: list[dict[str, Any]] = raw
        elif isinstance(raw, dict):
            plan_data = [raw]
        else:
            plan_data = json.loads(str(raw))

        if not plan_data:
            return None

        root_plan = plan_data[0].get("Plan", {})
        if not root_plan:
            return None
        return _parse_plan_node(root_plan)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to parse EXPLAIN JSON output", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Warning scanner
# ---------------------------------------------------------------------------


def _scan_for_warnings(plan: list[QueryPlanEntry]) -> list[str]:
    """Scan a flat plan list for common performance anti-patterns."""
    warnings: list[str] = []
    for entry in plan:
        node_type = entry.node_type

        if node_type == "Seq Scan":
            est = entry.estimated_rows or 0
            if est > _LARGE_TABLE_ROW_THRESHOLD:
                relation = entry.details.get("Relation Name", "unknown")
                warnings.append(f"Sequential scan on large table: {relation}")

        elif node_type == "Hash Join":
            loops = entry.details.get("Actual Loops")
            if loops is not None and int(loops) > 1:
                warnings.append(f"Hash Join with {loops} loops (consider index)")

        elif node_type == "Sort":
            sort_key = entry.details.get("Sort Key", [])
            key_str = ", ".join(sort_key) if isinstance(sort_key, list) else str(sort_key)
            if key_str:
                warnings.append(f"Sort on non-indexed columns: {key_str}")

    return warnings


# ---------------------------------------------------------------------------
# LLM recommendation
# ---------------------------------------------------------------------------

_RECOMMENDATION_SYSTEM = (
    "You are a SQL performance expert. "
    "Given a slow query, its execution time, and plan warnings, "
    "provide a concise optimization suggestion (≤ 3 sentences). "
    "Respond with only the suggestion text — no preamble."
)


def _generate_recommendation(sql: str, execution_time_ms: int, warnings: list[str]) -> str | None:
    try:
        from nlqueries.llm import get_llm_client  # noqa: PLC0415

        top_warnings = warnings[:3]
        warnings_text = "\n".join(f"- {w}" for w in top_warnings) if top_warnings else "(none)"
        user_prompt = (
            f"SQL query:\n{sql}\n\n"
            f"Execution time: {execution_time_ms} ms\n\n"
            f"Top warnings:\n{warnings_text}\n\n"
            "Provide a concise optimization suggestion (≤ 3 sentences)."
        )
        llm = get_llm_client()
        return llm.complete(_RECOMMENDATION_SYSTEM, user_prompt)
    except Exception:  # noqa: BLE001
        logger.debug("LLM recommendation failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_query(
    sql: str,
    dialect: str,
    execution_time_ms: int,
    connector: DatabaseConnector,
    generate_recommendation: bool = False,
) -> QueryAnalysis:
    """Analyse a completed SQL query execution.

    Steps:
    1. Mark the query as slow if ``execution_time_ms >= SLOW_QUERY_THRESHOLD_MS``.
    2. For Postgres: run ``EXPLAIN (ANALYZE, FORMAT JSON) {sql}`` via the connector
       and parse the JSON plan tree.  For other dialects: ``plan`` is ``None``.
    3. Scan the plan for known anti-patterns and populate ``warnings``.
    4. If ``is_slow`` and ``generate_recommendation=True``, call the LLM for an
       optimisation suggestion (at most the top 3 warnings are included).

    Args:
        sql: The SQL query that was executed (not the EXPLAIN wrapper).
        dialect: Database dialect — ``"postgres"`` / ``"postgresql"`` enable EXPLAIN;
            ``"snowflake"`` / ``"bigquery"`` set ``plan=None``.
        execution_time_ms: Wall-clock time the query took in milliseconds.
        connector: The database connector used to issue the EXPLAIN statement.
        generate_recommendation: When ``True`` and the query is slow, call the LLM
            to generate an optimisation suggestion.

    Returns:
        :class:`QueryAnalysis` with timing, plan, warnings, and optional
        recommendation fields populated.
    """
    is_slow = execution_time_ms >= SLOW_QUERY_THRESHOLD_MS

    plan: list[QueryPlanEntry] | None = None
    warnings: list[str] = []

    # --- Fetch and parse EXPLAIN plan (Postgres only) ---
    if dialect.lower() in _EXPLAIN_SUPPORTED_DIALECTS:
        try:
            result = connector.execute_query(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
            if not result.error:
                plan = _extract_plan(result.rows)
        except Exception:  # noqa: BLE001
            logger.debug("EXPLAIN execution failed; plan will be None", exc_info=True)

    # --- Scan plan for warnings ---
    if plan is not None:
        warnings = _scan_for_warnings(plan)

    # --- Optional LLM recommendation ---
    recommendation: str | None = None
    if is_slow and generate_recommendation:
        recommendation = _generate_recommendation(sql, execution_time_ms, warnings)

    return QueryAnalysis(
        sql=sql,
        dialect=dialect,
        execution_time_ms=execution_time_ms,
        is_slow=is_slow,
        plan=plan,
        warnings=warnings,
        recommendation=recommendation,
    )
