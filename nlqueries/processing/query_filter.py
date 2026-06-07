"""
nlqueries.processing.query_filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First stage of the Query History Processor: filter, normalize, and deduplicate
raw query records extracted from a database's query history.

Filter rules (a query is discarded if any of the following are true):
  - The statement is not a SELECT
  - It references a system schema (information_schema, pg_catalog, sys.)
  - The token count is outside [3, 500]
  - It is a duplicate of another record after normalization

Normalization: parse + re-serialize via sqlglot for canonical whitespace and
keyword casing.

Fingerprinting: replace all literals (strings → '?', numbers → 0) to produce
a structural signature that groups queries differing only in literal values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

from nlqueries.connectors.base import QueryRecord

_SYSTEM_SCHEMA_RE = re.compile(
    r"\b(?:information_schema|pg_catalog)\b|\bsys\s*\.",
    re.IGNORECASE,
)

_MIN_TOKENS = 3
_MAX_TOKENS = 500


@dataclass
class NormalizedQuery:
    """A filtered, normalized query ready for the clustering stage."""

    original_sql: str
    normalized_sql: str
    fingerprint: str
    execution_count: int
    tables_referenced: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers (exposed for testing)
# ---------------------------------------------------------------------------


def _is_select(sql: str) -> bool:
    """Return True only if the top-level SQL statement is a SELECT."""
    try:
        parsed = sqlglot.parse_one(sql)
        return isinstance(parsed, exp.Select)
    except Exception:  # noqa: BLE001
        return False


def _has_system_schema(sql: str) -> bool:
    """Return True if *sql* references a known system/catalog schema."""
    return bool(_SYSTEM_SCHEMA_RE.search(sql))


def _token_count(sql: str) -> int:
    return len(sql.split())


def _normalize(sql: str) -> str | None:
    """Parse and re-serialize *sql* via sqlglot for a canonical representation.

    Returns None if the SQL cannot be parsed (the query will be discarded).
    """
    try:
        return sqlglot.parse_one(sql).sql(pretty=False)
    except Exception:  # noqa: BLE001
        return None


def _make_fingerprint(normalized_sql: str) -> str:
    """Strip all literals from *normalized_sql* to produce a structural fingerprint.

    Strings   → '?'
    Numbers   → 0

    Queries that differ only in their literal values will share the same
    fingerprint, enabling near-duplicate detection in the clustering stage.
    """

    def _strip(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal):
            return exp.Literal.string("?") if node.is_string else exp.Literal.number(0)
        return node

    try:
        stripped = sqlglot.parse_one(normalized_sql).transform(_strip)
        return stripped.sql(pretty=False)
    except Exception:  # noqa: BLE001
        return normalized_sql


def _extract_tables(normalized_sql: str) -> list[str]:
    """Return a sorted unique list of table names referenced in *normalized_sql*."""
    try:
        parsed = sqlglot.parse_one(normalized_sql)
        return sorted({t.name.lower() for t in parsed.find_all(exp.Table) if t.name})
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_and_deduplicate(
    records: list[QueryRecord],
    *,
    min_executions: int = 1,
) -> list[NormalizedQuery]:
    """Filter, normalize, and deduplicate a list of raw query records.

    Args:
        records: Raw query records from a connector's extract_query_history().
        min_executions: Discard records with fewer executions than this threshold.

    Returns:
        Deduplicated list of NormalizedQuery objects. When two records normalize
        to the same SQL, their execution_count values are summed.
    """
    seen: dict[str, NormalizedQuery] = {}

    for record in records:
        sql = record.sql.strip()

        if record.execution_count < min_executions:
            continue
        if not _is_select(sql):
            continue
        if _has_system_schema(sql):
            continue
        token_count = _token_count(sql)
        if token_count < _MIN_TOKENS or token_count > _MAX_TOKENS:
            continue

        normalized = _normalize(sql)
        if normalized is None:
            continue

        if normalized in seen:
            seen[normalized].execution_count += record.execution_count
            continue

        seen[normalized] = NormalizedQuery(
            original_sql=sql,
            normalized_sql=normalized,
            fingerprint=_make_fingerprint(normalized),
            execution_count=record.execution_count,
            tables_referenced=_extract_tables(normalized),
        )

    return list(seen.values())
