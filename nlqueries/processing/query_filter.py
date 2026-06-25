"""
nlqueries.processing.query_filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First stage of the Query History Processor: filter, normalize, and deduplicate
raw query records extracted from a database's query history.

Filter rules (a query is discarded if any of the following are true):
  - The statement is a non-SQL system/driver command (SHOW, SET, RESET, …)
  - The statement is not a SELECT
  - It references a system schema/table (information_schema, pg_catalog, pg_type, …)
  - It only calls PostgreSQL connection-info functions (CURRENT_DATABASE, etc.)
  - The token count is outside [3, 500]
  - It is a duplicate of another record after normalization

Normalization: parse + re-serialize via sqlglot for canonical whitespace and
keyword casing. PostgreSQL positional parameters ($1, $2, …) are substituted
with 0 before parsing so pg_stat_statements output is handled correctly.

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

# Extended system-table pattern: catches pg_* catalog tables accessed directly
# (without the pg_catalog. prefix) by SQLAlchemy/psycopg2 driver handshakes.
_SYSTEM_TABLE_RE = re.compile(
    r"\b(pg_type|pg_namespace|pg_extension|pg_user|pg_depend|pg_class|"
    r"pg_stat_statements|pg_postmaster_start_time|pg_catalog\.|"
    r"information_schema\.)\b",
    re.IGNORECASE,
)

# Matches SELECT statements that contain only PostgreSQL session/connection
# functions with no FROM clause (pure driver handshake queries).
_SYSTEM_FUNCTION_ONLY_RE = re.compile(
    r"^\s*SELECT\s+(?:CURRENT_DATABASE|CURRENT_SCHEMAS?|CURRENT_USER|"
    r"VERSION|PG_POSTMASTER_START_TIME)(?:[\s($]|$)",
    re.IGNORECASE,
)

# Non-SQL commands emitted by PostgreSQL drivers that should be discarded
# before reaching sqlglot to avoid noisy "unsupported syntax" warnings.
_SYSTEM_PREFIXES = (
    "show ",
    "set ",
    "reset ",
    "begin",
    "commit",
    "rollback",
    "deallocate",
    "declare ",
    "close ",
    "fetch ",
    "discard ",
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


def _pg_params_to_literals(sql: str) -> str:
    """Replace pg_stat_statements positional params ($1, $2…) with 0.

    pg_stat_statements substitutes literal values with $N placeholders.
    sqlglot cannot always parse these (e.g. NTILE($1) expects a constant),
    so we normalize them to integer 0 before parsing.
    """
    return re.sub(r"\$\d+", "0", sql)


def _is_system_command(sql: str) -> bool:
    """Return True if *sql* is a non-SELECT system/driver command.

    Catches SHOW, SET, RESET, BEGIN, COMMIT, ROLLBACK, etc. — commands
    emitted by PostgreSQL drivers that would otherwise cause sqlglot to
    emit "unsupported syntax" warnings before the SELECT filter runs.
    """
    return sql.strip().lower().startswith(_SYSTEM_PREFIXES)


def _is_system_query(sql: str) -> bool:
    """Return True if *sql* only touches PostgreSQL system objects.

    Catches:
    - Queries referencing pg_* catalog tables (e.g. pg_type, pg_extension)
      accessed without the pg_catalog. prefix by driver handshakes.
    - Pure session-function SELECTs (CURRENT_DATABASE, CURRENT_USER, etc.)
      that have no FROM clause and carry no business meaning.
    """
    return bool(_SYSTEM_TABLE_RE.search(sql) or _SYSTEM_FUNCTION_ONLY_RE.match(sql))


def _is_select(sql: str) -> bool:
    """Return True only if the top-level SQL statement is a SELECT."""
    try:
        parsed = sqlglot.parse_one(_pg_params_to_literals(sql))
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
        return sqlglot.parse_one(_pg_params_to_literals(sql)).sql(pretty=False)
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
    _stats: dict[str, int] | None = None,
) -> list[NormalizedQuery]:
    """Filter, normalize, and deduplicate a list of raw query records.

    Args:
        records: Raw query records from a connector's extract_query_history().
        min_executions: Discard records with fewer executions than this threshold.
        _stats: Optional dict populated in-place with drop counts per reason.
            Keys: considered, too_few_executions, system_command, not_select,
            system_schema, token_count, normalize_failed, duplicate.

    Returns:
        Deduplicated list of NormalizedQuery objects. When two records normalize
        to the same SQL, their execution_count values are summed.
    """
    seen: dict[str, NormalizedQuery] = {}

    if _stats is not None:
        _stats["considered"] = len(records)
        for key in (
            "too_few_executions",
            "system_command",
            "not_select",
            "system_schema",
            "token_count",
            "normalize_failed",
            "duplicate",
        ):
            _stats.setdefault(key, 0)

    for record in records:
        sql = record.sql.strip()

        if record.execution_count < min_executions:
            if _stats is not None:
                _stats["too_few_executions"] += 1
            continue
        # Pre-filter system/driver commands before passing to sqlglot so it
        # never emits "unsupported syntax" warnings for SHOW/SET/etc.
        if _is_system_command(sql):
            if _stats is not None:
                _stats["system_command"] += 1
            continue
        if not _is_select(sql):
            if _stats is not None:
                _stats["not_select"] += 1
            continue
        if _has_system_schema(sql) or _is_system_query(sql):
            if _stats is not None:
                _stats["system_schema"] += 1
            continue
        token_count = _token_count(sql)
        if token_count < _MIN_TOKENS or token_count > _MAX_TOKENS:
            if _stats is not None:
                _stats["token_count"] += 1
            continue

        normalized = _normalize(sql)
        if normalized is None:
            if _stats is not None:
                _stats["normalize_failed"] += 1
            continue

        if normalized in seen:
            seen[normalized].execution_count += record.execution_count
            if _stats is not None:
                _stats["duplicate"] += 1
            continue

        seen[normalized] = NormalizedQuery(
            original_sql=sql,
            normalized_sql=normalized,
            fingerprint=_make_fingerprint(normalized),
            execution_count=record.execution_count,
            tables_referenced=_extract_tables(normalized),
        )

    return list(seen.values())
