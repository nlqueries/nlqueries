"""
nlqueries.orchestrator.candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Self-consistency candidate generation for hard SQL queries (Phase 6A).

Public API
----------
``_is_hard(question, knowledge_base) -> bool``
    Heuristic: True when the question has ≥ 2 complexity signals OR the schema
    has more than 20 tables.  Used to gate candidate generation.

``generate_candidates(llm, system, user, n) -> list[str]``
    Fire *n* parallel ``acomplete()`` calls with varying temperatures and return
    the extracted SQL strings.  Cheap after prompt-cache warm-up (Phase 3A).

``select_best(candidates, knowledge_base, dialect) -> str``
    Dedupe → drop invalid → majority vote on canonical AST → tie-break by
    choosing the shortest statement.  Returns the first candidate (possibly
    invalid) when the list is empty or all candidates are invalid.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlqueries.llm.client import LLMClient

# ---------------------------------------------------------------------------
# Complexity signals used by _is_hard()
# ---------------------------------------------------------------------------

_HARD_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(r"\bjoin\b", re.IGNORECASE),
    re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    re.compile(r"\bhaving\b", re.IGNORECASE),
    re.compile(r"\bsubquer(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bwindow\s+function\b", re.IGNORECASE),
    re.compile(r"\brank(ing)?\b", re.IGNORECASE),
    re.compile(r"\bpartition\s+by\b", re.IGNORECASE),
    re.compile(r"\bpercent(age|ile)?\b", re.IGNORECASE),
    re.compile(r"\bcumulative\b", re.IGNORECASE),
    re.compile(r"\bpivot\b", re.IGNORECASE),
    re.compile(r"\bnested\b", re.IGNORECASE),
    re.compile(r"\bbetween\b", re.IGNORECASE),
]

_HARD_TABLE_THRESHOLD = 20

# Temperatures to use for the N candidate calls.  The first call uses None
# (provider default, typically 1.0) so it is identical to a non-consistency
# run — ensuring reproducibility when only 1 candidate is requested.
_CANDIDATE_TEMPERATURES: list[float | None] = [None, 0.4, 0.8, 0.2, 0.6]


def _is_hard(question: str, knowledge_base: dict[str, Any]) -> bool:
    """Return True when the query is likely to benefit from self-consistency.

    A query is "hard" when any of these hold:
    - The question text contains ≥ 2 complexity signal keywords (JOINs,
      GROUP BY, window functions, subqueries, etc.)
    - The schema has more than *_HARD_TABLE_THRESHOLD* tables (many-table
      schemas require more precise table selection).
    """
    signal_count = sum(1 for pat in _HARD_KEYWORDS if pat.search(question))
    if signal_count >= 2:
        return True
    table_count = len(
        knowledge_base.get("schema", {}).get("tables", [])
    )
    return table_count > _HARD_TABLE_THRESHOLD


async def generate_candidates(
    llm: LLMClient,
    system: str | list[dict],
    user: str,
    n: int = 3,
) -> list[str]:
    """Generate *n* SQL candidates in parallel with temperature variation.

    Each candidate is produced by a separate ``acomplete()`` call so the
    shared prompt-cache prefix (set up in Phase 3A) is hit on every call —
    only the short correction suffix is billed at full token cost.

    Args:
        llm:    LLM client; must support ``acomplete()``.
        system: System prompt (string or list of cache-control blocks).
        user:   User/correction message.
        n:      Number of candidates to generate (default 3).

    Returns:
        List of extracted SQL strings (length ≤ *n*; may be shorter if some
        calls raise exceptions).
    """
    from nlqueries.orchestrator.sql_generation import _extract_sql  # noqa: PLC0415

    temperatures = (_CANDIDATE_TEMPERATURES + [None] * n)[:n]

    async def _one(temperature: float | None) -> str:
        try:
            raw = await llm.acomplete(system, user, max_tokens=512, temperature=temperature)
            return _extract_sql(raw)
        except Exception:  # noqa: BLE001
            return ""

    results = await asyncio.gather(*[_one(t) for t in temperatures])
    return [r for r in results if r.strip()]


def select_best(
    candidates: list[str],
    knowledge_base: dict[str, Any],
    dialect: str,
) -> str:
    """Select the best SQL from *candidates* via majority vote on canonical ASTs.

    Algorithm:
    1. Normalize each candidate to a canonical SQL string via
       ``sqlglot.parse_one().sql(normalize=True, dialect=dialect)``.
       Drop candidates that cannot be parsed.
    2. Validate each normalized candidate with ``_validate_sql()``.
       Prefer valid candidates; fall back to all when none are valid.
    3. Majority vote: return the canonical form that appears most often.
    4. Tie-break: shortest canonical string (proxy for simplicity).

    Args:
        candidates:     Raw SQL strings from ``generate_candidates()``.
        knowledge_base: For ``_validate_sql()`` table-name checks.
        dialect:        Target SQL dialect.

    Returns:
        The winning SQL string, or the first candidate when all are empty /
        unparseable.
    """
    import sqlglot  # noqa: PLC0415

    from nlqueries.orchestrator.sql_generation import _validate_sql  # noqa: PLC0415

    if not candidates:
        return ""

    # Normalize: parse → canonical SQL
    normalized: list[tuple[str, str]] = []  # (canonical, original)
    for original in candidates:
        if not original.strip():
            continue
        try:
            stmt = sqlglot.parse_one(original, dialect=dialect)
            if stmt is None:
                continue
            canonical = stmt.sql(dialect=dialect, normalize=True)
            normalized.append((canonical, original))
        except Exception:  # noqa: BLE001
            continue

    if not normalized:
        return candidates[0]

    # Separate valid from invalid
    valid = [(c, o) for c, o in normalized if _validate_sql(c, knowledge_base, dialect) is None]
    pool = valid if valid else normalized

    # Majority vote
    vote: dict[str, int] = {}
    for canonical, _ in pool:
        vote[canonical] = vote.get(canonical, 0) + 1

    max_votes = max(vote.values())
    winners = [c for c, v in vote.items() if v == max_votes]

    # Tie-break: shortest canonical string
    best_canonical = min(winners, key=len)

    # Return the original SQL corresponding to the winning canonical form
    for canonical, original in pool:
        if canonical == best_canonical:
            return original

    return candidates[0]
