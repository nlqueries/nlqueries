"""
nlqueries.processing.query_clusterer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Second stage of the Query History Processor: cluster normalized queries into
representative groups ready for parameterization.

Algorithm
---------
1. **Primary clustering by fingerprint** — group NormalizedQuery objects that
   share the same fingerprint (same structure, different literal values).

2. **Near-duplicate merging** — for every pair of distinct primary clusters,
   merge them when *both* conditions hold:
   - Jaccard(tables_referenced_A, tables_referenced_B) > 0.8
   - The fingerprints differ only in SELECT column order

3. **Priority scoring** — sort clusters by ``sum(execution_count)``, descending.

4. **Cap** at 1 000 clusters.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

from nlqueries.processing.query_filter import NormalizedQuery

_MAX_CLUSTERS = 1000


@dataclass
class QueryCluster:
    """A group of structurally related queries, ready for parameterization."""

    representative_sql: str
    fingerprint: str
    execution_count: int
    member_count: int
    tables_referenced: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers (exposed for testing)
# ---------------------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets.  Returns 0.0 if both are empty."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _select_without_columns(select: exp.Select) -> str:
    """Serialize the clauses of *select* that are NOT the column list.

    Used to decide whether two queries differ *only* in column order: if this
    serialization is the same for both, the column list is the only difference.

    Note: sqlglot stores the FROM clause under ``"from_"`` (with trailing
    underscore) to avoid shadowing Python's ``from`` keyword.
    """
    clause_keys = ("from_", "joins", "where", "group", "having", "order", "limit", "offset")
    parts: list[str] = []
    for key in clause_keys:
        val = select.args.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            parts.extend(item.sql() for item in val)
        else:
            parts.append(val.sql())
    return "|".join(parts)


def _differ_only_in_column_order(fp1: str, fp2: str) -> bool:
    """Return True when *fp1* and *fp2* are SELECT statements that differ only
    in the ordering of their column list.

    Both inputs are fingerprints (literals already stripped), so the comparison
    is purely structural.
    """
    try:
        p1 = sqlglot.parse_one(fp1)
        p2 = sqlglot.parse_one(fp2)
        if not isinstance(p1, exp.Select) or not isinstance(p2, exp.Select):
            return False

        cols1 = sorted(e.sql() for e in p1.expressions)
        cols2 = sorted(e.sql() for e in p2.expressions)
        if cols1 != cols2:
            return False

        return _select_without_columns(p1) == _select_without_columns(p2)
    except Exception:  # noqa: BLE001
        return False


def _merge_near_duplicates(clusters: list[QueryCluster]) -> list[QueryCluster]:
    """Merge cluster pairs that satisfy the near-duplicate criteria.

    Operates in-place on a copy; O(n²) which is acceptable for ≤ 1 000 clusters.
    """
    result = list(clusters)
    i = 0
    while i < len(result):
        j = i + 1
        while j < len(result):
            c1, c2 = result[i], result[j]
            t1, t2 = set(c1.tables_referenced), set(c2.tables_referenced)
            if _jaccard(t1, t2) > 0.8 and _differ_only_in_column_order(
                c1.fingerprint, c2.fingerprint
            ):
                # Keep the representative from the higher-traffic cluster.
                rep = (
                    c1.representative_sql
                    if c1.execution_count >= c2.execution_count
                    else c2.representative_sql
                )
                result[i] = QueryCluster(
                    representative_sql=rep,
                    fingerprint=c1.fingerprint,
                    execution_count=c1.execution_count + c2.execution_count,
                    member_count=c1.member_count + c2.member_count,
                    tables_referenced=sorted(t1 | t2),
                )
                result.pop(j)
            else:
                j += 1
        i += 1
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_queries(queries: list[NormalizedQuery]) -> list[QueryCluster]:
    """Cluster *queries* into representative groups.

    Args:
        queries: Output of filter_and_deduplicate(); each entry has a fingerprint.

    Returns:
        List of QueryCluster objects sorted by execution_count descending,
        capped at 1 000 clusters.
    """
    if not queries:
        return []

    # Step 1 — group by fingerprint
    groups: dict[str, list[NormalizedQuery]] = defaultdict(list)
    for q in queries:
        groups[q.fingerprint].append(q)

    # Step 2 — build one cluster per fingerprint group
    clusters: list[QueryCluster] = []
    for fingerprint, members in groups.items():
        representative = max(members, key=lambda q: q.execution_count)
        clusters.append(
            QueryCluster(
                representative_sql=representative.normalized_sql,
                fingerprint=fingerprint,
                execution_count=sum(q.execution_count for q in members),
                member_count=len(members),
                tables_referenced=sorted({t for q in members for t in q.tables_referenced}),
            )
        )

    # Step 3 — merge near-duplicates
    clusters = _merge_near_duplicates(clusters)

    # Step 4 — sort descending by priority score, cap
    clusters.sort(key=lambda c: c.execution_count, reverse=True)
    return clusters[:_MAX_CLUSTERS]
