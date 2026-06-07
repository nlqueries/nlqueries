"""
nlqueries.processing.parameterizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Third stage of the Query History Processor: convert cluster representative
SQL into typed Query Capsule templates by replacing every literal value with
a named, typed placeholder.

Placeholder format: ``[column_name:TYPE]`` when the column is known from the
comparison context, or ``[param_N:TYPE]`` when the column cannot be inferred.
If the same column name appears more than once the second occurrence becomes
``column_name_2``, the third ``column_name_3``, and so on.

Supported types
---------------
- ``INT``       — integer literal
- ``DECIMAL``   — float / decimal literal
- ``DATE``      — string that matches ``YYYY-MM-DD``
- ``TIMESTAMP`` — string that matches a timestamp pattern
- ``VARCHAR``   — all other string literals (default)

The ``SchemaSpec`` is optional; when supplied its column types are used to
override the default ``VARCHAR`` for plain string comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

from nlqueries.connectors.base import SchemaSpec
from nlqueries.processing.query_clusterer import QueryCluster

# ---------------------------------------------------------------------------
# Compiled regexes for date / timestamp pattern matching
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Placeholder:
    """A single typed placeholder extracted from a parameterized SQL template."""

    name: str
    type: str


@dataclass
class QueryCapsule:
    """A parameterized SQL template ready for embedding and LLM annotation."""

    template_sql: str
    placeholders: list[Placeholder]
    tables: list[str]
    columns: list[str]
    frequency: int
    auto_description: str
    intent: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_schema_type_map(schema: SchemaSpec | None) -> dict[str, str]:
    """Return a ``{column_name_lower: SQL_TYPE_UPPER}`` map from *schema*."""
    if schema is None:
        return {}
    return {
        col.name.lower(): col.type.upper()
        for table in schema.tables
        for col in table.columns
        if col.name
    }


def _get_column_from_parent(node: exp.Expression) -> str | None:
    """Infer the column name for a literal from its parent comparison node.

    Returns the lowercase column name when *node* is the value side of a
    binary predicate (``=``, ``>``, ``LIKE``, etc.), ``BETWEEN``, or ``IN``.
    Returns ``None`` when no column context can be determined.
    """
    parent = node.parent
    if parent is None:
        return None

    # Binary predicates: col = val, col > val, col LIKE val, etc.
    if isinstance(parent, (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Like, exp.ILike)):
        this = parent.args.get("this")
        expression = parent.args.get("expression")
        if isinstance(this, exp.Column) and isinstance(expression, exp.Literal):
            return this.name.lower() if this.name else None
        if isinstance(expression, exp.Column) and isinstance(this, exp.Literal):
            return expression.name.lower() if expression.name else None

    # BETWEEN: both bounds map to the same column
    if isinstance(parent, exp.Between):
        col = parent.args.get("this")
        if isinstance(col, exp.Column):
            return col.name.lower() if col.name else None

    # IN (val1, val2, ...): each element maps to the column
    if isinstance(parent, exp.In):
        col = parent.args.get("this")
        if isinstance(col, exp.Column):
            return col.name.lower() if col.name else None

    return None


def _infer_literal_type(literal: exp.Literal, schema_type: str | None) -> str:
    """Map a sqlglot ``Literal`` node to a SQL type name.

    Priority:
    1. Non-string (number): ``INT`` or ``DECIMAL`` based on whether the
       value string contains a decimal point.
    2. ISO date string (``YYYY-MM-DD``): ``DATE``.
    3. Timestamp string: ``TIMESTAMP``.
    4. Schema column type (from ``SchemaSpec``), if available.
    5. Default: ``VARCHAR``.
    """
    if not literal.is_string:
        return "DECIMAL" if "." in str(literal.this) else "INT"

    val = str(literal.this)
    if _DATE_RE.match(val):
        return "DATE"
    if _TIMESTAMP_RE.match(val):
        return "TIMESTAMP"
    if schema_type:
        return schema_type
    return "VARCHAR"


def _parameterize_sql(
    sql: str,
    schema_type_map: dict[str, str],
) -> tuple[str, list[Placeholder]]:
    """Replace every literal in *sql* with a typed placeholder string.

    Returns ``(template_sql, placeholders)``.  On parse failure returns
    the original SQL with an empty placeholder list.
    """
    try:
        ast = sqlglot.parse_one(sql)
    except Exception:  # noqa: BLE001
        return sql, []

    placeholders: list[Placeholder] = []
    used_names: dict[str, int] = {}
    counter = 0

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal counter
        if not isinstance(node, exp.Literal):
            return node

        col_name = _get_column_from_parent(node)
        schema_type = schema_type_map.get(col_name) if col_name else None
        param_type = _infer_literal_type(node, schema_type)

        # Determine base placeholder name
        if col_name:
            base = col_name
        else:
            counter += 1
            base = f"param_{counter}"

        # Deduplicate: append _2, _3, … for repeated column names
        if base in used_names:
            used_names[base] += 1
            name = f"{base}_{used_names[base]}"
        else:
            used_names[base] = 1
            name = base

        placeholders.append(Placeholder(name=name, type=param_type))
        return exp.Literal.string(f"[{name}:{param_type}]")

    try:
        transformed = ast.transform(_replace)
        template_sql = transformed.sql(pretty=False)
    except Exception:  # noqa: BLE001
        return sql, []

    return template_sql, placeholders


def _extract_tables(sql: str) -> list[str]:
    """Return a sorted list of table names referenced in *sql*."""
    try:
        return sorted(
            {t.name.lower() for t in sqlglot.parse_one(sql).find_all(exp.Table) if t.name}
        )
    except Exception:  # noqa: BLE001
        return []


def _extract_columns(sql: str) -> list[str]:
    """Return a sorted list of column names referenced in *sql*."""
    try:
        return sorted(
            {col.name.lower() for col in sqlglot.parse_one(sql).find_all(exp.Column) if col.name}
        )
    except Exception:  # noqa: BLE001
        return []


def _make_auto_description(tables: list[str], template_sql: str) -> str:
    """Generate a short human-readable description of a parameterized query.

    Format: ``"Query on {tables} filtering by {filter_columns}"``
    Falls back to ``"Query on {tables}"`` when no WHERE clause exists.
    """
    filter_cols: list[str] = []
    try:
        where = sqlglot.parse_one(template_sql).args.get("where")
        if where:
            filter_cols = sorted(
                {col.name.lower() for col in where.find_all(exp.Column) if col.name}
            )
    except Exception:  # noqa: BLE001
        pass

    tables_str = ", ".join(tables) if tables else "unknown"
    if filter_cols:
        return f"Query on {tables_str} filtering by {', '.join(filter_cols)}"
    return f"Query on {tables_str}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parameterize_cluster(
    cluster: QueryCluster,
    schema: SchemaSpec | None = None,
) -> QueryCapsule:
    """Convert a single *cluster* into a ``QueryCapsule`` template.

    Args:
        cluster: A cluster produced by ``cluster_queries()``.
        schema:  Optional ``SchemaSpec`` used to refine placeholder types for
                 plain string comparisons via column-type lookup.

    Returns:
        A ``QueryCapsule`` with ``intent`` left empty (filled by LLM annotator).
    """
    schema_type_map = _build_schema_type_map(schema)
    sql = cluster.representative_sql

    template_sql, placeholders = _parameterize_sql(sql, schema_type_map)
    tables = _extract_tables(sql)
    columns = _extract_columns(sql)
    auto_description = _make_auto_description(tables, template_sql)

    return QueryCapsule(
        template_sql=template_sql,
        placeholders=placeholders,
        tables=tables,
        columns=columns,
        frequency=cluster.execution_count,
        auto_description=auto_description,
        intent="",
    )


def parameterize_clusters(
    clusters: list[QueryCluster],
    schema: SchemaSpec | None = None,
) -> list[QueryCapsule]:
    """Convert a list of *clusters* into ``QueryCapsule`` templates.

    Args:
        clusters: Output of ``cluster_queries()``.
        schema:   Optional ``SchemaSpec`` for column-type inference.

    Returns:
        One ``QueryCapsule`` per cluster, in the same order.
    """
    return [parameterize_cluster(c, schema) for c in clusters]
