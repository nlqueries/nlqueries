# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.verbalizer.ast_walk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parse a single ``SELECT`` into a :class:`QueryShape` — the render-ready
intermediate the templates turn into controlled English. All sqlglot handling
lives here; :mod:`nlqueries.verbalizer.templates` reads the shape (and, for
predicates, walks the raw ``where`` / ``having`` nodes via :func:`resolve_column`).

Best-effort and total: anything not modelled degrades rather than raising. A
statement that is not a plain ``SELECT`` (a set operation, DML, a parse error)
comes back as ``ok=False``; a ``SELECT`` we can mostly model but whose FROM is a
subquery/CTE, or which contains an aggregate over an expression, comes back with
``complete=False`` and the offending fragment recorded in ``unhandled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

# Aggregate node type → the SQL function name we template on.
_AGGREGATES: dict[type[exp.Expression], str] = {
    exp.Sum: "SUM",
    exp.Avg: "AVG",
    exp.Min: "MIN",
    exp.Max: "MAX",
    exp.Count: "COUNT",
}


@dataclass
class SelectItem:
    """One projected column / expression."""

    kind: str  # "column" | "aggregate" | "count_star" | "star" | "opaque"
    table: str = ""
    column: str = ""
    func: str = ""  # SUM | AVG | MIN | MAX | COUNT (aggregate only)
    alias: str = ""
    raw: str = ""  # verbatim SQL for opaque items / aggregate-of-expression
    of_opaque: bool = False  # aggregate whose argument was not a plain column


@dataclass
class JoinClause:
    table: str
    kind: str = "inner"  # inner | left | right | full
    opaque: bool = False  # joined onto a subquery


@dataclass
class OrderItem:
    label_ref: tuple[str, str] | None  # (table, column) when a plain column
    text: str  # alias / raw expression when not a plain column
    descending: bool = False


@dataclass
class QueryShape:
    """Render-ready shape of a SELECT. ``ok`` gates rendering at all; ``complete``
    reports whether every part was modelled (else some fragment was quoted)."""

    ok: bool = True
    raw_sql: str = ""
    select_items: list[SelectItem] = field(default_factory=list)
    from_table: str = ""
    from_opaque: bool = False
    joins: list[JoinClause] = field(default_factory=list)
    where: exp.Expression | None = None
    group_by: list[tuple[str, str] | str] = field(default_factory=list)
    having: exp.Expression | None = None
    order_by: list[OrderItem] = field(default_factory=list)
    limit: int | None = None
    alias_map: dict[str, str] = field(default_factory=dict)
    complete: bool = True
    unhandled: list[str] = field(default_factory=list)

    def flag(self, fragment: str) -> None:
        """Record a fragment we could not model and mark the shape incomplete."""
        self.complete = False
        if fragment and fragment not in self.unhandled:
            self.unhandled.append(fragment)


# ---------------------------------------------------------------------------
# Column resolution (shared with templates for predicate rendering)
# ---------------------------------------------------------------------------


def resolve_column(column: exp.Column, alias_map: dict[str, str]) -> tuple[str, str]:
    """Resolve a column to ``(real_table, column_name)`` using the alias map.

    An unqualified column resolves to ``("", name)``; a qualifier that is a known
    alias is expanded to the real table, otherwise kept as written.
    """
    name = str(column.name)
    qualifier = str(column.table) if column.table else ""
    table = alias_map.get(qualifier, qualifier)
    return table, name


def _table_name(table: exp.Table) -> str:
    """Bare table name (schema/catalog stripped, quoting removed)."""
    return str(table.name)


def _build_alias_map(select: exp.Select) -> dict[str, str]:
    """alias-or-name → real table name for every table in the statement."""
    alias_map: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        real = _table_name(table)
        alias_map[table.alias or real] = real
    return alias_map


# ---------------------------------------------------------------------------
# Select-item classification
# ---------------------------------------------------------------------------


def _classify_item(expression: exp.Expression, shape: QueryShape) -> SelectItem:
    node = expression
    alias = ""
    if isinstance(node, exp.Alias):
        alias = str(node.alias)
        node = node.this

    if isinstance(node, exp.Star):
        return SelectItem(kind="star", alias=alias)

    if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
        return SelectItem(kind="count_star", func="COUNT", alias=alias)

    agg_func = next((fn for cls, fn in _AGGREGATES.items() if isinstance(node, cls)), "")
    if agg_func:
        arg = node.this
        if isinstance(arg, exp.Column):
            table, column = resolve_column(arg, shape.alias_map)
            return SelectItem(
                kind="aggregate", table=table, column=column, func=agg_func, alias=alias
            )
        arg_raw = str(arg.sql()) if arg is not None else ""
        shape.flag(str(node.sql()))
        return SelectItem(kind="aggregate", func=agg_func, alias=alias, raw=arg_raw, of_opaque=True)

    if isinstance(node, exp.Column):
        table, column = resolve_column(node, shape.alias_map)
        return SelectItem(kind="column", table=table, column=column, alias=alias)

    raw = str(node.sql())
    shape.flag(raw)
    return SelectItem(kind="opaque", alias=alias, raw=raw)


# ---------------------------------------------------------------------------
# Clause extraction
# ---------------------------------------------------------------------------


def _extract_from(select: exp.Select, shape: QueryShape, cte_names: set[str]) -> None:
    # sqlglot suffixes the reserved word: the FROM arg is "from_" (30.x); older
    # versions used "from". Accept either so we're not pinned to one release.
    from_clause = select.args.get("from_") or select.args.get("from")
    if not isinstance(from_clause, exp.From):
        shape.from_opaque = True
        shape.flag("FROM")
        return
    source = from_clause.this
    if isinstance(source, exp.Table):
        name = _table_name(source)
        shape.from_table = name
        if name in cte_names:
            shape.flag(name)  # CTE body is not described
    else:
        shape.from_opaque = True
        shape.flag(str(source.sql()))


def _extract_joins(select: exp.Select, shape: QueryShape) -> None:
    for join in select.args.get("joins") or []:
        if not isinstance(join, exp.Join):
            continue
        target = join.this
        if not isinstance(target, exp.Table):
            shape.joins.append(JoinClause(table="", opaque=True))
            shape.flag(str(target.sql()) if target else "join")
            continue
        side = str(join.args.get("side") or "").upper()
        kind = {"LEFT": "left", "RIGHT": "right", "FULL": "full"}.get(side, "inner")
        shape.joins.append(JoinClause(table=_table_name(target), kind=kind))


def _extract_group(select: exp.Select, shape: QueryShape) -> None:
    group = select.args.get("group")
    if not isinstance(group, exp.Group):
        return
    for item in group.expressions:
        if isinstance(item, exp.Column):
            shape.group_by.append(resolve_column(item, shape.alias_map))
        else:
            shape.group_by.append(str(item.sql()))


def _extract_order(select: exp.Select, shape: QueryShape) -> None:
    order = select.args.get("order")
    if not isinstance(order, exp.Order):
        return
    for ordered in order.expressions:
        if not isinstance(ordered, exp.Ordered):
            continue
        target = ordered.this
        descending = bool(ordered.args.get("desc"))
        if isinstance(target, exp.Column):
            ref = resolve_column(target, shape.alias_map)
            shape.order_by.append(OrderItem(label_ref=ref, text="", descending=descending))
        else:
            shape.order_by.append(
                OrderItem(label_ref=None, text=str(target.sql()), descending=descending)
            )


def _extract_limit(select: exp.Select, shape: QueryShape) -> None:
    limit = select.args.get("limit")
    if isinstance(limit, exp.Limit) and isinstance(limit.expression, exp.Literal):
        try:
            shape.limit = int(str(limit.expression.this))
        except (TypeError, ValueError):
            shape.limit = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyze(sql: str, dialect: str = "postgres") -> QueryShape:
    """Parse *sql* into a :class:`QueryShape`. Never raises."""
    shape = QueryShape(raw_sql=sql)
    if not sql or not sql.strip():
        shape.ok = False
        return shape

    try:
        statement = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError:
        shape.ok = False
        return shape
    except Exception:  # noqa: BLE001 — any sqlglot failure degrades, never raises
        shape.ok = False
        return shape

    if not isinstance(statement, exp.Select):
        shape.ok = False
        return shape

    select = statement  # narrowed to exp.Select above
    try:
        cte_names = {str(cte.alias) for cte in select.find_all(exp.CTE) if cte.alias}
        shape.alias_map = _build_alias_map(select)

        expressions = list(select.expressions)
        if any(isinstance(e, exp.Star) for e in expressions):
            shape.select_items.append(SelectItem(kind="star"))
        else:
            shape.select_items = [_classify_item(e, shape) for e in expressions]

        _extract_from(select, shape, cte_names)
        _extract_joins(select, shape)
        where = select.args.get("where")
        shape.where = where.this if isinstance(where, exp.Where) else None
        _extract_group(select, shape)
        having = select.args.get("having")
        shape.having = having.this if isinstance(having, exp.Having) else None
        _extract_order(select, shape)
        _extract_limit(select, shape)
    except Exception:  # noqa: BLE001 — partial model is fine; never raise to the caller
        shape.flag("(analysis error)")

    return shape
