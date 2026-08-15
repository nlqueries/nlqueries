# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.verbalizer.templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Render a :class:`~nlqueries.verbalizer.ast_walk.QueryShape` into one controlled
sentence of English. The style is fixed (declarative "Shows …") and the clause
order is fixed — subject → from → joins → filters → grouping → sort/limit — so
paraphrases read consistently and never hallucinate: anything not modelled is
quoted verbatim in backticks and the :class:`Paraphrase` is marked
``complete=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot.expressions as exp

from nlqueries.verbalizer.ast_walk import QueryShape, SelectItem, resolve_column
from nlqueries.verbalizer.vocab import Vocab

# Aggregate → sentence fragment ("{}" is the argument label).
_AGG_TEMPLATE: dict[str, str] = {
    "SUM": "the sum of {}",
    "AVG": "the average of {}",
    "MIN": "the smallest {}",
    "MAX": "the largest {}",
    "COUNT": "the number of {}",
}

# Aggregate node type → template key (for aggregates appearing in predicates,
# e.g. HAVING COUNT(*) > 3).
_AGG_TYPES: dict[type[exp.Expression], str] = {
    exp.Sum: "SUM",
    exp.Avg: "AVG",
    exp.Min: "MIN",
    exp.Max: "MAX",
    exp.Count: "COUNT",
}

# Comparison node → English operator phrase.
_COMPARATORS: dict[type[exp.Expression], str] = {
    exp.EQ: "is",
    exp.NEQ: "is not",
    exp.GT: "is greater than",
    exp.GTE: "is at least",
    exp.LT: "is less than",
    exp.LTE: "is at most",
}


@dataclass(frozen=True)
class Paraphrase:
    """A controlled-English rendering of a SQL statement.

    ``complete`` is ``False`` when any fragment could not be modelled (it is then
    quoted verbatim inside ``text`` and listed in ``unhandled``), so a consumer
    can choose to show a "described in part" hint or fall back.
    """

    text: str
    complete: bool = True
    unhandled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "complete": self.complete, "unhandled": list(self.unhandled)}


def _humanize_list(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _alias_suffix(alias: str, column: str) -> str:
    if alias and alias.lower() != column.lower():
        return f" (as {alias})"
    return ""


# ---------------------------------------------------------------------------
# Operands + predicates (WHERE / HAVING)
# ---------------------------------------------------------------------------


def _literal(node: exp.Expression) -> str:
    if isinstance(node, exp.Literal):
        return f"'{node.this}'" if node.is_string else str(node.this)
    if isinstance(node, exp.Boolean):
        return "true" if node.this else "false"
    if isinstance(node, exp.Null):
        return "null"
    return ""


def _aggregate_phrase(node: exp.Expression, shape: QueryShape, vocab: Vocab) -> str | None:
    """Phrase for an aggregate used as an operand (e.g. HAVING COUNT(*) > 3)."""
    if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
        return "the number of rows"
    func = next((fn for cls, fn in _AGG_TYPES.items() if isinstance(node, cls)), "")
    if not func:
        return None
    arg = node.this
    if isinstance(arg, exp.Column):
        table, column = resolve_column(arg, shape.alias_map)
        return _AGG_TEMPLATE[func].format(vocab.column_label(table, column))
    return _AGG_TEMPLATE[func].format(f"`{arg.sql()}`") if arg is not None else None


def _operand(node: exp.Expression, shape: QueryShape, vocab: Vocab) -> tuple[str, bool]:
    """Render one side of a predicate → (text, modelled?)."""
    if isinstance(node, exp.Column):
        table, column = resolve_column(node, shape.alias_map)
        return vocab.column_label(table, column), True
    aggregate = _aggregate_phrase(node, shape, vocab)
    if aggregate is not None:
        return aggregate, True
    literal = _literal(node)
    if literal:
        return literal, True
    return f"`{node.sql()}`", False


def _render_predicate(node: exp.Expression, shape: QueryShape, vocab: Vocab) -> tuple[str, bool]:
    """Render a WHERE/HAVING predicate tree → (text, modelled?). Recursive."""
    if isinstance(node, exp.Paren):
        return _render_predicate(node.this, shape, vocab)

    if isinstance(node, exp.And):
        left, lo = _render_predicate(node.this, shape, vocab)
        right, ro = _render_predicate(node.expression, shape, vocab)
        return f"{left} and {right}", lo and ro
    if isinstance(node, exp.Or):
        left, lo = _render_predicate(node.this, shape, vocab)
        right, ro = _render_predicate(node.expression, shape, vocab)
        return f"({left} or {right})", lo and ro
    if isinstance(node, exp.Not):
        inner = node.this
        if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
            operand, ok = _operand(inner.this, shape, vocab)
            return f"{operand} is present", ok
        if isinstance(inner, exp.In):
            return _render_in(inner, shape, vocab, negated=True)
        if isinstance(inner, exp.Between):
            return _render_between(inner, shape, vocab, negated=True)
        if isinstance(inner, (exp.Like, exp.ILike)):
            return _render_like(inner, shape, vocab, negated=True)
        text, ok = _render_predicate(inner, shape, vocab)
        return f"not {text}", ok

    comparator = next((op for cls, op in _COMPARATORS.items() if type(node) is cls), "")
    if comparator:
        left, lo = _operand(node.this, shape, vocab)
        right, ro = _operand(node.expression, shape, vocab)
        return f"{left} {comparator} {right}", lo and ro

    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        operand, ok = _operand(node.this, shape, vocab)
        # `IS NOT NULL` has two AST shapes depending on sqlglot version/dialect:
        # older builds wrap it as Not(Is(…)) — handled above — while newer ones
        # (30.17+ on postgres) keep a single Is node with negate=True. Missing
        # the flag inverts the sentence, which is worse than not rendering it.
        if node.args.get("negate"):
            return f"{operand} is present", ok
        return f"{operand} is missing", ok

    if isinstance(node, exp.In):
        return _render_in(node, shape, vocab, negated=False)

    if isinstance(node, exp.Between):
        return _render_between(node, shape, vocab, negated=False)

    if isinstance(node, (exp.Like, exp.ILike)):
        return _render_like(node, shape, vocab, negated=False)

    return f"`{node.sql()}`", False


def _render_in(node: exp.In, shape: QueryShape, vocab: Vocab, *, negated: bool) -> tuple[str, bool]:
    operand, ok = _operand(node.this, shape, vocab)
    values = node.args.get("expressions") or node.expressions
    if node.args.get("query") or not values:
        return f"`{node.sql()}`", False  # IN (subquery) not modelled in v1
    rendered = [_literal(v) or f"`{v.sql()}`" for v in values]
    phrase = "is not one of" if negated else "is one of"
    return f"{operand} {phrase} {', '.join(rendered)}", ok


def _render_between(
    node: exp.Between, shape: QueryShape, vocab: Vocab, *, negated: bool
) -> tuple[str, bool]:
    operand, ok = _operand(node.this, shape, vocab)
    low, lo = _operand(node.args["low"], shape, vocab)
    high, ho = _operand(node.args["high"], shape, vocab)
    phrase = "is not between" if negated else "is between"
    return f"{operand} {phrase} {low} and {high}", ok and lo and ho


def _render_like(
    node: exp.Like | exp.ILike, shape: QueryShape, vocab: Vocab, *, negated: bool
) -> tuple[str, bool]:
    operand, ok = _operand(node.this, shape, vocab)
    pattern, po = _operand(node.expression, shape, vocab)
    # sqlglot encodes ``NOT LIKE`` as a ``negate`` flag on the Like node itself,
    # rather than wrapping it in a Not.
    negated = negated or bool(node.args.get("negate"))
    phrase = "does not match" if negated else "matches"
    return f"{operand} {phrase} {pattern}", ok and po


# ---------------------------------------------------------------------------
# Select items
# ---------------------------------------------------------------------------


def _render_item(item: SelectItem, vocab: Vocab) -> str:
    if item.kind == "count_star":
        return "the number of rows" + _alias_suffix(item.alias, "")
    if item.kind == "column":
        label = vocab.column_label(item.table, item.column)
        return label + _alias_suffix(item.alias, item.column)
    if item.kind == "aggregate":
        arg = f"`{item.raw}`" if item.of_opaque else vocab.column_label(item.table, item.column)
        template = _AGG_TEMPLATE.get(item.func, "{}")
        return template.format(arg) + _alias_suffix(item.alias, item.column)
    # opaque
    return f"`{item.raw}`" + _alias_suffix(item.alias, "")


_JOIN_VERB = {
    "inner": "joined to",
    "left": "left-joined to",
    "right": "right-joined to",
    "full": "outer-joined to",
}


def _from_label(shape: QueryShape, vocab: Vocab) -> str:
    if shape.from_opaque:
        return "a subquery"
    if not shape.from_table:
        return "the result"
    return vocab.table_label(shape.from_table)


def _group_label(item: tuple[str, str] | str, vocab: Vocab) -> str:
    if isinstance(item, tuple):
        return vocab.column_label(item[0], item[1])
    return f"`{item}`"


def _limit_clause(shape: QueryShape) -> str:
    if shape.limit is None:
        return ""
    if shape.order_by:
        return f", top {shape.limit}"
    noun = "row" if shape.limit == 1 else "rows"
    return f", limited to {shape.limit} {noun}"


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------


def render(shape: QueryShape, vocab: Vocab | None = None) -> Paraphrase:
    """Render *shape* into a :class:`Paraphrase` (declarative, one sentence)."""
    vocab = vocab or Vocab()
    if not shape.ok:
        return Paraphrase(text="", complete=False, unhandled=list(shape.unhandled))

    incomplete = not shape.complete

    is_star = any(i.kind == "star" for i in shape.select_items)
    count_only = (
        len(shape.select_items) == 1
        and shape.select_items[0].kind == "count_star"
        and not shape.group_by
    )
    from_label = _from_label(shape, vocab)

    if count_only:
        prefix = "all " if shape.where is None else ""
        sentence = f"Counts {prefix}{from_label}".rstrip()
    elif is_star:
        sentence = f"Shows all columns from {from_label}"
    else:
        items = _humanize_list([_render_item(i, vocab) for i in shape.select_items])
        sentence = f"Shows {items} from {from_label}"

    # Joins, grouping, sorting read as a comma list; a WHERE/HAVING filter reads
    # better attached without a comma ("… from orders where …").
    for join in shape.joins:
        label = "a subquery" if join.opaque else vocab.table_label(join.table)
        sentence += f", {_JOIN_VERB.get(join.kind, 'joined to')} {label}"

    if shape.where is not None:
        text, ok = _render_predicate(shape.where, shape, vocab)
        incomplete = incomplete or not ok
        sentence += f" where {text}"

    if shape.group_by:
        labels = _humanize_list([_group_label(g, vocab) for g in shape.group_by])
        sentence += f", grouped by {labels}"

    if shape.having is not None:
        text, ok = _render_predicate(shape.having, shape, vocab)
        incomplete = incomplete or not ok
        sentence += f", keeping groups where {text}"

    if shape.order_by:
        parts = []
        for order in shape.order_by:
            if order.label_ref is not None:
                label = vocab.column_label(order.label_ref[0], order.label_ref[1])
            else:
                label = f"`{order.text}`"
            direction = "highest first" if order.descending else "lowest first"
            parts.append(f"{label} ({direction})")
        sentence += f", sorted by {_humanize_list(parts)}"

    sentence = (sentence + _limit_clause(shape)).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:] + "."

    return Paraphrase(text=sentence, complete=not incomplete, unhandled=list(shape.unhandled))
