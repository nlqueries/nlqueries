"""
nlqueries.orchestrator.reading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How the question was read, said in a line under the answer (owner feedback
F20, phase 1).

An answer that is wrong because of how the question was read looks as complete
as one that is right: "flights around noon" read as 11:30 to 12:30, "between
Boston and Denver" applied both ways, "the top customers" cut at ten. Here the
final statement is read for the choices it makes that the question may not
have, and each is said in a few words:

    Read as: origin 'SFO'; dep from 1130 to 1230; either way between 'BOS'
    and 'DEN'; top 10 by revenue

No model call: each item comes from the parsed statement by a fixed rule, so
it says what the statement does, never what the model meant. Read from the
statement alone, so a cached answer replayed later says the same. Only the
outermost ``SELECT`` is read, and of its ``WHERE`` only the top-level ``AND``
conditions:

- ``value``: a column compared with a text value, or a list of them;
- ``range``: a column held between two bounds, by ``BETWEEN`` or by a lower and
  an upper bound;
- ``both_directions``: two columns matched to two values either way round
  (``(a = 'X' AND b = 'Y') OR (a = 'Y' AND b = 'X')``);
- ``limit``: a row limit, with the ordering it cuts, or the lack of one.

Off unless ``NLQ_READING_LINE`` (``reading_line``) turns it on.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from nlqueries.sql_policy import _sqlglot_dialect

#: The most items said; a statement with more choices than this says the first.
MAX_ITEMS = 6


@dataclass(frozen=True)
class ReadingItem:
    """One choice the statement makes, in words."""

    kind: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text}


def _conjuncts(where: exp.Expr) -> list[exp.Expr]:
    node = where.this if isinstance(where, exp.Where) else where
    out: list[exp.Expr] = []

    def walk(e: exp.Expr) -> None:
        e = e.unnest()
        if isinstance(e, exp.And):
            walk(e.left)
            walk(e.right)
        else:
            out.append(e)

    walk(node)
    return out


def _name(e: exp.Expr, dialect: str) -> str:
    """A column by its name alone, anything else as written."""
    return e.name if isinstance(e, exp.Column) and e.name else e.sql(dialect=dialect)


def _text(e: exp.Expr) -> str | None:
    """*e* as a quoted text value, or None when it is not one."""
    return f"'{e.this}'" if isinstance(e, exp.Literal) and e.is_string else None


def _value(e: exp.Expr, dialect: str) -> str | None:
    """*e* as a bound: a number or a text value, as written."""
    return e.sql(dialect=dialect) if isinstance(e, exp.Literal) else None


def _column_and_literal(cond: exp.Expr) -> tuple[exp.Column, exp.Literal] | None:
    """``(column, literal)`` of a comparison between the two, either side."""
    left, right = cond.args.get("this"), cond.args.get("expression")
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
        return left, right
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
        return right, left
    return None


def _values(conds: list[exp.Expr], dialect: str) -> list[ReadingItem]:
    items: list[ReadingItem] = []
    for cond in conds:
        if isinstance(cond, exp.EQ) and (pair := _column_and_literal(cond)):
            column, literal = pair
            if (shown := _text(literal)) is not None:
                items.append(ReadingItem("value", f"{_name(column, dialect)} {shown}"))
        elif (
            isinstance(cond, exp.In)
            and isinstance(cond.this, exp.Column)
            # A subquery's `IN` lists no values: its query is not read.
            and cond.expressions
        ):
            shown_all = [_text(e) for e in cond.expressions]
            if all(s is not None for s in shown_all):
                listed = " or ".join(s for s in shown_all if s is not None)
                items.append(ReadingItem("value", f"{_name(cond.this, dialect)} {listed}"))
    return items


#: A bound's operator, as (side, inclusive), for a column on the left.
_BOUNDS: dict[type[exp.Expr], tuple[str, bool]] = {
    exp.GTE: ("lower", True),
    exp.GT: ("lower", False),
    exp.LTE: ("upper", True),
    exp.LT: ("upper", False),
}
_FLIPPED = {"lower": "upper", "upper": "lower"}


def _ranges(conds: list[exp.Expr], dialect: str) -> list[ReadingItem]:
    items: list[ReadingItem] = []
    # Keyed by the column as written, table and all: `o.created_at` and
    # `s.created_at` are two columns, and a bound on each is no range.
    bounds: dict[str, dict[str, tuple[str, bool]]] = {}
    names: dict[str, str] = {}
    order: list[str] = []
    for cond in conds:
        if isinstance(cond, exp.Between) and isinstance(cond.this, exp.Column):
            low = _value(cond.args["low"], dialect)
            high = _value(cond.args["high"], dialect)
            if low is not None and high is not None:
                items.append(
                    ReadingItem("range", f"{_name(cond.this, dialect)} from {low} to {high}")
                )
            continue
        bound = _BOUNDS.get(type(cond))
        pair = _column_and_literal(cond) if bound else None
        if bound is None or pair is None:
            continue
        column, literal = pair
        side, inclusive = bound
        if cond.args.get("this") is not column:
            side = _FLIPPED[side]  # `100 < price` holds price from below
        shown = _value(literal, dialect)
        if shown is None:
            continue
        key = column.sql(dialect=dialect)
        if key not in bounds:
            bounds[key] = {}
            names[key] = _name(column, dialect)
            order.append(key)
        bounds[key].setdefault(side, (shown, inclusive))
    for key in order:
        b = bounds[key]
        if "lower" in b and "upper" in b:
            # Said by name alone, unless another column bounded here has it.
            shared = sum(1 for k in order if names[k] == names[key]) > 1
            said = key if shared else names[key]
            (low, low_in), (high, high_in) = b["lower"], b["upper"]
            if low_in and high_in:
                items.append(ReadingItem("range", f"{said} from {low} to {high}"))
            else:
                lower = f"{'at least' if low_in else 'over'} {low}"
                upper = f"{'at most' if high_in else 'under'} {high}"
                items.append(ReadingItem("range", f"{said} {lower} and {upper}"))
    return items


def _equalities(branch: exp.Expr) -> dict[str, str] | None:
    """``{column: value}`` of a branch made only of column = text value, the
    column as written: in a self-join, `o.code` and `d.code` are two."""
    found: dict[str, str] = {}
    for cond in _conjuncts(branch):
        pair = _column_and_literal(cond) if isinstance(cond, exp.EQ) else None
        shown = _text(pair[1]) if pair else None
        if pair is None or shown is None:
            return None
        found[pair[0].sql()] = shown
    return found


def _both_directions(conds: list[exp.Expr]) -> list[ReadingItem]:
    items: list[ReadingItem] = []
    for cond in conds:
        if not isinstance(cond, exp.Or):
            continue
        one, two = _equalities(cond.left), _equalities(cond.right)
        if not one or not two or len(one) != 2 or set(one) != set(two):
            continue
        a, b = list(one)  # as the statement writes them
        if one[a] == two[b] and one[b] == two[a] and one[a] != one[b]:
            items.append(
                ReadingItem("both_directions", f"either way between {one[a]} and {one[b]}")
            )
    return items


def _limit(select: exp.Select, dialect: str) -> list[ReadingItem]:
    limit = select.args.get("limit")
    count = limit.expression if isinstance(limit, exp.Limit) else None
    if not isinstance(count, exp.Literal) or count.is_string:
        return []
    n = count.this
    # Rows skipped first: `LIMIT 10 OFFSET 10` is the 11th to the 20th.
    after = ""
    offset = select.args.get("offset")
    if offset is not None:
        skipped = offset.expression if isinstance(offset, exp.Offset) else None
        if not isinstance(skipped, exp.Literal) or skipped.is_string:
            return []
        after = f" after the first {skipped.this}"
    order = select.args.get("order")
    keys = order.expressions if isinstance(order, exp.Order) else []
    if not keys:
        return [ReadingItem("limit", f"any {n} rows{after}, in no set order")]
    first = keys[0]
    key = first.this if isinstance(first, exp.Ordered) else first
    word = "top" if isinstance(first, exp.Ordered) and first.args.get("desc") else "bottom"
    return [ReadingItem("limit", f"{word} {n} by {_name(key, dialect)}{after}")]


def read_as(sql: str, dialect: str) -> list[ReadingItem]:
    """The choices *sql* makes, in words; empty when it is not one ``SELECT``
    or makes none of the kinds said."""
    dialect = _sqlglot_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 — nothing to say about it
        return []
    if not isinstance(parsed, exp.Select):
        return []
    where = parsed.args.get("where")
    conds = _conjuncts(where) if where is not None else []
    items = [
        *_values(conds, dialect),
        *_ranges(conds, dialect),
        *_both_directions(conds),
        *_limit(parsed, dialect),
    ]
    return items[:MAX_ITEMS]
