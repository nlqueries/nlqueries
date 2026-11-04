"""
nlqueries.orchestrator.grain_check
==================================
The plan line, and a check of the statement against it.

Some wrong answers are right SQL for a different question about what one row
is, or what is counted: grouped by a name where two customers share it, or a
COUNT over a join that repeats each order once per item. The model is asked to
begin its reasoning with one line that says what the answer is made of::

    Plan: rows = one customer per row, key customers.id; counted = orders,
    once each by orders.id; filters = placed in 2024

Writing it makes the model decide those two things before it writes the
statement, and the user sees the reading in the reasoning they already get.

The line also says something a statement can be checked against, and two
disagreements are caught here without a model call:

- the rows are told apart by a key, and the statement groups that key's table
  only by other columns of it, so rows sharing those values are added
  together;
- each counted thing is counted once by a key, and the statement counts with a
  plain COUNT across a join to a table whose foreign key references that key,
  where each counted row appears once per row that refers to it.

The second needs the foreign key in the knowledge base (``references``), so it
is checked only where the knowledge base says which side of a join repeats.
Both are narrow on purpose: a disagreement costs one model call, and a check
that fires on right statements would spend it for nothing.

On a disagreement the model is asked once for the statement that matches what
the question asks, the plan where it is right or a corrected plan where it is
not. The answer replaces the statement only when it validates, differs, and
shows no disagreement with its own plan line. Advisory throughout: whatever
fails leaves the statement as it was.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

from nlqueries import config
from nlqueries.llm import output_budget
from nlqueries.orchestrator.plan_check import _room_for_rewrite
from nlqueries.orchestrator.provenance import record_grain_check
from nlqueries.orchestrator.sql_generation import (
    SQLGenerationResult,
    _extract_sql,
    _same_statement,
    _validate_sql,
)

if TYPE_CHECKING:
    from nlqueries.llm.client import LLMClient

_log = logging.getLogger(__name__)

#: The plan line: the first line whose text starts with ``Plan:``, allowing for
#: markdown emphasis or a list marker in front of it.
_PLAN_LINE = re.compile(r"^[\s>*_`#-]*plan\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
#: ``table.column``, unquoted or quoted with backticks or double quotes.
_QUALIFIED = r"[`\"]?([A-Za-z_][\w$]*)[`\"]?\.[`\"]?([A-Za-z_][\w$]*)[`\"]?"
_ROW_KEY = re.compile(r"\bkey\s+" + _QUALIFIED, re.IGNORECASE)
_COUNT_KEY = re.compile(r"\bonce\s+each\s+by\s+" + _QUALIFIED, re.IGNORECASE)


@dataclass(frozen=True)
class GrainPlan:
    """What the plan line says the answer is made of."""

    text: str
    #: ``(table, column)``, lower case, that tells the answer's rows apart.
    row_key: tuple[str, str] | None
    #: ``(table, column)``, lower case, by which each counted thing is counted once.
    count_key: tuple[str, str] | None


def _field(text: str, label: str) -> str:
    """The value of ``label = ...`` in the plan line, up to the next ``;``."""
    m = re.search(rf"\b{label}\s*=\s*([^;]*)", text, re.IGNORECASE)
    return m.group(1) if m else ""


def _key(pattern: re.Pattern[str], value: str) -> tuple[str, str] | None:
    m = pattern.search(value)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def parse_plan(output: str) -> GrainPlan | None:
    """The plan line in the model's *output*, or None when it wrote none."""
    m = _PLAN_LINE.search(output)
    if m is None:
        return None
    text = m.group(1).strip("*_` ")
    return GrainPlan(
        text=text,
        row_key=_key(_ROW_KEY, _field(text, "rows")),
        count_key=_key(_COUNT_KEY, _field(text, "counted")),
    )


def _references(knowledge_base: dict[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    """``{(table, column): (table, column) it references}``, lower case, from
    the knowledge base's foreign keys. A schema-qualified reference keeps its
    last two parts."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for table in knowledge_base.get("schema", {}).get("tables", []) or []:
        name = str(table.get("name") or "").lower()
        for col in table.get("columns", []) or []:
            ref = col.get("references")
            if isinstance(ref, str) and ref.count(".") >= 1:
                ref_table, ref_col = ref.lower().rsplit(".", 2)[-2:]
                out[(name, str(col.get("name") or "").lower())] = (ref_table, ref_col)
    return out


def _sources(select: exp.Select) -> dict[str, str]:
    """``{alias or name: table name}``, lower case, for the tables this
    select reads directly (not through a derived table or CTE)."""
    nodes: list[exp.Expr] = []
    source = select.args.get("from_") or select.args.get("from")
    if source is not None:
        nodes.append(source.this)
    nodes.extend(j.this for j in select.args.get("joins") or [])
    return {
        n.alias_or_name.lower(): n.name.lower()
        for n in nodes
        if isinstance(n, exp.Table) and n.name
    }


def _resolve(col: exp.Column, sources: dict[str, str]) -> tuple[str, str] | None:
    """The ``(table, column)`` *col* reads, when the statement says which."""
    if col.table:
        table = sources.get(col.table.lower())
        return (table, col.name.lower()) if table else None
    tables = set(sources.values())
    return (tables.pop(), col.name.lower()) if len(tables) == 1 else None


def _grouped_without_the_key(
    select: exp.Select, key: tuple[str, str], sources: dict[str, str]
) -> str | None:
    group = select.args.get("group")
    if group is None:
        return None
    grouped = [
        r
        for e in group.expressions
        if isinstance(e, exp.Column) and (r := _resolve(e, sources)) is not None
    ]
    same_table = [c for t, c in grouped if t == key[0]]
    if not same_table or key in grouped:
        return None
    table, column = key
    return (
        f"The plan tells the rows apart by {table}.{column}, but the statement groups "
        f"{table} by {', '.join(same_table)} without it, so {table} rows that share "
        f"{'that value' if len(same_table) == 1 else 'those values'} are added together."
    )


def _counted_through_a_repeating_join(
    select: exp.Select,
    key: tuple[str, str],
    sources: dict[str, str],
    references: dict[tuple[str, str], tuple[str, str]],
) -> str | None:
    plain = [
        c
        for c in select.find_all(exp.Count)
        if c.find_ancestor(exp.Select) is select and not isinstance(c.this, exp.Distinct)
    ]
    if not plain:
        return None
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            sides = (_resolve(left, sources), _resolve(right, sources))
            for this, other in (sides, sides[::-1]):
                if this == key and other is not None and references.get(other) == key:
                    table, column = key
                    return (
                        f"The plan counts each {table} row once, by {table}.{column}, but the "
                        f"statement counts the rows of its join to {other[0]}, where a {table} "
                        f"row appears once for every {other[0]} row that refers to it; "
                        f"COUNT(DISTINCT {table}.{column}) counts each once."
                    )
    return None


def mismatches(
    plan: GrainPlan, sql: str, dialect: str, knowledge_base: dict[str, Any]
) -> list[str]:
    """Where *sql* disagrees with *plan*, one sentence each; empty when it
    agrees, or when the statement is not a single SELECT this can read."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.SqlglotError, ValueError):
        return []
    if not isinstance(tree, exp.Select):
        return []
    sources = _sources(tree)
    found = []
    if plan.row_key is not None:
        found.append(_grouped_without_the_key(tree, plan.row_key, sources))
    if plan.count_key is not None:
        found.append(
            _counted_through_a_repeating_join(
                tree, plan.count_key, sources, _references(knowledge_base)
            )
        )
    return [f for f in found if f]


async def repair_to_plan(
    sql: str,
    plan: GrainPlan,
    problems: list[str],
    question: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    system: str | list[dict[str, Any]],
    attempt_count: int,
) -> tuple[SQLGenerationResult, GrainPlan]:
    """Ask the model once for the statement that matches what was asked.

    The plan line may be the part that is wrong, so the model may correct it
    instead; the answer's own plan line, when it writes one, is what the
    corrected statement is checked against. Validated as a first attempt is.
    """
    disagreements = "\n".join(f"- {p}" for p in problems)
    user = (
        "Your plan line and your SQL disagree:\n"
        f"{disagreements}\n\n"
        f"Question the SQL must answer:\n{question}\n\n"
        f"Plan line:\nPlan: {plan.text}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Please write the plan line again, as it should be for the question, and then a "
        f"{dialect} SELECT statement that matches it. If the SQL already answers the question "
        "as asked, return it unchanged. Wrap the SQL in <sql>...</sql> markers."
    )
    raw = await llm.acomplete(system, user, max_tokens=output_budget("correction"))
    repaired = _extract_sql(raw)
    error = _validate_sql(repaired, knowledge_base, dialect)
    result = SQLGenerationResult(
        sql=repaired,
        is_valid=error is None,
        validation_error=error,
        dialect=dialect,
        attempt_count=attempt_count + 1,
    )
    return result, parse_plan(raw) or plan


async def check_grain(
    result: SQLGenerationResult,
    output: str,
    *,
    question: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    system: str | list[dict[str, Any]],
    started: float | None = None,
    deadline: float | None = None,
) -> SQLGenerationResult:
    """The statement to go on with: a repair that matches the plan line in
    *output*, or *result* itself.

    The repair is asked for only when the statement disagrees with the plan
    line and the caller's *deadline* has room for a model call (the rule the
    plan check uses, with *started* when the attempt began), and it replaces
    *result* only when it is valid, different, and agrees with its own plan
    line. The outcome goes into the answer's provenance as ``grain_check``.
    """
    if not config.GRAIN_CHECK or not result.is_valid:
        return result
    plan = parse_plan(output)
    if plan is None:
        return result
    try:
        problems = mismatches(plan, result.sql, dialect, knowledge_base)
    except Exception:  # noqa: BLE001 - a check that fails says nothing
        _log.debug("Checking a statement against its plan line failed.", exc_info=True)
        return result
    if not problems:
        return result
    _log.info("The statement disagrees with its plan line: %s", " ".join(problems))

    def outcome(reason: str | None, attempted: bool = True) -> None:
        record_grain_check(
            plan=plan.text,
            mismatches=problems,
            attempted=attempted,
            changed=reason is None,
            reason=reason,
        )

    if not _room_for_rewrite(started, deadline):
        outcome("skipped: too little time before the deadline", attempted=False)
        return result
    try:
        repaired, new_plan = await repair_to_plan(
            result.sql,
            plan,
            problems,
            question,
            knowledge_base,
            dialect,
            llm,
            system,
            result.attempt_count,
        )
    except Exception:  # noqa: BLE001 - the statement as written still answers
        _log.warning("Repairing a statement to its plan line failed.", exc_info=True)
        outcome("failed: the model call raised")
        return result
    if not repaired.is_valid:
        outcome("rejected: refused by validation")
        return result
    if _same_statement(repaired.sql, result.sql, dialect):
        outcome("rejected: returned unchanged")
        return result
    try:
        still = mismatches(new_plan, repaired.sql, dialect, knowledge_base)
    except Exception:  # noqa: BLE001
        still = ["the repair could not be checked"]
    if still:
        outcome("rejected: still disagrees with its plan line")
        return result
    outcome(None)
    return repaired
