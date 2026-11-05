"""
nlqueries.orchestrator.plan_check
=================================
A look at a statement's plan before it runs, for a subquery the database will
re-run once for every row of the outer query.

``WHERE o.total > (SELECT AVG(total) FROM orders WHERE customer_id =
o.customer_id)`` reads as one question and is written as one statement, and
most engines run it as a loop: the subquery once per outer row, each time
reading ``orders`` again. Over two tables of 50,000 rows that is two and a half
billion row reads for an answer a join to a grouped derived table gives in one
pass, and the user waits for the statement timeout and gets nothing. The
model writes it this way because the question is phrased this way, and the
schema check and the policy see nothing wrong with it, because nothing is.

Only some engines' plans say so plainly, and a plain ``EXPLAIN`` costs them
milliseconds without running the statement:

- PostgreSQL: a node under ``"Parent Relationship": "SubPlan"`` is evaluated
  per row of its parent, unless the parent's condition names it ``hashed``
  (built once and probed). ``InitPlan`` runs once. A ``Seq Scan`` inside it
  reads its table in full each time. How many times, PostgreSQL's own costing
  says: the parent's total cost includes the subplan's cost once per
  evaluation.
- MySQL and MariaDB: a ``DEPENDENT SUBQUERY`` row with access type ``ALL``
  (a table scan) or ``index`` (a full index scan), with the rows examined per
  run and the outer rows it runs for both in the plan.
- SQLite: a ``CORRELATED ... SUBQUERY`` entry with a ``SCAN`` beneath it. A
  ``SEARCH`` is an index lookup and is not counted. The plan gives no sizes,
  so they come from the knowledge base, and the outer side is the largest
  table the outer query scans.

DuckDB rewrites correlated subqueries into joins itself, so its plans have
nothing to find, and Snowflake, BigQuery and SQL Server are not checked.

When the rows the re-runs read reach ``PLAN_CHECK_MIN_ROW_READS``, the model
is asked once for the same statement with the subquery computed once. The
answer is validated like any model output, and replaces the statement only
when its own plan shows no such subquery; anything else, including a failed
``EXPLAIN``, leaves the statement as it was. The sizes are estimates, the
planner's or the knowledge base's at export time, so the check is a heuristic:
an estimate too high costs one model call, and a rewrite still has to pass its
own plan before it replaces anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import ScopeType, traverse_scope

from nlqueries.llm import output_budget
from nlqueries.orchestrator.literal_grounding import _row_counts
from nlqueries.orchestrator.provenance import record_plan_rewrite
from nlqueries.orchestrator.sql_generation import (
    SQLGenerationResult,
    _extract_sql,
    _same_statement,
    _validate_sql,
)
from nlqueries.query_settings import current as current_settings
from nlqueries.sql_policy import _sqlglot_dialect, evaluate

if TYPE_CHECKING:
    from nlqueries.connectors.base import QueryResult
    from nlqueries.llm.client import LLMClient

_log = logging.getLogger(__name__)

#: What reads each engine's plan without running the statement. Never
#: ``ANALYZE``: that executes it, which is the cost this check exists to avoid.
_EXPLAIN_PREFIX: dict[str, str] = {
    "postgres": "EXPLAIN (FORMAT JSON) ",
    "mysql": "EXPLAIN ",
    "sqlite": "EXPLAIN QUERY PLAN ",
}

#: A plain EXPLAIN plans and returns; this bounds one that does not.
_EXPLAIN_TIMEOUT_SECONDS = 5.0
#: Below this much time before the caller's deadline no plan is read and no
#: rewrite asked for: connectors take a statement timeout in whole seconds, so
#: a shorter remainder cannot be enforced as one. The orchestrator's
#: ``_MIN_CORRECTION_SECONDS`` is the same bound for the same reason.
_MIN_SECONDS = 1.0
#: Plan rows read back. A plan is a few dozen rows; this only bounds a strange one.
_EXPLAIN_MAX_ROWS = 2000

_SQLITE_CORRELATED = re.compile(r"^CORRELATED \w+ SUBQUERY \d+")
_SQLITE_SUBQUERY = re.compile(r"^(?:CORRELATED )?\w+ SUBQUERY \d+")
_SQLITE_DERIVED = re.compile(r"^(?:CO-ROUTINE|MATERIALIZE) (\S+)")
#: ``SCAN x`` (3.36 and later, the alias where there is one) and ``SCAN TABLE
#: t AS x`` (earlier). Not ``SCAN CONSTANT ROW`` or ``SCAN (subquery-1)``.
_SQLITE_SCAN = re.compile(r"^SCAN (?:TABLE )?([^\s(]\S*)(?: AS (\S+))?")

_MYSQL_DEPENDENT = frozenset({"DEPENDENT SUBQUERY", "DEPENDENT UNION"})
_MYSQL_FULL_READ = frozenset({"ALL", "index"})
_MYSQL_OUTER = frozenset({"PRIMARY", "SIMPLE"})


@dataclass(frozen=True)
class PlanCheck:
    """What `check_plan` decided: the statement to run, and whether a model
    call was made on the way, which the caller's deadline has to account for."""

    result: SQLGenerationResult
    asked_model: bool = False


@dataclass(frozen=True)
class PerRowScan:
    """A table a subquery reads in full, once for each row of the outer query."""

    table: str
    runs: int
    rows_per_run: int

    @property
    def row_reads(self) -> int:
        return self.runs * self.rows_per_run

    def describe(self) -> str:
        return (
            f"a subquery reads {self.table} in full (about {self.rows_per_run:,} rows) "
            f"once for each of about {self.runs:,} rows of the outer query, "
            f"about {self.row_reads:,} rows read in all"
        )


def _engine(dialect: str) -> str | None:
    """The key into `_EXPLAIN_PREFIX` for *dialect*, or None when it is not checked."""
    name = _sqlglot_dialect(dialect)
    return name if name in _EXPLAIN_PREFIX else None


def _has_expression_subquery(sql: str, dialect: str) -> bool:
    """Whether *sql* has a subquery inside an expression, the only kind a plan
    re-runs per row. A derived table, a CTE or a union branch is planned once."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        return any(s.scope_type == ScopeType.SUBQUERY for s in traverse_scope(tree))
    except (sqlglot.errors.SqlglotError, ValueError):
        return False


def _aliases(sql: str, dialect: str) -> dict[str, set[str]]:
    """``{alias or name (lower): {table names (lower)}}`` for the tables *sql* reads."""
    out: dict[str, set[str]] = {}
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.SqlglotError, ValueError):
        return out
    for table in tree.find_all(exp.Table):
        if table.name:
            out.setdefault(table.alias_or_name.lower(), set()).add(table.name.lower())
    return out


# -- PostgreSQL -------------------------------------------------------------


def _postgres_root(rows: list[list[Any]]) -> dict[str, Any] | None:
    raw = rows[0][0] if rows and rows[0] else None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        root = raw[0].get("Plan")
        return root if isinstance(root, dict) else None
    return None


def _hashed(parent: dict[str, Any], subplan: dict[str, Any]) -> bool:
    """Whether *parent* probes *subplan* as a hash table built once: its
    condition names it ``hashed SubPlan 1`` where it would otherwise say
    ``SubPlan 1``."""
    name = str(subplan.get("Subplan Name") or "")
    own = json.dumps({k: v for k, v in parent.items() if k != "Plans"})
    return bool(name) and re.search(rf"hashed {re.escape(name)}\b", own) is not None


def _postgres_full_scans(node: dict[str, Any]) -> list[str]:
    """Tables a ``Seq Scan`` reads each time *node* runs, not below a nested
    subplan, which is counted on its own.

    Below a ``Hash`` or ``Materialize`` too. PostgreSQL keeps one across runs
    only when nothing beneath it depends on the outer row, and the plan says
    which only in its condition text; a hash of ``b WHERE b.k = a.k`` is
    rebuilt every run. So this can count a scan PostgreSQL would keep, and the
    rewrite it leads to must still pass its own plan.
    """
    found = []
    if node.get("Node Type") in ("Seq Scan", "Parallel Seq Scan") and node.get("Relation Name"):
        found.append(str(node["Relation Name"]))
    for child in node.get("Plans", []) or []:
        if child.get("Parent Relationship") in ("SubPlan", "InitPlan"):
            continue
        found.extend(_postgres_full_scans(child))
    return found


def _postgres_scans(rows: list[list[Any]], sizes: dict[str, int]) -> list[PerRowScan]:
    root = _postgres_root(rows)
    if root is None:
        return []
    out: list[PerRowScan] = []

    # *loops*: how many times *node* itself runs. A node's costs are for one
    # run, so a subplan's evaluations per run of its parent are the parent's
    # cost over the subplan's, and a nested loop's inner side runs once per
    # row of its outer side.
    def walk(node: dict[str, Any], loops: float) -> None:
        plans = node.get("Plans", []) or []
        outer_rows = next(
            (
                float(c.get("Plan Rows") or 0)
                for c in plans
                if c.get("Parent Relationship") == "Outer"
            ),
            1.0,
        )
        for child in plans:
            child_loops = loops
            relationship = child.get("Parent Relationship")
            if relationship == "SubPlan" and not _hashed(node, child):
                each = float(child.get("Total Cost") or 0)
                child_loops = loops * (float(node.get("Total Cost") or 0) / each if each > 0 else 0)
                runs = round(child_loops)
                for table in _postgres_full_scans(child):
                    size = sizes.get(table.lower())
                    if size and runs > 1:
                        out.append(PerRowScan(table, runs, size))
            elif relationship == "Inner" and node.get("Node Type") == "Nested Loop":
                child_loops = loops * max(outer_rows, 1.0)
            walk(child, child_loops)

    walk(root, 1.0)
    return out


# -- MySQL and MariaDB ------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mysql_scans(qr: QueryResult) -> list[PerRowScan]:
    cols = [str(c).lower() for c in qr.columns]
    if not {"select_type", "table", "type", "rows"} <= set(cols):
        return []
    recs = [dict(zip(cols, row, strict=False)) for row in qr.rows]
    # The outer query's rows, joined in plan order. A condition is attached to
    # the last table it needs, so a subquery in the WHERE runs for every row
    # that table examines, after the others' filters have applied.
    outer = [r for r in recs if r.get("select_type") in _MYSQL_OUTER]
    runs = 1.0
    for i, r in enumerate(outer):
        rows = _number(r.get("rows")) or 0.0
        share = 1.0
        if i < len(outer) - 1:
            share = (_number(r.get("filtered")) or 100.0) / 100
        runs *= rows * share
    out = []
    for r in recs:
        if r.get("select_type") in _MYSQL_DEPENDENT and r.get("type") in _MYSQL_FULL_READ:
            per_run = int(_number(r.get("rows")) or 0)
            if per_run and runs > 1:
                out.append(PerRowScan(str(r.get("table")), int(runs), per_run))
    return out


# -- SQLite -----------------------------------------------------------------


def _sqlite_scans(
    qr: QueryResult, sql: str, sizes: dict[str, int], dialect: str
) -> list[PerRowScan]:
    cols = [str(c).lower() for c in qr.columns]
    if not {"id", "parent", "detail"} <= set(cols):
        return []
    entries = [dict(zip(cols, row, strict=False)) for row in qr.rows]
    children: dict[Any, list[dict[str, Any]]] = {}
    for e in entries:
        children.setdefault(e["parent"], []).append(e)
    derived = {
        m.group(1).lower() for e in entries if (m := _SQLITE_DERIVED.match(str(e["detail"])))
    }
    aliases = _aliases(sql, dialect)

    def size(detail: str) -> tuple[str, int] | None:
        m = _SQLITE_SCAN.match(detail)
        if m is None:
            return None
        name = m.group(1).lower()
        if m.group(2) is not None:  # `SCAN TABLE t AS x` names the table
            tables = {name}
        elif name in derived:  # a derived table or CTE, sized by nothing known
            return None
        else:
            tables = aliases.get(name, {name})
        known = [(sizes[t], t) for t in tables if t in sizes]
        # One alias for two tables in different scopes: the smaller, since the
        # estimate should err towards leaving the statement alone.
        return min(known)[::-1] if known else None

    def outer_scans(e: dict[str, Any]) -> list[int]:
        detail = str(e["detail"])
        if _SQLITE_SUBQUERY.match(detail) or _SQLITE_DERIVED.match(detail):
            return []
        found = [s[1] for s in [size(detail)] if s is not None]
        for c in children.get(e["id"], []):
            found.extend(outer_scans(c))
        return found

    def inner_scans(e: dict[str, Any]) -> list[tuple[str, int]]:
        found = []
        for c in children.get(e["id"], []):
            detail = str(c["detail"])
            if _SQLITE_SUBQUERY.match(detail):
                continue  # a nested subquery runs per row of this one, counted alone
            s = size(detail)
            if s is not None:
                found.append(s)
            found.extend(inner_scans(c))
        return found

    tops = [e for e in entries if e["parent"] == 0]
    runs = max([n for e in tops for n in outer_scans(e)], default=0)
    if runs <= 1:
        return []
    return [
        PerRowScan(table, runs, per_run)
        for e in entries
        if _SQLITE_CORRELATED.match(str(e["detail"]))
        for table, per_run in inner_scans(e)
    ]


# -- Reading the plan -------------------------------------------------------


async def per_row_scans(
    sql: str,
    dialect: str,
    connector: Any,
    knowledge_base: dict[str, Any],
    timeout: float = _EXPLAIN_TIMEOUT_SECONDS,
) -> list[PerRowScan] | None:
    """The tables *sql*'s plan reads in full once per outer row, or None when
    the plan could not be read. *timeout* bounds the ``EXPLAIN``.

    An empty list for a statement with no subquery inside an expression, which
    is not sent to the database at all. The statement is checked against the
    SQL policy here, on its own, as `analyze_query` checks before its
    ``EXPLAIN``: the wrapper is not a statement the policy would pass, and
    nothing here knows where *sql* came from.
    """
    engine = _engine(dialect)
    if engine is None:
        return None
    statement = sql.strip().rstrip(";").strip()
    if not _has_expression_subquery(statement, engine):
        return []
    if not evaluate(statement, engine).allowed:
        return None
    try:
        qr = await asyncio.to_thread(
            connector.execute_query,
            _EXPLAIN_PREFIX[engine] + statement,
            timeout,
            _EXPLAIN_MAX_ROWS,
        )
    except Exception:  # noqa: BLE001
        # A wrapper that refuses the EXPLAIN form, enterprise's row filter for
        # one: the statement runs as it would have.
        _log.debug("Reading a plan failed; not checking it.", exc_info=True)
        return None
    if qr.error:
        _log.debug("Reading a plan failed; not checking it: %s", qr.error)
        return None
    sizes = _row_counts(knowledge_base)
    try:
        if engine == "postgres":
            return _postgres_scans(qr.rows, sizes)
        if engine == "mysql":
            return _mysql_scans(qr)
        return _sqlite_scans(qr, statement, sizes, engine)
    except Exception:  # noqa: BLE001 - a plan in a shape not expected is not checked
        _log.debug("Reading a plan failed; not checking it.", exc_info=True)
        return None


def _over_limit(scans: list[PerRowScan] | None) -> list[PerRowScan]:
    limit = current_settings().plan_check_min_row_reads
    return [s for s in scans or [] if s.row_reads >= limit]


def _explain_timeout(deadline: float | None) -> float | None:
    """The ``EXPLAIN``'s timeout: `_EXPLAIN_TIMEOUT_SECONDS`, or the time left
    before *deadline* (a ``time.monotonic()`` value) when that is less. None
    when less than `_MIN_SECONDS` is left, and then no plan is read."""
    if deadline is None:
        return _EXPLAIN_TIMEOUT_SECONDS
    left = deadline - time.monotonic()
    return None if left < _MIN_SECONDS else min(_EXPLAIN_TIMEOUT_SECONDS, left)


def _room_for_rewrite(started: float | None, deadline: float | None) -> bool:
    """Whether a rewrite may start: the orchestrator's `_retry_after_error` rule.

    A rewrite is a model call, a second ``EXPLAIN`` and the execution, about
    what this attempt has cost so far, so it starts only when at least that
    much time is left before *deadline*, and at least `_MIN_SECONDS`. Asked
    just before the call, after the first ``EXPLAIN`` has taken its share.
    *started* is when the attempt began; with no *deadline* there is no limit.
    """
    if deadline is None:
        return True
    now = time.monotonic()
    left = deadline - now
    if left < _MIN_SECONDS:
        return False
    return started is None or left >= now - started


async def rewrite_per_row_subqueries(
    sql: str,
    scans: list[PerRowScan],
    question: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    system: str | list[dict[str, Any]],
    attempt_count: int,
) -> SQLGenerationResult:
    """Ask the model once for *sql* with its per-row subqueries computed once.

    *system* is the prompt the statement was generated with, so the
    prompt-cache prefix is reused, and *question* anchors the rewrite to what
    was asked. The answer is validated as a first attempt is -- policy, then
    schema -- and the caller runs it only if it is valid.

    The usual rewrite, a join to a grouped derived table, differs from the
    subquery for an outer row the subquery finds nothing for: there a scalar
    subquery gives NULL, and COUNT gives 0, where an inner join drops the row.
    The prompt names that, since it is the mistake that changes the answer.
    """
    findings = "\n".join(f"- {s.describe()}" for s in scans)
    user = (
        "The database's plan for your SQL re-runs a subquery once for every row of "
        "the outer query, which will be slow:\n"
        f"{findings}\n\n"
        f"Question the SQL must answer:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Please rewrite it as a {dialect} SELECT statement that returns exactly the "
        "same rows, with each such subquery computed once rather than once per row: "
        "for example, aggregate it in a derived table or CTE grouped by the columns "
        "it is correlated on, and join to that. An outer row the subquery finds "
        "nothing for must come out as it does now (a scalar subquery over no rows "
        "gives NULL, and COUNT gives 0), so use a LEFT JOIN where that matters. Keep "
        "every filter and the columns and their order as they are. If it cannot be "
        "rewritten without changing what it returns, return the same statement "
        "unchanged. Wrap the SQL in <sql>...</sql> markers."
    )
    raw = await llm.acomplete(system, user, max_tokens=output_budget("correction"))
    rewritten = _extract_sql(raw)
    error = _validate_sql(rewritten, knowledge_base, dialect)
    return SQLGenerationResult(
        sql=rewritten,
        is_valid=error is None,
        validation_error=error,
        dialect=dialect,
        attempt_count=attempt_count + 1,
    )


async def check_plan(
    result: SQLGenerationResult,
    connector: Any,
    *,
    question: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    system: str | list[dict[str, Any]],
    started: float | None = None,
    deadline: float | None = None,
) -> PlanCheck:
    """The statement to run in place of *result*: a rewrite, or *result* itself.

    *result* comes back unchanged unless its plan re-runs a subquery per row
    over more than ``PLAN_CHECK_MIN_ROW_READS`` rows, the caller's *deadline*
    has room for a rewrite (`_room_for_rewrite`, with *started* when the
    attempt began), and the rewrite is valid, different, and its own plan shows
    no such subquery. Each ``EXPLAIN`` is bounded by the time left. Advisory
    throughout: whatever fails leaves *result* to run. The caller learns too
    whether the model was asked (`PlanCheck.asked_model`), since whatever
    runs after that call starts later than the caller planned for.

    Whenever a rewrite is due, the outcome goes into the answer's provenance as
    ``plan_rewrite``, with the statement as generated, since a structural
    rewrite is the step after validation most likely to change an answer.
    """
    if not current_settings().plan_check or not result.is_valid:
        return PlanCheck(result)
    timeout = _explain_timeout(deadline)
    if timeout is None:
        return PlanCheck(result)
    flagged = _over_limit(
        await per_row_scans(result.sql, dialect, connector, knowledge_base, timeout)
    )
    if not flagged:
        return PlanCheck(result)
    findings = [s.describe() for s in flagged]
    _log.info("The plan re-runs a subquery for every outer row: %s", "; ".join(findings))

    def outcome(reason: str | None, attempted: bool = True) -> None:
        record_plan_rewrite(
            attempted=attempted,
            changed=reason is None,
            original_sql=result.sql,
            findings=findings,
            reason=reason,
        )

    if not _room_for_rewrite(started, deadline):
        outcome("skipped: too little time before the deadline", attempted=False)
        return PlanCheck(result)
    try:
        rewritten = await rewrite_per_row_subqueries(
            result.sql,
            flagged,
            question,
            knowledge_base,
            dialect,
            llm,
            system,
            result.attempt_count,
        )
    except Exception:  # noqa: BLE001 - the statement as written still answers
        _log.warning("Rewriting a per-row subquery failed; running it as written.", exc_info=True)
        outcome("failed: the model call raised")
        return PlanCheck(result, asked_model=True)
    if not rewritten.is_valid:
        _log.info(
            "A rewrite of a per-row subquery was refused by validation and not run: %s",
            rewritten.validation_error,
        )
        outcome("rejected: refused by validation")
        return PlanCheck(result, asked_model=True)
    if _same_statement(rewritten.sql, result.sql, dialect):
        outcome("rejected: returned unchanged")
        return PlanCheck(result, asked_model=True)
    timeout = _explain_timeout(deadline)
    after = (
        None
        if timeout is None
        else await per_row_scans(rewritten.sql, dialect, connector, knowledge_base, timeout)
    )
    if after is None:
        _log.info("The plan of a rewrite could not be read; running the statement as written.")
        outcome("rejected: its plan could not be read")
        return PlanCheck(result, asked_model=True)
    if _over_limit(after):
        _log.info("A rewrite of a per-row subquery was no better; running it as written.")
        outcome("rejected: still re-runs a subquery per row")
        return PlanCheck(result, asked_model=True)
    outcome(None)
    return PlanCheck(rewritten, asked_model=True)
