"""
nlqueries.orchestrator.literal_grounding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Match string literals in a statement's WHERE clause to what the column stores.

A model writes ``status = 'legal'`` where the data has ``'Legal'``, or copies
``' = '`` from a hint where the column holds ``'='``. The statement is valid,
runs, and returns no rows: a wrong answer with nothing to say so. Before the
statement runs, each such comparison is checked against the database:

- the literal exists as written: left alone;
- exactly one stored value matches it ignoring case and surrounding spaces,
  or failing that, also ignoring punctuation at either end (see
  :func:`_fold`): the literal is replaced with that value, and the
  substitution is recorded in provenance;
- none, or several: the statement is left alone, and a note lists up to five
  nearby stored values for the LLM repair step, if one runs.

Read-only and bounded: at most ``_MAX_LITERALS`` comparisons per statement,
``_LOOKUP_TIMEOUT_S`` per lookup, ``_TOTAL_BUDGET_S`` for the whole step. Any
error skips the literal (or the step) and is logged at DEBUG; grounding can
make a statement better, never fail it. Outcomes are cached per database for
``_CACHE_TTL_S``, since a workload repeats its literals.

Safe under concurrent calls from different threads and event loops. Each call
opens its own connector rather than sharing the pooled one (see
:func:`~nlqueries.connectors.loader.lookup_source` for the deadlock sharing
caused), runs every database call on a daemon thread, and stops waiting for one
at the step's deadline: a stuck lookup is abandoned, never awaited, and its
connector is closed when it returns.

Only columns the knowledge base knows, of a text type, that are not keys: a
key's value is an identifier, and "the nearest identifier" is not a fix.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from nlqueries.orchestrator.provenance import record_literal_grounded

if TYPE_CHECKING:
    from nlqueries.connectors.loader import LookupSource

_log = logging.getLogger(__name__)

#: Most comparisons grounded per statement.
_MAX_LITERALS = 8
#: Server-side budget for one lookup, where the connector supports one.
_LOOKUP_TIMEOUT_S = 1.0
#: Wall-clock budget for the whole step, opening the connector included. No
#: database call is waited for past it.
_TOTAL_BUDGET_S = 3.0
#: Longer literals are free text, not a stored category value.
_MAX_LITERAL_CHARS = 100
#: Nearby values offered in a note.
_MAX_NEARBY = 5
#: Above this many distinct values, no similarity pass over the column.
_MAX_DISTINCT_FOR_SIMILARITY = 2000
#: Stored values containing the literal that the punctuation pass compares.
_MAX_CONTAINING = 20
#: What :func:`_fold` strips from either end of a value, besides whitespace.
_FOLD_ENDS = ".,;:!?'\"()[]{}-"
#: Shorter than this once folded, a literal is matched exactly or not at all:
#: "?" or "-" folds to nothing, and "nothing" is not a match.
_MIN_FOLDED_CHARS = 3

#: Outcomes kept, across databases; the least recently used goes first.
_MAX_CACHED = 10_000
#: How long an outcome is trusted. A value added to a column is seen after this.
_CACHE_TTL_S = 900.0

_TEXT_TYPE_MARKERS = ("CHAR", "TEXT", "STRING", "CLOB")


@dataclass
class GroundingResult:
    """The statement after grounding, and what happened on the way."""

    sql: str
    #: ``(table, column, before, after)`` for each literal replaced.
    substitutions: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: One line per literal no stored value matched, for the repair prompt.
    notes: list[str] = field(default_factory=list)
    #: Whether a literal that matched no stored value came with nearby ones:
    #: the case a repair can act on.
    offers_nearby: bool = False


@dataclass(frozen=True)
class _Outcome:
    """What a column holds for one literal."""

    kind: str  # "found" | "one" | "none" | "many"
    values: tuple[str, ...] = ()


# Keyed by the source's key first, so two databases never share an answer.
_CacheKey = tuple[str, str, str, str]
_cache: OrderedDict[_CacheKey, tuple[float, _Outcome]] = OrderedDict()
# Threading, not asyncio: callers run on different event loops in different
# threads. Held around the dict operations alone, never around a database call.
_cache_lock = threading.Lock()


def _cached(key: _CacheKey) -> _Outcome | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[0] > _CACHE_TTL_S:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return entry[1]


def _remember(key: _CacheKey, outcome: _Outcome) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), outcome)
        _cache.move_to_end(key)
        while len(_cache) > _MAX_CACHED:
            _cache.popitem(last=False)


async def ground_literals(
    sql: str,
    knowledge_base: dict[str, Any],
    dialect: str,
    source: LookupSource | None,
) -> GroundingResult:
    """Ground the string literals *sql* compares columns to; see the module docstring.

    Returns *sql* unchanged when *source* is ``None``, when nothing needs
    grounding, or on any error. Opens a connector from *source* only for a
    literal the cache cannot answer, and closes it before returning or, if a
    lookup was abandoned, when that lookup returns.
    """
    if source is None or not sql.strip():
        return GroundingResult(sql=sql)
    try:
        statement = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001
        _log.debug("Literal grounding skipped: the statement did not parse.", exc_info=True)
        return GroundingResult(sql=sql)
    if statement is None:
        return GroundingResult(sql=sql)

    tables = _kb_tables(knowledge_base)
    if not tables:
        return GroundingResult(sql=sql)

    result = GroundingResult(sql=sql)
    session = _Session(source, time.monotonic() + _TOTAL_BUDGET_S)
    changed = False
    try:
        for count, (table_node, column_node, literal_node) in enumerate(
            _comparisons(statement, tables)
        ):
            if count >= _MAX_LITERALS or not session.usable:
                break
            outcome = await _outcome(session, dialect, table_node, column_node, literal_node.this)
            if outcome is not None:
                changed |= _apply(result, outcome, table_node, column_node, literal_node)
    finally:
        session.close()
    if changed:
        try:
            result.sql = statement.sql(dialect=dialect)
        except Exception:  # noqa: BLE001
            _log.debug("Literal grounding skipped: the statement did not render.", exc_info=True)
            return GroundingResult(sql=sql)
    return result


async def _outcome(
    session: _Session, dialect: str, table: exp.Table, column: exp.Column, literal: str
) -> _Outcome | None:
    """What *table*.*column* holds for *literal*, from the cache or the database;
    ``None`` when the lookup failed or ran out of time, which is not cached."""
    key = (session.key, _table_key(table), column.name, literal)
    outcome = _cached(key)
    if outcome is not None:
        return outcome
    try:
        outcome = await _look_up(session, dialect, table, column, literal)
    except Exception:  # noqa: BLE001
        _log.debug(
            "Literal grounding skipped %s.%s = %r: the lookup failed.",
            table.name,
            column.name,
            literal,
            exc_info=True,
        )
        return None
    if outcome is not None:
        _remember(key, outcome)
    return outcome


def _apply(
    result: GroundingResult,
    outcome: _Outcome,
    table: exp.Table,
    column: exp.Column,
    literal_node: exp.Literal,
) -> bool:
    """Record *outcome* on *result*; ``True`` when the literal was replaced."""
    literal = literal_node.this
    if outcome.kind == "one":
        stored = outcome.values[0]
        literal_node.set("this", stored)
        result.substitutions.append((table.name, column.name, literal, stored))
        record_literal_grounded(table.name, column.name, literal, stored)
        return True
    if outcome.kind == "none":
        result.offers_nearby = result.offers_nearby or bool(outcome.values)
        nearby = ", ".join(_quote(v) for v in outcome.values)
        result.notes.append(
            f"No row has {column.name} = {_quote(literal)}"
            + (f"; nearby values: {nearby}." if nearby else ".")
        )
    elif outcome.kind == "many":
        several = ", ".join(_quote(v) for v in outcome.values)
        result.notes.append(
            f"No row has {column.name} = {_quote(literal)} exactly; ignoring case, "
            f"spacing and punctuation at either end it matches several stored values: "
            f"{several}."
        )
    return False


# ---------------------------------------------------------------------------
# Finding the comparisons
# ---------------------------------------------------------------------------


def _kb_tables(knowledge_base: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{table name (lower): {column name (lower): column entry}}`` from the KB."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for table in knowledge_base.get("schema", {}).get("tables", []) or []:
        name = str(table.get("name") or "").lower()
        if not name:
            continue
        out[name] = {
            str(col.get("name") or "").lower(): col for col in table.get("columns", []) or []
        }
    return out


def _groundable(column: dict[str, Any]) -> bool:
    """A text column that is not a key."""
    if column.get("is_primary_key") or column.get("is_foreign_key"):
        return False
    col_type = str(column.get("type") or "").upper()
    # An empty type is SQLite's untyped column; the lookups are text lookups and
    # an error on any other type is skipped.
    return not col_type or any(marker in col_type for marker in _TEXT_TYPE_MARKERS)


def _string_literal(node: exp.Expr | None) -> exp.Literal | None:
    if isinstance(node, exp.Literal) and node.is_string:
        literal = node.this
        if isinstance(literal, str) and 0 < len(literal) <= _MAX_LITERAL_CHARS:
            return node
    return None


def _column_and_literals(node: exp.Expr) -> tuple[exp.Column, list[exp.Literal]] | None:
    """The column and string literals of ``col = 'x'``, ``col IN ('x', 'y')`` or
    ``col LIKE 'x'`` with no wildcard; ``None`` for anything else."""
    if isinstance(node, exp.EQ):
        left, right = node.this, node.expression
        if isinstance(right, exp.Column):
            left, right = right, left
        literal = _string_literal(right)
        if isinstance(left, exp.Column) and literal is not None:
            return left, [literal]
        return None
    if isinstance(node, exp.In):
        if not isinstance(node.this, exp.Column) or node.args.get("query") is not None:
            return None
        literals = [lit for e in node.expressions if (lit := _string_literal(e)) is not None]
        return (node.this, literals) if literals else None
    if isinstance(node, exp.Like):
        literal = _string_literal(node.expression)
        if (
            isinstance(node.this, exp.Column)
            and literal is not None
            and "%" not in literal.this
            and "_" not in literal.this
        ):
            return node.this, [literal]
    return None


def _source_table(
    column: exp.Column, scope: Scope, tables: dict[str, dict[str, dict[str, Any]]]
) -> exp.Table | None:
    """The table *column* reads, resolved through this scope's FROM and JOINs."""
    sources = {
        alias: source for alias, source in scope.sources.items() if isinstance(source, exp.Table)
    }
    if column.table:
        source = sources.get(column.table)
        return source if isinstance(source, exp.Table) else None
    owners = [
        source
        for source in sources.values()
        if column.name.lower() in tables.get(source.name.lower(), {})
    ]
    return owners[0] if len(owners) == 1 else None


def _comparisons(
    statement: exp.Expr, tables: dict[str, dict[str, dict[str, Any]]]
) -> Iterator[tuple[exp.Table, exp.Column, exp.Literal]]:
    """``(table, column, literal)`` for every groundable comparison, scope by scope."""
    try:
        scopes = traverse_scope(statement)
    except Exception:  # noqa: BLE001
        _log.debug("Literal grounding skipped: scopes could not be built.", exc_info=True)
        return
    for scope in scopes:
        for node in scope.find_all(exp.EQ, exp.In, exp.Like):
            found = _column_and_literals(node)
            if found is None:
                continue
            column, literals = found
            table = _source_table(column, scope, tables)
            if table is None:
                continue
            entry = tables.get(table.name.lower(), {}).get(column.name.lower())
            if entry is None or not _groundable(entry):
                continue
            for literal in literals:
                yield table, column, literal


# ---------------------------------------------------------------------------
# Looking the value up
# ---------------------------------------------------------------------------


def _table_key(table: exp.Table) -> str:
    return ".".join(part for part in (table.catalog, table.db, table.name) if part).lower()


def _target(table: exp.Table, column: exp.Column) -> tuple[exp.Table, exp.Column]:
    """The table without its alias and the column without its qualifier."""
    bare_table = table.copy()
    bare_table.set("alias", None)
    return bare_table, exp.Column(this=column.this.copy())


class _DaemonThreads(concurrent.futures.Executor):
    """One daemon thread per call.

    Not a ThreadPoolExecutor, whose threads the interpreter joins at exit: a
    lookup stuck in the database would hang the process on its way out. The
    future is running from the moment it is returned, so it cannot be cancelled
    before its call starts: every call submitted is made, which is what lets a
    session count the calls it has out.
    """

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        future.set_running_or_notify_cancel()

        def _call() -> None:
            try:
                value = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - handed to whoever awaits it
                future.set_exception(exc)
            else:
                future.set_result(value)

        threading.Thread(target=_call, name="literal-grounding", daemon=True).start()
        return future


_THREADS = _DaemonThreads()
_ABANDONED = object()


class _Session:
    """One grounding call's own connector.

    Opened on the first lookup the cache cannot answer. Every database call, the
    open included, runs on a daemon thread and is waited for until the step's
    deadline and no longer. A call still running then is abandoned and the
    session takes no more lookups. The connector is closed by whichever comes
    last, :meth:`close` or the last call returning, so never under a statement
    still running on it.
    """

    def __init__(self, source: LookupSource, deadline: float) -> None:
        self.key = source.key
        self._source = source
        self._deadline = deadline
        self._connector: Any = None
        self._lock = threading.Lock()
        self._running = 0
        self._closed = False
        self._failed = False

    @property
    def usable(self) -> bool:
        return not self._failed and time.monotonic() < self._deadline

    async def run(self, statement: exp.Expr, dialect: str, max_rows: int) -> list[Any] | None:
        """The first column of *statement*'s rows; ``None`` when out of time."""
        sql = statement.sql(dialect=dialect)
        if self._connector is None:
            try:
                if await self._in_thread(self._open) is _ABANDONED:
                    return None
            except Exception:
                self._failed = True  # no connector, so nothing more to try
                raise
        rows = await self._in_thread(functools.partial(self._execute, sql, max_rows))
        return None if rows is _ABANDONED else rows

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle = self._running == 0
        if idle:
            self._release()

    async def _in_thread(self, call: Callable[[], Any]) -> Any:
        remaining = self._deadline - time.monotonic()
        if self._failed or remaining <= 0:
            return _ABANDONED
        with self._lock:
            self._running += 1
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_THREADS, self._counted, call), remaining
            )
        except TimeoutError:
            # Not left to the deadline check: asyncio fires a timer up to its
            # clock resolution early, and the abandoned call still holds the
            # connector, so a next lookup would share it.
            self._failed = True
            _log.debug("Literal grounding abandoned a lookup still running at its deadline.")
            return _ABANDONED

    def _counted(self, call: Callable[[], Any]) -> Any:
        try:
            return call()
        finally:
            with self._lock:
                self._running -= 1
                release = self._closed and self._running == 0
            if release:
                self._release()

    def _open(self) -> bool:
        self._connector = self._source.open()
        return True

    def _execute(self, sql: str, max_rows: int) -> list[Any]:
        result = self._connector.execute_query(sql, _LOOKUP_TIMEOUT_S, max_rows=max_rows)
        if result.error:
            raise RuntimeError(result.error)
        return [row[0] for row in result.rows]

    def _release(self) -> None:
        with self._lock:
            connector, self._connector = self._connector, None
        if connector is not None:
            try:
                connector.close()
            except Exception:  # noqa: BLE001
                _log.debug("Literal grounding could not close its connector.", exc_info=True)


def _normalised(expression: exp.Expr) -> exp.Expr:
    return exp.Lower(this=exp.Trim(this=expression))


def _loose_match(col: exp.Column, value: exp.Literal, dialect: str) -> exp.Expr:
    """*col* equal to *value* ignoring case and surrounding spaces.

    On SQLite, ``TRIM(col) = TRIM(value) COLLATE NOCASE`` rather than ``LOWER``
    on both sides. ``LOWER`` scaled badly with concurrent scans on separate
    connections: on a 200,000-row table, four threads took 9.8 s for 20 scans
    each where one thread took 0.6 s for its 20, and with the collation four
    took 0.28 s. NOCASE folds ASCII letters only, as SQLite's ``LOWER`` does.
    """
    if dialect == "sqlite":
        return exp.EQ(
            this=exp.Trim(this=col),
            expression=exp.Collate(this=exp.Trim(this=value), expression=exp.Var(this="NOCASE")),
        )
    return exp.EQ(this=_normalised(col), expression=_normalised(value))


def _fold(value: str) -> str:
    """*value* lowercased, its runs of whitespace collapsed to one space, and
    the characters of ``_FOLD_ENDS`` and spaces stripped from both ends.

    Only the ends: ``'multi-dimensional data?'`` folds to
    ``'multi-dimensional data'``, hyphen kept.
    """
    return " ".join(value.lower().split()).strip(_FOLD_ENDS + " ")


def _containing(source: exp.Table, col: exp.Column, literal: str, limit: int) -> exp.Expr:
    """Distinct stored values that contain *literal*, ignoring case."""
    return (
        exp.select(col.copy())
        .distinct()
        .from_(source.copy())
        .where(
            exp.Like(
                this=exp.Lower(this=col.copy()),
                expression=exp.Literal.string(f"%{literal.strip().lower()}%"),
            )
        )
        .limit(limit)
    )


async def _look_up(
    session: _Session,
    dialect: str,
    table: exp.Table,
    column: exp.Column,
    literal: str,
) -> _Outcome | None:
    """What *table*.*column* holds for *literal*; ``None`` if time ran out."""
    source, col = _target(table, column)
    value = exp.Literal.string(literal)

    exact = exp.select("1").from_(source.copy()).where(exp.EQ(this=col.copy(), expression=value))
    found = await session.run(exact.limit(1), dialect, max_rows=1)
    if found is None:
        return None
    if found:
        return _Outcome("found")

    loose = (
        exp.select(col.copy())
        .distinct()
        .from_(source.copy())
        .where(_loose_match(col.copy(), value.copy(), dialect))
        .limit(2)
    )
    rows = await session.run(loose, dialect, max_rows=2)
    if rows is None:
        return None
    matches = [str(v) for v in rows if v is not None]
    if len(matches) == 1:
        return _Outcome("one", (matches[0],))
    if len(matches) > 1:
        return _Outcome("many", tuple(matches))

    # Third pass: the stored value differs by punctuation at an end, as a
    # title stored with its question mark does from the same title quoted
    # without it. Compared in Python, over the values that contain the literal.
    containing: list[str] | None = None
    folded = _fold(literal)
    if len(folded) >= _MIN_FOLDED_CHARS:
        rows = await session.run(
            _containing(source, col, literal, _MAX_CONTAINING), dialect, _MAX_CONTAINING
        )
        if rows is None:
            return None
        containing = [str(v) for v in rows if v is not None]
        same = [v for v in containing if _fold(v) == folded]
        if len(same) == 1:
            return _Outcome("one", (same[0],))
        if len(same) > 1:
            return _Outcome("many", tuple(same[:_MAX_NEARBY]))

    nearby = await _nearby(session, dialect, source, col, literal, containing)
    # Not "none" without its nearby values: cached, the note would stay that
    # way for as long as the outcome is kept.
    return None if nearby is None else _Outcome("none", nearby)


async def _nearby(
    session: _Session,
    dialect: str,
    source: exp.Table,
    col: exp.Column,
    literal: str,
    containing: list[str] | None = None,
) -> tuple[str, ...] | None:
    """Up to five stored values near *literal*: containing it, then similar to
    it. *containing* is the first pass's values when the caller has already
    fetched them, so the query is not run twice. ``None`` if time ran out
    before the first pass."""
    if containing is None:
        rows = await session.run(
            _containing(source, col, literal, _MAX_NEARBY), dialect, _MAX_NEARBY
        )
        if rows is None:
            return None
        containing = [str(v) for v in rows if v is not None]
    found = list(containing[:_MAX_NEARBY])
    if len(found) >= _MAX_NEARBY:
        return tuple(found[:_MAX_NEARBY])

    every = (
        exp.select(col.copy())
        .distinct()
        .from_(source.copy())
        .limit(_MAX_DISTINCT_FOR_SIMILARITY + 1)
    )
    rows = await session.run(every, dialect, _MAX_DISTINCT_FOR_SIMILARITY + 1)
    values = [str(v) for v in rows or () if v is not None]
    if rows is not None and len(values) <= _MAX_DISTINCT_FOR_SIMILARITY:
        by_lower = {v.lower(): v for v in values}
        for close in difflib.get_close_matches(
            literal.strip().lower(), list(by_lower), n=_MAX_NEARBY, cutoff=0.5
        ):
            if by_lower[close] not in found:
                found.append(by_lower[close])
    return tuple(found[:_MAX_NEARBY])


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
