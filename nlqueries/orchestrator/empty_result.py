"""
nlqueries.orchestrator.empty_result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What to do when a statement returns no rows because a value it names is not
stored (owner feedback F19, 2026-11-17).

The question asks for flights to 'DALLAS FORT WORTH'; the model writes
``city_name = 'DALLAS FORT WORTH'``; the table holds 'DALLAS' and 'FORT WORTH'
as two cities. Grounding found no stored value to match it, the statement ran,
nothing came back, and all the user heard was that a filter may be narrow. Here,
only after such an empty result, each literal that matches no stored value is
given the stored values it may mean (:func:`~nlqueries.orchestrator.
literal_grounding.empty_result_literals`), and then by ``NLQ_AMBIGUITY``:

- ``ask``: the user is asked to choose, with the existing ``choices`` shape;
  their answer comes back as ``chosen_values`` and grounding uses it;
- ``pick``: one literal with exactly one candidate is replaced and the
  statement run once more, with a note saying so; anything else is not
  guessed, and a note names the literal, the column and what it may mean;
- no candidates: the note names the literal that matched nothing.

The specific note replaces the generic one. One rerun at most, on the same
connector (so the same row filters apply), within the time left.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nlqueries.orchestrator.literal_grounding import empty_result_literals, with_literal
from nlqueries.orchestrator.match_notes import Choice, choice_key
from nlqueries.query_settings import current as current_settings

if TYPE_CHECKING:
    from nlqueries.connectors.base import QueryResult
    from nlqueries.connectors.loader import LookupSource
    from nlqueries.orchestrator.sql_generation import SQLGenerationResult

_log = logging.getLogger(__name__)


@dataclass
class EmptyResultFollowup:
    """What an empty result became: a statement run again, notes, choices."""

    result: SQLGenerationResult
    qr: QueryResult
    #: Notes that replace the generic empty-result note.
    notes: list[str] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _listed(values: tuple[str, ...]) -> str:
    return ", ".join(_quoted(v) for v in values)


async def follow_up_empty_result(
    result: SQLGenerationResult,
    qr: QueryResult,
    *,
    connector: Any,
    kb: dict[str, Any],
    dialect: str,
    lookups: LookupSource | None,
    run_timeout: float | None,
    may_rerun: bool,
) -> EmptyResultFollowup | None:
    """Follow up an empty *qr* from *result*; ``None`` when there is nothing to
    add (no literal matches nothing stored), so the generic note stands.

    *may_rerun* is whether there is time left for one more statement, and
    *run_timeout* the timeout to run it with.
    """
    if qr.error is not None or qr.rows or qr.truncated:
        return None
    unmatched = await empty_result_literals(result.sql, kb, dialect, lookups)
    if not unmatched:
        return None

    followup = EmptyResultFollowup(result=result, qr=qr)
    offered = [u for u in unmatched if u.candidates]
    if current_settings().ambiguity == "ask" and offered:
        for u in offered:
            followup.choices.append(
                Choice(choice_key(u.table, u.column, u.literal), u.column, u.literal, u.candidates)
            )
            followup.notes.append(
                f"No {u.column} is {_quoted(u.literal)}. Did you mean: {_listed(u.candidates)}?"
            )
    else:
        single = [u for u in offered if len(u.candidates) == 1]
        if len(offered) == 1 and len(single) == 1 and may_rerun:
            u = single[0]
            rerun = await _rerun(
                result, u.position, u.candidates[0], connector, dialect, run_timeout
            )
            if rerun is not None:
                followup.result, followup.qr = rerun
                followup.notes.append(
                    f"No {u.column} is {_quoted(u.literal)}; the answer uses "
                    f"{_quoted(u.candidates[0])}, the stored value it is closest to."
                )
                return followup
        for u in offered:
            followup.notes.append(
                f"No {u.column} is {_quoted(u.literal)}. Stored values it may mean: "
                f"{_listed(u.candidates)}."
            )
    for u in unmatched:
        if not u.candidates:
            followup.notes.append(
                f"No rows matched: no stored {u.column} is {_quoted(u.literal)}, and no "
                "stored value is close to it."
            )
    return followup


async def _rerun(
    result: SQLGenerationResult,
    position: int,
    value: str,
    connector: Any,
    dialect: str,
    run_timeout: float | None,
) -> tuple[SQLGenerationResult, QueryResult] | None:
    """*result*'s statement with the literal at *position* replaced by *value*,
    run once; ``None`` when it cannot be built, fails, or is empty too."""
    sql = with_literal(result.sql, dialect, position, value)
    if sql is None:
        return None
    try:
        retried = await asyncio.to_thread(connector.execute_query, sql, run_timeout)
    except Exception:  # noqa: BLE001 — the empty result stands, with its note
        _log.warning("Running a statement again with a stored value failed.", exc_info=True)
        return None
    if retried.error is not None or not retried.rows:
        return None
    return dataclasses.replace(result, sql=sql), retried
