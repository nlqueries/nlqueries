"""
nlqueries.orchestrator.candidate_selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Execution-verified candidate selection for the first answer (the
``candidate_selection`` query setting, ``NLQ_CANDIDATE_SELECTION``; off by
default).

Without it the first answer is one streamed call. With it, ``CANDIDATE_COUNT``
candidates are generated concurrently, each from a prompt variant in
:data:`VARIANTS`: Claude 5 models ignore temperature, so varying the prompt is
what makes the candidates differ rather than being samples of one distribution.

Each candidate goes through :func:`validate_and_repair` exactly as a lone answer
would, so it is what would have been returned on its own, with its own
provenance, match notes and choices: only the chosen one's become the answer's
(:func:`adopt`). Every valid one is then run read-only on a connector of its
own, opened from the agent's :class:`~nlqueries.connectors.loader.LookupSource`
as literal grounding does, so a row filter bound for the request applies to it
as to any query. The answer is a member of the largest group of candidates that
returned the same rows.

Measured on a public text-to-SQL benchmark before it was merged: the candidates
agreed on most questions and were often wrong together, so choosing among them
changed little, at about three times the cost of one call. It stays off by
default, for experimenting per agent; how far the candidates agree predicts
whether the answer is right, which ``provenance.candidate_selection`` records.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot

from nlqueries import config
from nlqueries.llm import output_was_cut_off
from nlqueries.orchestrator.literal_grounding import _Session
from nlqueries.orchestrator.match_notes import (
    Choice,
    _Note,
    add_choice,
    add_match_note,
    choices_for,
    collect_choices,
    collect_match_notes,
)
from nlqueries.orchestrator.prompt_assembly import AssembledPrompt, _build_static_system
from nlqueries.orchestrator.provenance import (
    Provenance,
    record_candidate_selection,
    record_literal_grounded,
    record_literal_repair,
    use_provenance,
)
from nlqueries.orchestrator.sql_generation import (
    SQLGenerationResult,
    _extract_sql,
    _validate_sql,
    validate_and_repair,
)

if TYPE_CHECKING:
    from nlqueries.connectors.loader import LookupSource
    from nlqueries.llm.client import LLMClient

_log = logging.getLogger(__name__)

_OPEN_TAG = "<sql>"
_CLOSE_TAG = "</sql>"


@dataclass(frozen=True)
class Variant:
    """How one candidate's prompt differs from the plain one."""

    name: str
    #: Whether the schema shows sample and value lists. Without them it shows
    #: names, types, keys and descriptions.
    samples: bool = True
    #: Appended to the user message.
    user_suffix: str = ""


#: The prompt variants, in candidate order; a count above their number cycles
#: through them. The first is the prompt a lone answer gets.
VARIANTS: tuple[Variant, ...] = (
    Variant("prompt"),
    Variant("no_samples", samples=False),
    Variant(
        "tables_first",
        user_suffix=(
            "\n\nBefore writing the SQL, list the tables and columns you will use, "
            "then write the SQL between <sql> and </sql>."
        ),
    ),
)


@dataclass
class Candidate:
    """One candidate answer and what became of it."""

    index: int
    variant: str
    #: The model's whole reply.
    text: str = ""
    #: The SQL as the model wrote it.
    raw_sql: str = ""
    #: Valid as written, with no repair of any kind.
    clean: bool = False
    result: SQLGenerationResult | None = None
    provenance: Provenance = field(default_factory=Provenance)
    #: The match notes and choices its grounding left, for :func:`adopt`.
    notes: list[_Note] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    #: "no_reply", "invalid", "valid", "awaiting_choice", "ok", "error" or
    #: "timeout".
    status: str = "no_reply"
    rows: frozenset[tuple[Any, ...]] | None = None
    row_count: int | None = None
    error: str | None = None


async def choose_first_answer(
    *,
    prompt: AssembledPrompt,
    knowledge_base: dict[str, Any],
    llm: LLMClient,
    dialect: str,
    question: str,
    source: LookupSource,
    lookups: LookupSource | None,
    cache: bool,
) -> Candidate:
    """Generate, validate, run and choose; see the module docstring.

    Records ``candidate_selection`` in provenance. The chosen candidate's own
    grounding records are left for :func:`adopt`, which the caller makes where
    the request's match notes and choices are collected. Raises the first
    generation error when no candidate got a reply.
    """
    variants = [VARIANTS[i % len(VARIANTS)] for i in range(max(1, config.CANDIDATE_COUNT))]
    systems: dict[bool, Any] = {}

    def system_for(variant: Variant) -> Any:
        if variant.samples not in systems:
            static = (
                prompt.static_system
                if variant.samples
                else _build_static_system(_without_values(knowledge_base))
            )
            systems[variant.samples] = AssembledPrompt(
                static_system=static,
                dynamic_context=prompt.dynamic_context,
                user_question=prompt.user_question,
            ).system_blocks(cache=cache)
        return systems[variant.samples]

    candidates = [Candidate(index=i, variant=v.name) for i, v in enumerate(variants)]
    errors = await asyncio.gather(
        *(
            _answer(c, v, system_for(v), prompt, knowledge_base, dialect, llm, lookups, question)
            for c, v in zip(candidates, variants, strict=True)
        )
    )
    if all(c.status == "no_reply" for c in candidates):
        raise next(e for e in errors if e is not None)

    await asyncio.gather(*(_execute(c, source, dialect) for c in candidates if c.status == "valid"))

    chosen, reason, groups = _pick(candidates, dialect)
    record_candidate_selection(
        {
            "count": len(candidates),
            "executed": sum(1 for c in candidates if c.status == "ok"),
            "groups": groups,
            "chosen": chosen.index,
            "reason": reason,
            "candidates": [_summary(c) for c in candidates],
        }
    )
    return chosen


def adopt(chosen: Candidate) -> None:
    """Record what the chosen candidate's grounding recorded for itself, as a
    lone answer would have: its literals grounded and literal repair into the
    request's provenance, its match notes and choices into the collections
    open where this is called."""
    for entry in chosen.provenance.literals_grounded:
        record_literal_grounded(**entry)
    if chosen.provenance.literal_repair is not None:
        repair = chosen.provenance.literal_repair
        record_literal_repair(repair["changed"], repair["notes"], repair.get("reason"))
    for note, values in chosen.notes:
        add_match_note(note, values)
    for choice in chosen.choices:
        add_choice(choice)


def _without_values(knowledge_base: dict[str, Any]) -> dict[str, Any]:
    """*knowledge_base* with no sample or value lists on its columns."""
    schema = knowledge_base.get("schema", {})
    tables = [
        {
            **table,
            "columns": [
                {k: v for k, v in col.items() if k not in ("samples", "values_complete")}
                for col in table.get("columns", [])
            ],
        }
        for table in schema.get("tables", [])
    ]
    return {**knowledge_base, "schema": {**schema, "tables": tables}}


def _cut_off(text: str) -> bool:
    """Whether *text* stopped at its output budget before its statement was
    whole, as the streamed answer is judged: inside ``<sql>`` whatever the
    provider said, or cut off with no closed block at all."""
    opened = text.rfind(_OPEN_TAG)
    inside = opened != -1 and _CLOSE_TAG not in text[opened:]
    closed = opened != -1 and not inside
    return inside or (output_was_cut_off() and not closed)


async def _answer(
    candidate: Candidate,
    variant: Variant,
    system: Any,
    prompt: AssembledPrompt,
    knowledge_base: dict[str, Any],
    dialect: str,
    llm: LLMClient,
    lookups: LookupSource | None,
    question: str,
) -> BaseException | None:
    """Generate the candidate, then validate it through the lone answer's
    path, in one task: whether the reply was cut off is read from the call
    just made in it. Its provenance, notes and choices are its own; only the
    chosen one's become the answer's. Returns the generation error, if any."""
    try:
        candidate.text = await llm.acomplete(system, prompt.user_content() + variant.user_suffix)
    except Exception as exc:  # noqa: BLE001 - recorded; raised if every call failed
        candidate.error = str(exc)[:500]
        return exc
    cut_off = _cut_off(candidate.text)
    candidate.raw_sql = _extract_sql(candidate.text).strip()
    candidate.clean = (
        bool(candidate.raw_sql)
        and _validate_sql(candidate.raw_sql, knowledge_base, dialect) is None
    )
    with (
        use_provenance(candidate.provenance),
        collect_match_notes() as notes,
        collect_choices() as choices,
    ):
        candidate.result = await validate_and_repair(
            candidate.raw_sql,
            knowledge_base,
            dialect,
            llm,
            system,
            lookups=lookups,
            question=question,
            cut_off=cut_off,
        )
    candidate.notes, candidate.choices = list(notes), list(choices)
    candidate.status = "valid" if candidate.result.is_valid else "invalid"
    return None


async def _execute(candidate: Candidate, source: LookupSource, dialect: str) -> None:
    """Run the candidate on a connector of its own, released when it is done.

    Not one that leaves a value for the user to choose: nothing runs until
    they have, since any value taken for them would be a guess."""
    if candidate.result is None:
        return
    if choices_for(candidate.choices, candidate.result.sql, dialect):
        candidate.status = "awaiting_choice"
        return
    timeout = config.CANDIDATE_EXEC_TIMEOUT_S
    session = _Session(source, time.monotonic() + timeout, statement_timeout_s=timeout)
    try:
        rows = await session.run_sql(candidate.result.sql)
    except Exception as exc:  # noqa: BLE001 - recorded; an error never wins
        candidate.status, candidate.error = "error", str(exc)[:500]
        return
    finally:
        session.close()
    if rows is None:
        candidate.status, candidate.error = "timeout", f"no result within {timeout:g} s"
        return
    # Ordered before the cap, so two candidates returning the same large set in
    # different orders are compared on the same rows.
    kept = sorted(rows, key=repr)[: config.CANDIDATE_MAX_ROWS]
    candidate.rows, candidate.row_count, candidate.status = frozenset(kept), len(kept), "ok"


def _normalised_sql(sql: str, dialect: str) -> str:
    try:
        return str(sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect, normalize=True))
    except Exception:  # noqa: BLE001
        return sql


def _pick(candidates: list[Candidate], dialect: str) -> tuple[Candidate, str, list[int]]:
    """The answer, why, and the sizes of the groups of identical results.

    The largest group wins; ties go to a non-empty result over an empty one, then
    to a group holding a candidate that was valid without any repair, then to
    the shortest normalised SQL. A candidate whose run failed or timed out, or
    that was not run, is in no group and never wins. With no candidate run, the
    first one that got a reply, as without selection.
    """
    executed = [c for c in candidates if c.status == "ok"]
    if not executed:
        fallback = next((c for c in candidates if c.result is not None), candidates[0])
        return fallback, "no candidate executed: the first candidate, as without selection", []

    def member_key(c: Candidate) -> tuple[int, int, int]:
        sql = c.result.sql if c.result is not None else c.raw_sql
        return (0 if c.clean else 1, len(_normalised_sql(sql, dialect)), c.index)

    by_rows: dict[frozenset[tuple[Any, ...]], list[Candidate]] = {}
    for c in executed:
        by_rows.setdefault(c.rows or frozenset(), []).append(c)

    def group_key(members: list[Candidate]) -> tuple[int, int, int, int, int]:
        best = min(members, key=member_key)
        return (-len(members), 0 if members[0].rows else 1, *member_key(best))

    ranked = sorted(by_rows.values(), key=group_key)
    winners = ranked[0]
    chosen = min(winners, key=member_key)
    reason = f"{len(winners)} of {len(executed)} executed candidates returned the same rows"
    if len(ranked) > 1 and len(ranked[1]) == len(winners):
        first, second = group_key(winners), group_key(ranked[1])
        if first[1] != second[1]:
            reason += "; tie on size, won by a non-empty result"
        elif first[2] != second[2]:
            reason += "; tie on size, won by a candidate valid without repair"
        else:
            reason += "; tie on size, won by the shorter SQL"
    return chosen, reason, sorted((len(m) for m in by_rows.values()), reverse=True)


def _summary(candidate: Candidate) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "variant": candidate.variant,
        "sql": candidate.result.sql if candidate.result is not None else candidate.raw_sql,
        "status": candidate.status,
    }
    if candidate.status == "ok":
        entry["rows"] = candidate.row_count
    else:
        entry["error"] = candidate.error or (
            candidate.result.validation_error if candidate.result is not None else None
        )
    return entry
