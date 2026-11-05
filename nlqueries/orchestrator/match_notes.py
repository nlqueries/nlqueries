"""
nlqueries.orchestrator.match_notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What the user should know about how a value the question named was matched.

Literal grounding sometimes decides for the user: ``'active'`` is stored as both
``'Active'`` and ``'ACTIVE'``, so the statement compares with both; or, with
values matched by ``contains``, ``'Schools'`` is part of several stored values
and the closest is taken. The answer is only as right as that decision, so it
is said with the answer, as a note beside the rows.

Grounding runs deep inside validation, which does not know who asked, so the
notes are collected per request, the way provenance is: the orchestrator opens
a collection around a question with :func:`collect_match_notes`, and
:func:`add_match_note` adds to whichever collection is open, or to none. Each
note carries the stored values it is about, and :func:`notes_for` keeps only
those the statement finally answered with still compares with: a later step
may replace the statement, and a note about one that did not run is not true
of the answer.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.expressions as exp

_Note = tuple[str, tuple[str, ...]]

_collection: contextvars.ContextVar[list[_Note] | None] = contextvars.ContextVar(
    "nlqueries_match_notes", default=None
)


@contextmanager
def collect_match_notes() -> Iterator[list[_Note]]:
    """Collect, into the list it yields, the notes added within."""
    notes: list[_Note] = []
    token = _collection.set(notes)
    try:
        yield notes
    finally:
        _collection.reset(token)


def add_match_note(note: str, values: tuple[str, ...]) -> None:
    """Add *note*, about the stored *values*, to the open collection, once; no
    collection, no note."""
    notes = _collection.get()
    if notes is not None and (note, values) not in notes:
        notes.append((note, values))


def notes_for(notes: list[_Note], sql: str, dialect: str) -> list[str]:
    """The notes of *notes* whose values are all string literals of *sql*."""
    if not notes:
        return []
    try:
        present = {
            node.this
            for node in sqlglot.parse_one(sql, read=dialect).find_all(exp.Literal)
            if node.is_string
        }
    except Exception:  # noqa: BLE001 - unparsable: nothing is shown to be there
        return []
    return [note for note, values in notes if all(v in present for v in values)]


# ---------------------------------------------------------------------------
# Choices: when the user is asked which value was meant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """A value the question named that is part of several different stored
    values, left for the user to choose between (``NLQ_AMBIGUITY=ask``)."""

    #: What the caller passes back with the chosen value: ``table.column:value``.
    key: str
    column: str
    #: The value as the question named it.
    asked: str
    #: The stored values it is part of.
    options: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "column": self.column,
            "asked": self.asked,
            "options": list(self.options),
        }


def choice_key(table: str, column: str, literal: str) -> str:
    """The key a :class:`Choice` for *literal* in *table*.*column* is known by."""
    return f"{table}.{column}:{literal}"


_choices: contextvars.ContextVar[list[Choice] | None] = contextvars.ContextVar(
    "nlqueries_match_choices", default=None
)
_chosen: contextvars.ContextVar[Mapping[str, str] | None] = contextvars.ContextVar(
    "nlqueries_chosen_values", default=None
)


@contextmanager
def collect_choices() -> Iterator[list[Choice]]:
    """Collect, into the list it yields, the choices left for the user within."""
    choices: list[Choice] = []
    token = _choices.set(choices)
    try:
        yield choices
    finally:
        _choices.reset(token)


def add_choice(choice: Choice) -> None:
    """Add *choice* to the open collection, once."""
    choices = _choices.get()
    if choices is not None and choice not in choices:
        choices.append(choice)


def choices_for(choices: list[Choice], sql: str, dialect: str) -> list[Choice]:
    """The choices of *choices* whose value *sql* still compares with as the
    question named it: one a later step settled is no longer the user's to make."""
    if not choices:
        return []
    try:
        present = {
            node.this
            for node in sqlglot.parse_one(sql, read=dialect).find_all(exp.Literal)
            if node.is_string
        }
    except Exception:  # noqa: BLE001 - unparsable: nothing is shown to be there
        return []
    return [c for c in choices if c.asked in present]


@contextmanager
def use_chosen_values(chosen: Mapping[str, str]) -> Iterator[None]:
    """Answer the enclosed question with the values the user chose, by
    :attr:`Choice.key`."""
    token = _chosen.set(dict(chosen))
    try:
        yield
    finally:
        _chosen.reset(token)


def chosen_value(key: str) -> str | None:
    """The value the user chose for the choice known by *key*, if any."""
    chosen = _chosen.get()
    return None if chosen is None else chosen.get(key)


def any_chosen() -> bool:
    """Whether this question is being answered with values the user chose."""
    return bool(_chosen.get())
