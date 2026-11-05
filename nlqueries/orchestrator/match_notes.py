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
from collections.abc import Iterator
from contextlib import contextmanager

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
