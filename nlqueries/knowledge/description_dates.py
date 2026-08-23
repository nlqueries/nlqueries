"""
nlqueries.knowledge.description_dates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When each table and column description last actually changed.

A knowledge base records what a description says and where it came from
(``description_source``: manual, llm, dbt, schema) but never when it was
written. So a description generated months ago, against a schema that has since
moved on, is indistinguishable from one someone wrote this morning — and the
first thing anyone asks of a suspicious description is how old it is.

There is nothing to derive it from. The file's own mtime gives one date for the
whole knowledge base and changes on every save. The usage events record that a
description run happened for an agent, not which table it touched. It has to be
stored.

**Only a real change moves the date.** Re-saving an untouched description keeps
the stamp it already had; otherwise the field degrades into "when did anyone
last press Save", which is a fact about the editor rather than the description.

**A description with no stamp stays without one.** Every knowledge base that
exists predates this, and back-filling today's date would claim every one of
them was written the day this shipped — the exact wrong answer for the columns
whose age is worth knowing.

Lives here, in core, because three writers need identical behaviour: this
package's generator (which rebuilds every table dict from scratch on each run),
the enterprise KB editor (which saves the whole file), and the enterprise dbt
sync (which merges docs into a loaded one). Written three times it would drift
three ways.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

#: The field, beside ``description_source``, on both tables and columns.
FIELD = "description_updated_at"


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat()


def _descriptions_by_name(kb: dict[str, Any] | None) -> dict[str, tuple[str, str]]:
    """Map ``"table"`` / ``"table.column"`` to its (description, stamp).

    Flat rather than nested: the callers only ever ask "what did this one say
    before", and a table and its columns are the same question twice.
    """
    found: dict[str, tuple[str, str]] = {}
    if not isinstance(kb, dict):
        return found
    schema = kb.get("schema")
    tables = schema.get("tables") if isinstance(schema, dict) else None
    if not isinstance(tables, list):
        return found
    for table in tables:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name") or "")
        if not name:
            continue
        found[name] = (str(table.get("description") or ""), str(table.get(FIELD) or ""))
        columns = table.get("columns")
        if not isinstance(columns, list):
            continue
        for column in columns:
            if not isinstance(column, dict):
                continue
            col_name = str(column.get("name") or "")
            if not col_name:
                continue
            found[f"{name}.{col_name}"] = (
                str(column.get("description") or ""),
                str(column.get(FIELD) or ""),
            )
    return found


def stamp_descriptions(
    kb: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> int:
    """Date the descriptions in *kb* that differ from *previous*, in place.

    Returns how many entries this changed — counting a date corrected or removed,
    not only one newly written. Zero means the caller can store its input
    untouched; anything else means the input disagreed with the truth and has to
    be re-serialised.

    A description whose text is unchanged keeps whatever stamp it had, including
    none. One that is new or edited gets *now*. An empty description is left
    alone entirely — there is nothing to have been written, and dating it would
    put "updated today" beside a blank.
    """
    # A caller passing its own clock is how this stays testable without freezing
    # time globally.
    moment = _iso(now or datetime.now(UTC))
    before = _descriptions_by_name(previous)
    stamped = 0

    schema = kb.get("schema")
    tables = schema.get("tables") if isinstance(schema, dict) else None
    if not isinstance(tables, list):
        return 0

    for table in tables:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name") or "")
        if not name:
            continue
        stamped += _stamp_one(table, before.get(name), moment)
        columns = table.get("columns")
        if not isinstance(columns, list):
            continue
        for column in columns:
            if not isinstance(column, dict):
                continue
            col_name = str(column.get("name") or "")
            if not col_name:
                continue
            stamped += _stamp_one(column, before.get(f"{name}.{col_name}"), moment)
    return stamped


def _stamp_one(entry: dict[str, Any], was: tuple[str, str] | None, moment: str) -> int:
    """Set one table or column's date. Returns 1 if this changed the entry.

    The date is derived here in every case, never taken from *entry*. What
    arrives in that field is whatever the caller was handed — for the KB editor,
    a value that has been round-tripped through a browser — and a date claiming
    a review that never happened is worse than no date at all.
    """
    incoming = entry.get(FIELD)
    description = str(entry.get("description") or "")

    if not description:
        # Nothing written, nothing to date. This also clears a stamp stranded by
        # a description someone deleted.
        wanted: str | None = None
    else:
        previous_description, previous_stamp = was or ("", "")
        # Unchanged text keeps the date it already had, which may be none.
        wanted = (previous_stamp or None) if description == previous_description else moment

    if wanted is None:
        entry.pop(FIELD, None)
    else:
        entry[FIELD] = wanted
    return 1 if incoming != wanted else 0
