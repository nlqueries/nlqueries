"""
tests.test_description_dates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a table or column description last actually changed.

A knowledge base said what a description was and where it came from, never when
it was written — so one generated months ago against a schema that has since
moved on read exactly like one written this morning.

The behaviour worth pinning is what does *not* move the date: re-saving the same
text, and a knowledge base that never had one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from nlqueries.knowledge.description_dates import FIELD, stamp_descriptions

NOW = datetime(2026, 8, 23, 9, 0, tzinfo=UTC)
LAST_WEEK = "2026-08-16T09:00:00+00:00"


def kb(*tables: dict[str, Any]) -> dict[str, Any]:
    return {"schema": {"tables": list(tables)}}


def table(name: str, description: str, *, stamp: str | None = None, **cols: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "description": description,
        "columns": [{"name": c, "description": d} for c, d in cols.items()],
    }
    if stamp:
        entry[FIELD] = stamp
    return entry


def test_a_new_description_is_dated() -> None:
    current = kb(table("orders", "All orders."))

    assert stamp_descriptions(current, None, now=NOW) == 1
    assert current["schema"]["tables"][0][FIELD] == "2026-08-23T09:00:00+00:00"


def test_an_edited_description_is_redated() -> None:
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK))
    current = kb(table("orders", "Orders, one row per line item.", stamp=LAST_WEEK))

    stamp_descriptions(current, previous, now=NOW)

    assert current["schema"]["tables"][0][FIELD] == "2026-08-23T09:00:00+00:00"


def test_an_unchanged_description_keeps_its_date() -> None:
    """Otherwise the label reads "when did anyone last press Save".

    Which is a fact about the editor, not about the description, and it would
    make every column look freshly reviewed after any unrelated edit.
    """
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK))
    current = kb(table("orders", "Orders.", stamp=LAST_WEEK))

    # Nothing changed, so the caller can store what it was handed.
    assert stamp_descriptions(current, previous, now=NOW) == 0
    assert current["schema"]["tables"][0][FIELD] == LAST_WEEK


def test_a_date_the_caller_invented_is_replaced() -> None:
    """The date is derived here in every case, never read out of the input.

    What arrives in the field is whatever the caller was handed — for the KB
    editor, a value round-tripped through a browser. A date claiming a review
    that never happened is worse than no date at all, so an unchanged
    description gets the stored date back whatever it was sent with.
    """
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK))
    current = kb(table("orders", "Orders.", stamp="2099-12-31T00:00:00+00:00"))

    # Counted, because the caller's copy now disagrees and has to be rewritten.
    assert stamp_descriptions(current, previous, now=NOW) == 1
    assert current["schema"]["tables"][0][FIELD] == LAST_WEEK


def test_columns_are_dated_independently_of_their_table() -> None:
    """Editing one column must not re-date the eighty beside it."""
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK, id="The id.", total="Total."))
    previous["schema"]["tables"][0]["columns"][0][FIELD] = LAST_WEEK
    previous["schema"]["tables"][0]["columns"][1][FIELD] = LAST_WEEK
    current = kb(table("orders", "Orders.", stamp=LAST_WEEK, id="The id.", total="Total in cents."))
    current["schema"]["tables"][0]["columns"][0][FIELD] = LAST_WEEK
    current["schema"]["tables"][0]["columns"][1][FIELD] = LAST_WEEK

    # Only the edited column.
    assert stamp_descriptions(current, previous, now=NOW) == 1

    columns = current["schema"]["tables"][0]["columns"]
    assert columns[0][FIELD] == LAST_WEEK
    assert columns[1][FIELD] == "2026-08-23T09:00:00+00:00"
    assert current["schema"]["tables"][0][FIELD] == LAST_WEEK


def test_an_existing_knowledge_base_is_not_back_filled() -> None:
    """Every knowledge base that exists predates this field.

    Stamping them all today would claim each one was written the day this
    shipped — precisely wrong for the descriptions whose age matters.
    """
    previous = kb(table("orders", "Orders."))  # no stamp anywhere
    current = kb(table("orders", "Orders."))

    assert stamp_descriptions(current, previous, now=NOW) == 0
    assert FIELD not in current["schema"]["tables"][0]


def test_an_empty_description_is_not_dated() -> None:
    """ "Updated today" beside a blank says something that is not true."""
    current = kb(table("orders", "", id=""))

    assert stamp_descriptions(current, None, now=NOW) == 0
    assert FIELD not in current["schema"]["tables"][0]
    assert FIELD not in current["schema"]["tables"][0]["columns"][0]


def test_clearing_a_description_drops_its_date() -> None:
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK))
    current = kb(table("orders", "", stamp=LAST_WEEK))

    stamp_descriptions(current, previous, now=NOW)

    assert FIELD not in current["schema"]["tables"][0]


def test_a_table_added_since_the_last_build_is_dated() -> None:
    previous = kb(table("orders", "Orders.", stamp=LAST_WEEK))
    current = kb(table("orders", "Orders.", stamp=LAST_WEEK), table("refunds", "Refunds."))

    assert stamp_descriptions(current, previous, now=NOW) == 1
    assert current["schema"]["tables"][1][FIELD] == "2026-08-23T09:00:00+00:00"


def test_a_malformed_knowledge_base_is_left_alone() -> None:
    """Called on every save, so it must not be the thing that raises."""
    assert stamp_descriptions({}, None, now=NOW) == 0
    assert stamp_descriptions({"schema": "not a mapping"}, None, now=NOW) == 0
    assert stamp_descriptions({"schema": {"tables": ["nonsense", None]}}, None, now=NOW) == 0
    assert stamp_descriptions(kb(table("t", "d")), {"schema": None}, now=NOW) == 1
