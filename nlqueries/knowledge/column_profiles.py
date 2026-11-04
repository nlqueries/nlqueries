# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.knowledge.column_profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What a whole column holds, beyond the few values the knowledge base samples.

A sample shows the first few rows. It does not show that nearly every value of
a join key starts with a space, so an equality join matches nothing; that a
year column holds ``'c. 2009'`` beside ``'2009'``; that a date table ends years
before the data does; or that a column is empty in all but a handful of rows. Each of those
turns a correct-looking statement into a wrong answer, and each is visible in
one aggregate over the column. ``export-kb`` runs that aggregate per table
(:func:`collect_column_profiles`), stores the result on each column as
``profile``, and the prompt shows it as a short note (:func:`profile_notes`).

Only counts, lengths and a date column's first and last date are stored: never
a text value, and nothing for a column whose name marks it as personal data.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from nlqueries.connectors.base import SchemaSpec
from nlqueries.knowledge.kb_generator import (
    _VALUES_TIMEOUT_S,
    VALUES_MAX_ROWS,
    _is_pii_column,
    _is_text_type,
    too_large_for_values,
)

logger = logging.getLogger(__name__)

#: Columns per profiling statement. Each profiles a text column with four
#: aggregates, and engines cap a select list (SQL Server at 4096 columns); a
#: wider table is profiled in more than one statement, each reading the table.
COLUMNS_PER_QUERY = 250

_DATE_TYPE = re.compile(r"DATE|TIME|YEAR", re.IGNORECASE)

#: Types that hold several values, or a structure: TRIM, LENGTH and MIN reject
#: them, and one such column fails the statement for its whole table. A
#: BigQuery REPEATED field is reported by its element type and cannot be seen
#: here; the counts-only retry in `collect_column_profiles` covers it.
_COMPOSITE_TYPE = re.compile(r"\[\]|ARRAY|LIST|REPEATED|STRUCT|RECORD|MAP\b", re.IGNORECASE)

#: SQL Server's legacy large types, which TRIM, LEN and <> reject.
_TSQL_LEGACY_TEXT = re.compile(r"^\s*N?TEXT\s*$", re.IGNORECASE)

#: A share of NULLs below this is not worth a note.
_NOTED_NULL_SHARE = 0.1


def profile_kind(col_type: str, dialect: str | None = None) -> str:
    """``"date"``, ``"text"`` or ``"plain"``: what to profile a column of
    *col_type* for. A date first, since an untyped column reads as text. A
    composite type, and SQL Server's ``text`` and ``ntext``, only have their
    values counted."""
    if col_type and _COMPOSITE_TYPE.search(col_type):
        return "plain"
    if (dialect or "").lower() in ("tsql", "mssql") and _TSQL_LEGACY_TEXT.match(col_type or ""):
        return "plain"
    if col_type and _DATE_TYPE.search(col_type):
        return "date"
    if _is_text_type(col_type):
        return "text"
    return "plain"


def collect_column_profiles(
    connector: Any,
    schema: SchemaSpec,
    dialect: str | None = None,
    max_rows: int = VALUES_MAX_ROWS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{table: {column: profile}}`` for every table not over *max_rows* rows.

    A profile holds what is worth saying and nothing else:

    - ``nulls``: the share of rows with no value, when there are any;
    - ``padded``: how many text values have spaces before or after them;
    - ``length``: a text column's shortest and longest value;
    - ``range``: a date column's earliest and latest value.

    Keys are profiled -- a padded join key is the case this exists for -- and
    columns whose names mark them as personal data are not. Read-only; a
    statement that fails or times out leaves that table without profiles. A
    table over *max_rows* is skipped, as for column values
    (:func:`too_large_for_values`): the statement reads every row, and on a
    warehouse billed by bytes read that is charged in full.
    """

    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    for table in schema.tables:
        if too_large_for_values(table, max_rows):
            continue
        columns = [
            (col.name, profile_kind(col.type, dialect))
            for col in table.columns
            if not _is_pii_column(col.name)
        ]
        for start in range(0, len(columns), COLUMNS_PER_QUERY):
            batch = columns[start : start + COLUMNS_PER_QUERY]
            row = _profile_row(connector, table.name, table.schema, batch, dialect)
            if row is None and any(kind != "plain" for _, kind in batch):
                # One column the text or date aggregates do not take fails the
                # statement for the whole batch. Counting each column's values
                # works on any type, so the batch keeps its share of NULLs.
                logger.info("Profiling %s failed; counting its columns' values only.", table.name)
                batch = [(name, "plain") for name, _ in batch]
                row = _profile_row(connector, table.name, table.schema, batch, dialect)
            if row is None:
                logger.info("Profiling %s failed; it has no profiles.", table.name)
                break
            found = _read_profiles(batch, row)
            if found:
                profiles.setdefault(table.name, {}).update(found)
    return profiles


def _profile_row(
    connector: Any,
    table: str,
    schema: str | None,
    batch: list[tuple[str, str]],
    dialect: str | None,
) -> list[Any] | None:
    """The one row the profile statement for *batch* returns, or ``None`` when
    it failed."""
    from nlqueries.connectors.base import column_profile_sql  # noqa: PLC0415

    try:
        result = connector.execute_query(
            column_profile_sql(table, schema, batch, dialect), _VALUES_TIMEOUT_S
        )
    except Exception:  # noqa: BLE001
        logger.debug("Profiling %s raised.", table, exc_info=True)
        return None
    if result.error or not result.rows:
        logger.debug("Profiling %s failed: %s", table, result.error)
        return None
    return list(result.rows[0])


def _read_profiles(batch: list[tuple[str, str]], row: list[Any]) -> dict[str, dict[str, Any]]:
    """The profiles in *row*, laid out as `column_profile_sql` builds it."""
    values = iter(row)
    total = _number(next(values, None))
    found: dict[str, dict[str, Any]] = {}
    for column, kind in batch:
        profile: dict[str, Any] = {}
        present = _number(next(values, None))
        if total and present is not None and present < total:
            profile["nulls"] = round(1 - present / total, 3)
        if kind == "text":
            padded, shortest, longest = (_number(next(values, None)) for _ in range(3))
            if padded:
                profile["padded"] = int(padded)
            if shortest is not None and longest is not None:
                profile["length"] = [int(shortest), int(longest)]
        elif kind == "date":
            earliest, latest = next(values, None), next(values, None)
            if earliest is not None and latest is not None:
                profile["range"] = [str(earliest), str(latest)]
        if profile:
            found[column] = profile
    return found


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def profile_notes(column: dict[str, Any]) -> list[str]:
    """What the prompt says about *column*'s ``profile``, if anything.

    A length range is said only where it contradicts the samples: every
    sample one length and the column not. That is a fixed format with values
    the samples do not show (``'2009'`` beside ``'c. 2009'``); a column of names
    varies in length by nature, and saying so would be noise on every one.
    """
    profile = column.get("profile")
    if not isinstance(profile, dict):
        return []
    notes: list[str] = []
    nulls = profile.get("nulls")
    if isinstance(nulls, int | float) and nulls >= _NOTED_NULL_SHARE:
        notes.append(f"{round(nulls * 100)}% NULL")
    padded = profile.get("padded")
    if isinstance(padded, int) and padded > 0:
        name = column.get("name", "column")
        notes.append(f"{padded} values have spaces around them: compare TRIM({name})")
    length = profile.get("length")
    samples = [str(s) for s in column.get("samples") or []]
    if (
        isinstance(length, list)
        and len(length) == 2
        and samples
        and len({len(s) for s in samples}) == 1
        and (length[0] < len(samples[0]) or length[1] > len(samples[0]))
    ):
        notes.append(f"lengths {length[0]}-{length[1]}, unlike the samples")
    span = profile.get("range")
    if isinstance(span, list) and len(span) == 2:
        notes.append(f"range {span[0]} to {span[1]}")
    return notes
