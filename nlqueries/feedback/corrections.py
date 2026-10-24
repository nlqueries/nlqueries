# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.feedback.corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Which thumbs-down corrections become examples in the knowledge base.

``export-kb`` puts each accepted correction at the front of the agent's example
queries, and the prompt shows the first few of those to the model for every
later question on that agent. So a correction is accepted only on the terms a
promoted thumbs-up pair is:

- it comes from a source that identifies who gave it, unless the operator opts
  in to the rest (``--include-anonymous``, as for ``promote-feedback``);
- its SQL passes the SQL policy in the agent's dialect and reads only tables
  the knowledge base has;
- its question is one line of bounded length, and its SQL keeps no comments.
  The question is shown as a comment above the SQL, so a line break in it would
  end the comment and put the rest of the text in the prompt as an instruction;
  a comment inside the SQL is shown verbatim and could do the same.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from nlqueries.feedback.models import QueryFeedback
    from nlqueries.processing.parameterizer import QueryCapsule

#: The longest question kept on a correction, in characters.
MAX_INTENT_CHARS = 200


@dataclass
class CorrectionSelection:
    """The corrections accepted, and how many were left out and why."""

    capsules: list[QueryCapsule] = field(default_factory=list)
    unattributed: int = 0
    refused: int = 0
    #: Why each refused correction was refused, as counts per reason.
    reasons: Counter[str] = field(default_factory=Counter)


def one_line(text: str, limit: int = MAX_INTENT_CHARS) -> str:
    """*text* with every run of whitespace, line breaks included, made one space,
    and cut to *limit* characters.

    Stripped after the cut as well as before, so applying it twice changes
    nothing: a cut that landed on a space left a trailing one, which a second
    application removed, and pruning, which checks a stored question is in
    this form, deleted the pair promotion had just written, on every run."""
    return " ".join(str(text).split())[:limit].rstrip()


def select_corrections(
    records: list[QueryFeedback],
    schema_tables: set[str],
    dialect: str,
    *,
    include_unattributed: bool = False,
) -> CorrectionSelection:
    """The thumbs-down corrections in *records* fit to become examples.

    *schema_tables* are the tables of the knowledge base being written, as
    `promoter.known_tables` gives them; *dialect* is the agent's SQL dialect.
    """
    from nlqueries.feedback.models import ATTRIBUTED_SOURCES  # noqa: PLC0415
    from nlqueries.feedback.promoter import _why_not_kept, without_comments  # noqa: PLC0415
    from nlqueries.processing.parameterizer import (  # noqa: PLC0415
        QueryCapsule,
        _extract_columns,
        _extract_tables,
        _parameterize_sql,
    )
    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    selection = CorrectionSelection()
    for fb in records:
        if fb.rating != "down" or not fb.corrected_sql or not fb.corrected_sql.strip():
            continue
        if not include_unattributed and fb.source not in ATTRIBUTED_SOURCES:
            selection.unattributed += 1
            continue
        sql = fb.corrected_sql.strip()
        # The same check, in the same order, as a verified pair: both reach the
        # prompt as examples, and one sequence cannot drift from itself.
        reason = _why_not_kept(sql, schema_tables, dialect)
        if reason is not None:
            selection.refused += 1
            selection.reasons[reason] += 1
            continue
        question = one_line(fb.question)
        # Literals become typed placeholders, as in every other capsule: the
        # example shows the shape of the query, not values someone typed in.
        # Comments go first: parameterising keeps them, line breaks and all.
        template, placeholders = _parameterize_sql(
            without_comments(sql, dialect), {}, dialect=_sqlglot_dialect(dialect)
        )
        # ("", []) is the parameteriser's "could not": a capsule from it would
        # carry the submitter's question with no SQL beneath it.
        if not template:
            selection.refused += 1
            selection.reasons["it could not be made into a template"] += 1
            continue
        selection.capsules.append(
            QueryCapsule(
                template_sql=template,
                placeholders=placeholders,
                tables=_extract_tables(sql, _sqlglot_dialect(dialect)),
                columns=_extract_columns(sql, _sqlglot_dialect(dialect)),
                frequency=999,
                auto_description=f"User correction for: {question}",
                intent=f"Correction: {question}",
            )
        )
    return selection
