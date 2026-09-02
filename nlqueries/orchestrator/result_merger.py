"""
nlqueries.orchestrator.result_merger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Merges results from the SQL agent and Document agent into a single unified
hybrid response.

When both results are present the LLM is called once with a synthesis prompt
that includes the SQL data (formatted as a markdown table, ≤ 20 rows) and the
top-3 document citations.  When only one result is present the result is passed
through without an additional LLM call.

Public API
----------
``merge_results(question, sql_result, document_result)``
    Returns a :class:`HybridQueryResult` with a ``merged_answer`` string and
    the raw source data for UI rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlqueries.connectors.base import QueryResult
from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult


@dataclass
class HybridQueryResult:
    """Merged result combining SQL and document agent outputs."""

    sql_answer: str | None
    sql_table: QueryResult | None
    document_answer: str | None
    citations: list[Citation]
    merged_answer: str  # LLM-synthesised unified response (or passthrough)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _cell(value: object) -> str:
    """One table cell, safe to sit between pipes.

    Real data reaches this renderer now. A value containing a pipe breaks the
    column alignment of the table handed to the synthesis model; one containing
    a newline ends the row early and can produce a line shaped like the
    renderer's own trailing note, which the model then cannot tell apart from
    the data. Before the hybrid path carried executed rows the only cell was the
    generated statement, so none of this was reachable.

    The backslash is escaped first, or a value ending in one would turn the
    delimiter that follows it into an escaped pipe and merge two cells.
    """
    text = str(value).replace("\\", "\\\\").replace("|", "\\|")
    # Collapses newlines and tabs along with runs of spaces. A prompt table does
    # not need the original whitespace, and every alternative leaves a way to
    # end the row early.
    return " ".join(text.split())


def _format_sql_table(sql_result: QueryResult, max_rows: int = 20) -> str:
    """Render a :class:`QueryResult` as a GitHub-style markdown table.

    A note is appended whenever this shows fewer rows than the result holds.
    The synthesis model is asked to describe this table and has no other way to
    tell a complete answer from a fragment, so without it a total or a maximum
    taken over the first twenty rows is presented as the answer.
    :class:`QueryResult` puts the principle plainly: silently returning the
    first N rows of a larger answer is a wrong answer, not a partial one.

    The cap never bit before the hybrid path carried executed rows -- the table
    it rendered was one cell holding the statement text.
    """
    if not sql_result.columns:
        return "(no data)"
    header = "| " + " | ".join(_cell(c) for c in sql_result.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(sql_result.columns)) + " |"
    shown = sql_result.rows[:max_rows]
    body_lines = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in shown]
    lines = [header, separator, *body_lines]

    retrieved = len(sql_result.rows)
    total = max(sql_result.row_count, retrieved)

    # Whether `total` is the number that matched, or only the number reached.
    #
    # Every connector sets `row_count=len(rows)`, so after a `row_budget` or
    # `byte_budget` stop the count describes what was retrieved and is itself a
    # fragment -- quoting it as "of N rows" asserts a total that is not one.
    # `sql_table_chunk` is the opposite: it caps `rows` while reporting the full
    # `row_count`. So the two being unequal is what says the total is real, and
    # that holds without this renderer having to interpret
    # `truncation_reason` -- including for a reason added later.
    total_is_exact = sql_result.row_count > retrieved or not sql_result.truncated

    if len(shown) < total:
        if total_is_exact:
            lines.append(f"\n(showing the first {len(shown)} of {total} rows)")
        else:
            lines.append(
                f"\n(showing the first {len(shown)} of at least {total} rows; "
                "this result stopped short of the full answer)"
            )
    elif sql_result.truncated:
        # Everything retrieved is shown, but the query stopped early, so the
        # true total is unknown and there is no count worth quoting.
        lines.append("\n(this result stopped short of the full answer)")
    return "\n".join(lines)


def _format_citations_text(citations: list[Citation], top_n: int = 3) -> str:
    """Render the top-N citations as labelled text excerpts."""
    parts: list[str] = []
    for i, c in enumerate(citations[:top_n], 1):
        page = f", page {c.page_number}" if c.page_number is not None else ""
        parts.append(f"[{i}] {c.source_name}{page}: {c.excerpt}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def merge_results(
    question: str,
    sql_result: QueryResult | None,
    document_result: DocumentRetrievalResult | None,
) -> HybridQueryResult:
    """Merge SQL and document results into a unified hybrid response.

    When both *sql_result* and *document_result* are provided the LLM is
    invoked once to synthesise a single answer that references both sources.
    SQL data is formatted as a markdown table (up to 20 rows, and it says so
    when there are more); the top 3
    document citations are included as labelled excerpts.

    When only one result is provided the function passes it through without
    an additional LLM call — ``merged_answer`` equals the single-source
    narrative.

    Args:
        question:        Original user question.
        sql_result:      SQL execution result, or ``None`` if the SQL agent
                         was not run or produced no result.
        document_result: Document retrieval result, or ``None`` if the
                         document agent was not run.

    Returns:
        :class:`HybridQueryResult` with all fields populated.
    """
    # ------------------------------------------------------------------ #
    # Passthrough: only SQL result present                                #
    # ------------------------------------------------------------------ #
    if sql_result is not None and document_result is None:
        sql_answer = _format_sql_table(sql_result)
        return HybridQueryResult(
            sql_answer=sql_answer,
            sql_table=sql_result,
            document_answer=None,
            citations=[],
            merged_answer=sql_answer,
        )

    # ------------------------------------------------------------------ #
    # Passthrough: only document result present                           #
    # ------------------------------------------------------------------ #
    if document_result is not None and sql_result is None:
        document_answer = _format_citations_text(document_result.citations)
        return HybridQueryResult(
            sql_answer=None,
            sql_table=None,
            document_answer=document_answer,
            citations=document_result.citations,
            merged_answer=document_answer,
        )

    # ------------------------------------------------------------------ #
    # Both present: LLM synthesis                                        #
    # ------------------------------------------------------------------ #
    if sql_result is not None and document_result is not None:
        sql_table_text = _format_sql_table(sql_result)
        doc_text = _format_citations_text(document_result.citations)

        system = (
            "You are a data analyst synthesizing information from two sources: "
            "structured SQL query results and document excerpts. "
            "Produce a concise, unified answer that references both sources clearly."
        )
        user = (
            f"Question: {question}\n\n"
            f"SQL Data:\n{sql_table_text}\n\n"
            f"Document Context:\n{doc_text}\n\n"
            "Provide a unified answer that combines insights from both the SQL data "
            "and the document context."
        )

        llm = get_llm_client()
        merged = llm.complete(system, user)

        return HybridQueryResult(
            sql_answer=sql_table_text,
            sql_table=sql_result,
            document_answer=_format_citations_text(document_result.citations),
            citations=document_result.citations,
            merged_answer=merged,
        )

    # ------------------------------------------------------------------ #
    # Both None: empty result                                             #
    # ------------------------------------------------------------------ #
    return HybridQueryResult(
        sql_answer=None,
        sql_table=None,
        document_answer=None,
        citations=[],
        merged_answer="",
    )
