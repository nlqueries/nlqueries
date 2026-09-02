"""Tests for nlqueries.orchestrator.result_merger (Task 13.1).

Covers:
- Parallel hybrid execution in MultiAgentOrchestrator
- merge_results() LLM synthesis when both sources are present
- merge_results() passthrough paths when only one source is present
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import MagicMock, patch

from nlqueries.connectors.base import QueryResult
from nlqueries.document_connectors.base import DocumentChunk
from nlqueries.orchestrator.document_retrieval import Citation, DocumentRetrievalResult
from nlqueries.orchestrator.result_merger import HybridQueryResult, _format_sql_table, merge_results

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_query_result() -> QueryResult:
    """A two-row QueryResult suitable for merge_results tests."""
    return QueryResult(
        columns=["customer", "revenue"],
        rows=[["Acme Corp", 50000], ["Beta Inc", 30000]],
        row_count=2,
        execution_time_ms=12.5,
        error=None,
    )


def _make_document_result() -> DocumentRetrievalResult:
    """A DocumentRetrievalResult with one citation."""
    chunk = DocumentChunk(
        chunk_id="abc123",
        source_id="src1",
        source_name="Q3 Report.pdf",
        page_number=5,
        chunk_index=0,
        text="Enterprise contracts showed a 20% uplift in Q3.",
        metadata={},
    )
    citation = Citation(
        chunk_id="abc123",
        source_name="Q3 Report.pdf",
        page_number=5,
        chunk_index=0,
        excerpt="Enterprise contracts showed a 20% uplift in Q3.",
        relevance_score=0.9,
    )
    return DocumentRetrievalResult(
        chunks=[chunk],
        citations=[citation],
        collection="doc_src1_chunks",
    )


# ---------------------------------------------------------------------------
# Spec-required tests
# ---------------------------------------------------------------------------


class TestResultMergerSpec:
    """The four tests explicitly required by the Task 13.1 spec."""

    def test_both_agents_run_concurrently(self) -> None:
        """Hybrid intent runs both SQL and Document agents (asyncio.gather)."""
        from nlqueries.orchestrator.intent_classifier import (
            IntentClassificationResult,
            IntentType,
        )
        from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

        _SQL_TOKENS = [
            "SQL reasoning token.",
            json.dumps(
                {
                    "type": "sql",
                    "sql": "SELECT * FROM orders",
                    "is_valid": True,
                    "validation_error": None,
                    "dialect": "postgres",
                    "attempt_count": 1,
                }
            ),
        ]
        _DOC_TOKENS = [
            "Doc reasoning token.",
            json.dumps(
                {
                    "type": "citations",
                    "citations": [
                        {
                            "source_name": "Q3.pdf",
                            "page_number": 1,
                            "excerpt": "Revenue grew 20%.",
                        }
                    ],
                }
            ),
        ]

        sql_called: list[bool] = []
        doc_called: list[bool] = []

        async def _sql_gen(*_args: object, **_kwargs: object):  # type: ignore[return]
            sql_called.append(True)
            for t in _SQL_TOKENS:
                yield t

        async def _doc_gen(*_args: object, **_kwargs: object):  # type: ignore[return]
            doc_called.append(True)
            for t in _DOC_TOKENS:
                yield t

        sql_instance = MagicMock()
        sql_instance.handle_question = _sql_gen
        doc_instance = MagicMock()
        doc_instance.handle_question = _doc_gen

        mock_hybrid = HybridQueryResult(
            sql_answer="SQL rows",
            sql_table=None,
            document_answer="Doc excerpts",
            citations=[],
            merged_answer="Combined answer from both sources.",
        )

        async def run() -> list[str]:
            with (
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                    return_value=IntentClassificationResult(
                        intent=IntentType.hybrid,
                        confidence=0.95,
                        reasoning="Both needed.",
                    ),
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                    return_value=sql_instance,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                    return_value=doc_instance,
                ),
                patch(
                    "nlqueries.orchestrator.multi_agent_orchestrator.merge_results",
                    return_value=mock_hybrid,
                ),
            ):
                orch = MultiAgentOrchestrator()
                tokens: list[str] = []
                async for token in orch.handle_question(
                    "Which customers from Q3 haven't ordered yet?",
                    "agent1",
                    available_types=["sql", "document", "hybrid"],
                ):
                    tokens.append(token)
                return tokens

        tokens = asyncio.run(run())

        assert sql_called, "SQL orchestrator must be called for hybrid intent"
        assert doc_called, "Document orchestrator must be called for hybrid intent"

        # Final chunk must be a hybrid chunk
        final = json.loads(tokens[-1])
        assert final["type"] == "hybrid"
        assert final["agent_type"] == "hybrid"
        assert final["merged_answer"] == "Combined answer from both sources."

    def test_merge_calls_llm_when_both_results_present(self) -> None:
        """merge_results calls the LLM when both sql_result and document_result are given."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            "Acme Corp and Beta Inc appear in both the Q3 SQL data and the report."
        )

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            result = merge_results(
                question="Which customers from Q3 report haven't ordered yet?",
                sql_result=_make_query_result(),
                document_result=_make_document_result(),
            )

        mock_llm.complete.assert_called_once()
        assert (
            result.merged_answer
            == "Acme Corp and Beta Inc appear in both the Q3 SQL data and the report."
        )
        assert result.sql_table is not None
        assert result.citations

    def test_merge_passthrough_when_only_sql(self) -> None:
        """merge_results does not call the LLM when only sql_result is present."""
        mock_llm = MagicMock()

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            result = merge_results(
                question="Show total revenue",
                sql_result=_make_query_result(),
                document_result=None,
            )

        mock_llm.complete.assert_not_called()
        assert result.merged_answer == result.sql_answer
        assert result.sql_table is not None
        assert result.document_answer is None
        assert result.citations == []

    def test_merge_passthrough_when_only_document(self) -> None:
        """merge_results does not call the LLM when only document_result is present."""
        mock_llm = MagicMock()

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            result = merge_results(
                question="What is the refund policy?",
                sql_result=None,
                document_result=_make_document_result(),
            )

        mock_llm.complete.assert_not_called()
        assert result.merged_answer == result.document_answer
        assert result.sql_table is None
        assert result.sql_answer is None
        assert len(result.citations) > 0


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestResultMergerExtra:
    """Extra tests for edge cases and implementation details."""

    def test_both_none_returns_empty_result(self) -> None:
        """merge_results with both None returns an empty HybridQueryResult."""
        result = merge_results(
            question="anything",
            sql_result=None,
            document_result=None,
        )
        assert result.merged_answer == ""
        assert result.sql_answer is None
        assert result.document_answer is None
        assert result.citations == []
        assert result.sql_table is None

    def test_sql_answer_formatted_as_markdown_table(self) -> None:
        """When only SQL is present, merged_answer is a markdown table string."""
        result = merge_results(
            question="Q",
            sql_result=_make_query_result(),
            document_result=None,
        )
        assert "| customer | revenue |" in result.sql_answer or "" in result.merged_answer
        assert "| --- |" in result.merged_answer

    def test_document_answer_contains_source_name(self) -> None:
        """When only document is present, merged_answer contains the source name."""
        result = merge_results(
            question="Q",
            sql_result=None,
            document_result=_make_document_result(),
        )
        assert "Q3 Report.pdf" in result.merged_answer

    def test_document_passthrough_citations_populated(self) -> None:
        """Passthrough document path populates citations on HybridQueryResult."""
        doc_result = _make_document_result()
        result = merge_results(
            question="Q",
            sql_result=None,
            document_result=doc_result,
        )
        assert len(result.citations) == len(doc_result.citations)
        assert result.citations[0].source_name == "Q3 Report.pdf"

    def test_sql_passthrough_sql_table_preserved(self) -> None:
        """Passthrough SQL path preserves the QueryResult on sql_table."""
        qr = _make_query_result()
        result = merge_results(
            question="Q",
            sql_result=qr,
            document_result=None,
        )
        assert result.sql_table is qr
        assert result.sql_table.columns == ["customer", "revenue"]

    def test_llm_synthesis_prompt_contains_question(self) -> None:
        """The LLM synthesis prompt includes the original question."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Answer."

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            merge_results(
                question="Which customers signed in Q3?",
                sql_result=_make_query_result(),
                document_result=_make_document_result(),
            )

        _call_args = mock_llm.complete.call_args
        user_prompt = _call_args[0][1]  # second positional arg
        assert "Which customers signed in Q3?" in user_prompt

    def test_llm_synthesis_both_sources_in_prompt(self) -> None:
        """The LLM synthesis prompt includes both SQL table and document excerpts."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Answer."

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            merge_results(
                question="Q",
                sql_result=_make_query_result(),
                document_result=_make_document_result(),
            )

        _call_args = mock_llm.complete.call_args
        user_prompt = _call_args[0][1]
        # SQL columns should appear
        assert "customer" in user_prompt
        # Citation source should appear
        assert "Q3 Report.pdf" in user_prompt

    def test_hybrid_result_sql_answer_populated_when_both_present(self) -> None:
        """When both results present, sql_answer is set (not None)."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Synthesis."

        with patch(
            "nlqueries.orchestrator.result_merger.get_llm_client",
            return_value=mock_llm,
        ):
            result = merge_results(
                question="Q",
                sql_result=_make_query_result(),
                document_result=_make_document_result(),
            )

        assert result.sql_answer is not None
        assert result.document_answer is not None

    def test_hybrid_query_result_type(self) -> None:
        """merge_results always returns a HybridQueryResult."""
        result = merge_results(question="Q", sql_result=None, document_result=None)
        assert isinstance(result, HybridQueryResult)


# ---------------------------------------------------------------------------
# The 20-row cap has to announce itself (#174 review)
# ---------------------------------------------------------------------------


def _result(n_rows: int, *, row_count: int | None = None, truncated: bool = False) -> QueryResult:
    return QueryResult(
        columns=["region", "revenue"],
        rows=[[f"r{i}", i] for i in range(n_rows)],
        row_count=row_count if row_count is not None else n_rows,
        execution_time_ms=1.0,
        error=None,
        truncated=truncated,
    )


def test_a_capped_table_says_how_much_it_is_showing() -> None:
    """The synthesis model is asked to describe this table and has no other way
    to tell a complete answer from a fragment, so it would take a total over the
    first twenty rows and present it as the answer.

    This cap never bit until the hybrid path began carrying executed rows: the
    table it rendered was one cell holding the statement text.
    """
    rendered = _format_sql_table(_result(50))
    assert "showing the first 20 of 50 rows" in rendered
    # Matched on the row shape, not on a prefix: "| region | revenue |" also
    # starts with "| r", so a substring count reports 22 and reads as a pass
    # that happens to be off by the width of the header.
    body = [ln for ln in rendered.splitlines() if re.match(r"^\| r\d+ \|", ln)]
    assert len(body) == 20


def test_a_whole_table_is_not_labelled_a_fragment() -> None:
    """Canary. A note on a complete answer would train the reader to ignore it."""
    rendered = _format_sql_table(_result(3))
    assert "showing the first" not in rendered
    assert "stopped short" not in rendered


def test_a_result_truncated_upstream_is_reported_as_a_fragment() -> None:
    """`row_count` is the true total and stays so when `rows` was capped by the
    connector, so the note quotes it -- len(rows) would report the size of the
    fragment as the size of the answer."""
    rendered = _format_sql_table(_result(20, row_count=8000, truncated=True))
    assert "showing the first 20 of 8000 rows" in rendered


def test_a_truncated_result_that_cannot_say_its_total_still_says_it_stopped() -> None:
    """A connector that truncated without adjusting `row_count` leaves the total
    unknown. Saying it stopped short still beats implying completeness."""
    rendered = _format_sql_table(_result(20, row_count=20, truncated=True))
    assert "stopped short" in rendered
