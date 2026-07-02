"""Tests for nlqueries.orchestrator.candidates — Phase 6A self-consistency."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from nlqueries.orchestrator.candidates import (
    _HARD_TABLE_THRESHOLD,
    _is_hard,
    generate_candidates,
    select_best,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kb(table_names: list[str]) -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {"name": n, "columns": [{"name": "id", "type": "INTEGER"}]} for n in table_names
            ]
        }
    }


def _make_large_kb(n_tables: int = 25) -> dict[str, Any]:
    return _make_kb([f"table_{i}" for i in range(n_tables)])


# ---------------------------------------------------------------------------
# _is_hard()
# ---------------------------------------------------------------------------


class TestIsHard:
    def test_simple_count_is_not_hard(self) -> None:
        kb = _make_kb(["orders"])
        assert not _is_hard("How many orders?", kb)

    def test_join_and_group_by_is_hard(self) -> None:
        kb = _make_kb(["orders", "customers"])
        question = "Show orders with JOIN on customers GROUP BY region"
        assert _is_hard(question, kb)

    def test_single_signal_is_not_hard(self) -> None:
        kb = _make_kb(["orders"])
        question = "Group orders by status"
        assert not _is_hard(question, kb)

    def test_large_schema_is_hard(self) -> None:
        kb = _make_large_kb(_HARD_TABLE_THRESHOLD + 1)
        assert _is_hard("Simple count", kb)

    def test_exactly_threshold_tables_not_hard(self) -> None:
        kb = _make_large_kb(_HARD_TABLE_THRESHOLD)
        assert not _is_hard("Simple count", kb)

    def test_window_function_and_partition_is_hard(self) -> None:
        kb = _make_kb(["sales"])
        question = "Calculate rank using PARTITION BY region window function"
        assert _is_hard(question, kb)

    def test_subquery_and_join_is_hard(self) -> None:
        kb = _make_kb(["orders"])
        question = "Find customers where subquery matches JOIN on orders"
        assert _is_hard(question, kb)

    def test_join_and_having_is_hard(self) -> None:
        kb = _make_kb(["orders", "items"])
        assert _is_hard("JOIN orders HAVING total > 100", kb)

    def test_case_insensitive_signals(self) -> None:
        kb = _make_kb(["orders", "customers"])
        assert _is_hard("join orders group by status", kb)

    def test_empty_question_not_hard(self) -> None:
        assert not _is_hard("", _make_kb(["orders"]))


# ---------------------------------------------------------------------------
# generate_candidates()
# ---------------------------------------------------------------------------


class TestGenerateCandidates:
    def _make_llm(self, responses: list[str]) -> MagicMock:
        llm = MagicMock()
        idx = [0]

        async def fake_acomplete(system, user, max_tokens=512, *, temperature=None) -> str:
            i = idx[0]
            idx[0] += 1
            if i < len(responses):
                return responses[i]
            return ""

        llm.acomplete = fake_acomplete
        return llm

    def test_returns_list_of_sql(self) -> None:
        llm = self._make_llm(
            [
                "<sql>SELECT COUNT(*) FROM orders</sql>",
                "<sql>SELECT COUNT(*) FROM orders</sql>",
                "<sql>SELECT COUNT(*) FROM orders</sql>",
            ]
        )
        results = asyncio.run(generate_candidates(llm, "system", "user", n=3))
        assert len(results) == 3
        assert all("SELECT" in r for r in results)

    def test_n_equals_1_returns_one(self) -> None:
        llm = self._make_llm(["<sql>SELECT 1</sql>"])
        results = asyncio.run(generate_candidates(llm, "system", "user", n=1))
        assert len(results) == 1

    def test_empty_responses_excluded(self) -> None:
        llm = self._make_llm(["", "<sql>SELECT 1</sql>", ""])
        results = asyncio.run(generate_candidates(llm, "system", "user", n=3))
        assert len(results) == 1

    def test_exception_in_one_call_does_not_raise(self) -> None:
        llm = MagicMock()
        call_count = [0]

        async def raise_sometimes(system, user, max_tokens=512, *, temperature=None) -> str:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("LLM error")
            return "<sql>SELECT 1</sql>"

        llm.acomplete = raise_sometimes
        results = asyncio.run(generate_candidates(llm, "system", "user", n=3))
        assert len(results) == 2  # 2 successes, 1 exception silently dropped

    def test_calls_vary_temperature(self) -> None:
        """generate_candidates must pass varying temperatures to acomplete."""
        temperatures_used: list[float | None] = []
        llm = MagicMock()

        async def capture(system, user, max_tokens=512, *, temperature=None) -> str:
            temperatures_used.append(temperature)
            return "<sql>SELECT 1</sql>"

        llm.acomplete = capture
        asyncio.run(generate_candidates(llm, "system", "user", n=3))
        assert len(set(temperatures_used)) > 1


# ---------------------------------------------------------------------------
# select_best()
# ---------------------------------------------------------------------------


class TestSelectBest:
    def test_majority_vote_wins(self) -> None:
        kb = _make_kb(["orders"])
        candidates = [
            "SELECT COUNT(*) FROM orders",
            "SELECT COUNT(*) FROM orders",
            "SELECT * FROM orders",
        ]
        result = select_best(candidates, kb, "postgres")
        assert "COUNT" in result

    def test_single_candidate_returned(self) -> None:
        kb = _make_kb(["orders"])
        result = select_best(["SELECT * FROM orders"], kb, "postgres")
        assert "orders" in result

    def test_empty_candidates_returns_empty(self) -> None:
        result = select_best([], _make_kb(["orders"]), "postgres")
        assert result == ""

    def test_prefers_valid_over_invalid(self) -> None:
        kb = _make_kb(["orders"])
        candidates = [
            "NOT VALID SQL",
            "NOT VALID SQL",
            "SELECT COUNT(*) FROM orders",
        ]
        result = select_best(candidates, kb, "postgres")
        assert "SELECT" in result

    def test_all_invalid_returns_first(self) -> None:
        kb = _make_kb(["orders"])
        candidates = ["NOT SQL !!!", "ALSO NOT SQL !!!"]
        result = select_best(candidates, kb, "postgres")
        assert result in candidates

    def test_whitespace_only_candidates_excluded(self) -> None:
        kb = _make_kb(["orders"])
        candidates = ["   ", "SELECT COUNT(*) FROM orders"]
        result = select_best(candidates, kb, "postgres")
        assert "SELECT" in result

    def test_two_valid_candidates_one_returned(self) -> None:
        kb = _make_kb(["orders"])
        candidates = [
            "SELECT COUNT(*) FROM orders",
            "SELECT * FROM orders",
        ]
        result = select_best(candidates, kb, "postgres")
        assert "orders" in result  # some valid result returned


# ---------------------------------------------------------------------------
# validate_and_repair() with self-consistency wired in
# ---------------------------------------------------------------------------


class TestValidateAndRepairWithSelfConsistency:
    """Integration: self-consistency wiring via nlqueries.config.SELF_CONSISTENCY."""

    def _run_repair(self, sql: str, sc_mode: str = "off") -> Any:
        from nlqueries.orchestrator.sql_generation import validate_and_repair

        kb = _make_kb(["orders"])
        llm = MagicMock()
        llm.acomplete = AsyncMock(return_value="<sql>SELECT COUNT(*) FROM orders</sql>")

        # Patch the module-level attribute; the lazy `from nlqueries import config as _cfg`
        # inside validate_and_repair reads `_cfg.SELF_CONSISTENCY` at call time, so
        # patching the attribute on the live module object is sufficient.
        with patch("nlqueries.config.SELF_CONSISTENCY", sc_mode):
            return asyncio.run(validate_and_repair(sql, kb, "postgres", llm))

    def test_sc_off_does_not_call_candidates(self) -> None:
        """When SELF_CONSISTENCY=off, generate_candidates is not called."""
        bad_sql = "SELECT * FROM ghost_table"

        with patch(
            "nlqueries.orchestrator.candidates.generate_candidates",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = []
            self._run_repair(bad_sql, sc_mode="off")

        mock_gen.assert_not_called()

    def test_sc_all_calls_generate_candidates(self) -> None:
        """When SELF_CONSISTENCY=all, generate_candidates is called on repair path."""
        bad_sql = "SELECT * FROM ghost_table"
        fixed = "SELECT COUNT(*) FROM orders"

        with (
            patch(
                "nlqueries.orchestrator.candidates.generate_candidates",
                new_callable=AsyncMock,
            ) as mock_gen,
            patch(
                "nlqueries.orchestrator.candidates.select_best",
                return_value=fixed,
            ),
        ):
            mock_gen.return_value = [fixed, fixed, fixed]
            self._run_repair(bad_sql, sc_mode="all")

        mock_gen.assert_called_once()

    def test_valid_sql_skips_repair_entirely(self) -> None:
        """Valid SQL must return on first attempt without any repair."""
        valid_sql = "SELECT COUNT(*) FROM orders"

        with patch(
            "nlqueries.orchestrator.candidates.generate_candidates",
            new_callable=AsyncMock,
        ) as mock_gen:
            result = self._run_repair(valid_sql, sc_mode="all")

        mock_gen.assert_not_called()
        assert result.is_valid is True
        assert result.attempt_count == 1
