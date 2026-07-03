"""Tests for nlqueries.mcp_server.server."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    answer: str = "There are 42 orders.",
    agent_type: str = "sql",
    sql: str | None = "SELECT COUNT(*) FROM orders",
    sql_result: Any = None,
    citations: list[Any] | None = None,
    latency_ms: int = 250,
) -> MagicMock:
    r = MagicMock()
    r.answer = answer
    r.agent_type = agent_type
    r.sql = sql
    r.sql_result = sql_result
    r.citations = citations or []
    r.latency_ms = latency_ms
    return r


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_returns_empty_when_kb_path_missing(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_agents

        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path / "nonexistent"):
            result = list_agents()

        assert result == []

    def test_returns_sorted_agent_ids(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_agents

        for name in ("zebra", "alpha", "sales"):
            (tmp_path / f"{name}.yaml").write_text("schema: {}")

        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            result = list_agents()

        assert result == ["alpha", "sales", "zebra"]

    def test_ignores_non_yaml_files(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_agents

        (tmp_path / "agent.yaml").write_text("")
        (tmp_path / "notes.txt").write_text("")
        (tmp_path / "agent.json").write_text("")

        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            result = list_agents()

        assert result == ["agent"]

    def test_returns_empty_when_dir_is_empty(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_agents

        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            result = list_agents()

        assert result == []


# ---------------------------------------------------------------------------
# query — output formatting
# ---------------------------------------------------------------------------


class TestQuery:
    def test_answer_always_present(self) -> None:
        from nlqueries.mcp_server.server import query

        result = _make_result(answer="42 orders.", sql=None)
        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("How many orders?", "sales")

        assert "42 orders." in out

    def test_sql_block_included_when_present(self) -> None:
        from nlqueries.mcp_server.server import query

        result = _make_result(sql="SELECT COUNT(*) FROM orders")
        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("How many orders?", "sales")

        assert "```sql" in out
        assert "SELECT COUNT(*) FROM orders" in out

    def test_sql_block_absent_when_none(self) -> None:
        from nlqueries.mcp_server.server import query

        result = _make_result(sql=None)
        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("Summarise the docs", "docs_agent")

        assert "```sql" not in out

    def test_citations_included_when_present(self) -> None:
        from nlqueries.mcp_server.server import query

        citation = MagicMock()
        citation.source_name = "report.pdf"
        citation.page_number = 3
        citation.excerpt = "Revenue grew 15% YoY."
        result = _make_result(agent_type="document", sql=None, citations=[citation])

        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("What drove revenue?", "docs_agent")

        assert "report.pdf" in out
        assert "page 3" in out
        assert "Revenue grew 15% YoY." in out

    def test_citation_without_page_number(self) -> None:
        from nlqueries.mcp_server.server import query

        citation = MagicMock()
        citation.source_name = "wiki.md"
        citation.page_number = None
        citation.excerpt = ""
        result = _make_result(agent_type="document", sql=None, citations=[citation])

        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("What is X?", "docs_agent")

        assert "wiki.md" in out
        assert "page" not in out

    def test_long_excerpt_truncated(self) -> None:
        from nlqueries.mcp_server.server import query

        citation = MagicMock()
        citation.source_name = "big.pdf"
        citation.page_number = None
        citation.excerpt = "x" * 300
        result = _make_result(agent_type="document", sql=None, citations=[citation])

        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("Tell me about X", "docs_agent")

        assert "…" in out
        # excerpt in output must be at most 200 chars + the ellipsis
        excerpt_start = out.index('"') + 1
        excerpt_end = out.index("…")
        assert excerpt_end - excerpt_start <= 200

    def test_sql_execution_error_shown(self) -> None:
        from nlqueries.mcp_server.server import query

        sql_result = MagicMock()
        sql_result.error = "relation 'orders' does not exist"
        result = _make_result(sql_result=sql_result)

        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("How many orders?", "sales")

        assert "relation 'orders' does not exist" in out

    def test_metadata_footer_present(self) -> None:
        from nlqueries.mcp_server.server import query

        result = _make_result(agent_type="sql", latency_ms=123)
        with patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result):
            out = query("How many orders?", "sales")

        assert "sql" in out
        assert "123 ms" in out

    def test_agent_not_found_returns_friendly_message(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import query

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.run_query_sync",
                side_effect=FileNotFoundError("kb not found"),
            ),
            patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path),
        ):
            out = query("How many orders?", "ghost_agent")

        assert "ghost_agent" in out
        assert "not found" in out.lower()

    def test_unexpected_exception_returns_error_string(self) -> None:
        from nlqueries.mcp_server.server import query

        with patch(
            "nlqueries.orchestrator.sync_runner.run_query_sync",
            side_effect=RuntimeError("LLM timeout"),
        ):
            out = query("How many orders?", "sales")

        assert "LLM timeout" in out

    def test_passes_dialect_to_run_query_sync(self) -> None:
        from nlqueries.mcp_server.server import query

        result = _make_result()
        with patch(
            "nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result
        ) as mock_run:
            query("How many orders?", "sales", dialect="snowflake")

        _, kwargs = mock_run.call_args
        assert kwargs.get("dialect") == "snowflake"


# ---------------------------------------------------------------------------
# Server object
# ---------------------------------------------------------------------------


class TestMcpServerObject:
    def test_server_name(self) -> None:
        from nlqueries.mcp_server.server import mcp

        assert mcp.name == "nlqueries"

    def test_list_agents_registered_as_tool(self) -> None:
        from nlqueries.mcp_server.server import mcp

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "list_agents" in tool_names

    def test_query_registered_as_tool(self) -> None:
        from nlqueries.mcp_server.server import mcp

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "query" in tool_names
