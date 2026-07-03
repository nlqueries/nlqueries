"""Tests for nlqueries.mcp_server.server — all 9 tools."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
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


def _feedback(
    question: str = "How many orders?",
    generated_sql: str = "SELECT COUNT(*) FROM orders",
    rating: str = "up",
    agent_id: str = "sales",
    corrected_sql: str | None = None,
    ts: datetime | None = None,
) -> MagicMock:
    fb = MagicMock()
    fb.question = question
    fb.generated_sql = generated_sql
    fb.rating = rating
    fb.agent_id = agent_id
    fb.corrected_sql = corrected_sql
    fb.timestamp = ts or datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return fb


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
# get_agent_schema
# ---------------------------------------------------------------------------


class TestGetAgentSchema:
    def _write_kb(self, tmp_path: Path, content: str, name: str = "sales") -> None:
        (tmp_path / f"{name}.yaml").write_text(textwrap.dedent(content))

    def test_agent_not_found(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("ghost")

        assert "ghost" in out
        assert "not found" in out.lower()

    def test_table_names_present(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        self._write_kb(
            tmp_path,
            """
            schema:
              tables:
                - name: orders
                  columns:
                    - name: order_id
                      type: BIGINT
                      primary_key: true
                    - name: customer_id
                      type: BIGINT
            """,
        )
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "orders" in out
        assert "order_id" in out
        assert "customer_id" in out

    def test_table_description_included(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        self._write_kb(
            tmp_path,
            """
            schema:
              tables:
                - name: orders
                  description: One row per placed order
                  columns: []
            """,
        )
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "One row per placed order" in out

    def test_pk_flag_shown(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        self._write_kb(
            tmp_path,
            """
            schema:
              tables:
                - name: orders
                  columns:
                    - name: order_id
                      type: BIGINT
                      primary_key: true
            """,
        )
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "PK" in out

    def test_fk_flag_shown(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        self._write_kb(
            tmp_path,
            """
            schema:
              tables:
                - name: orders
                  columns:
                    - name: customer_id
                      type: BIGINT
                      foreign_key: customers.customer_id
            """,
        )
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "FK" in out
        assert "customers.customer_id" in out

    def test_no_tables_message(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        self._write_kb(tmp_path, "schema:\n  tables: []\n")
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "no tables" in out.lower()

    def test_bad_yaml_returns_error(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import get_agent_schema

        (tmp_path / "sales.yaml").write_text("{{invalid: yaml: content: [")
        with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
            out = get_agent_schema("sales")

        assert "Failed" in out or "error" in out.lower()


# ---------------------------------------------------------------------------
# submit_feedback
# ---------------------------------------------------------------------------


class TestSubmitFeedback:
    def test_thumbs_up_success(self) -> None:
        from nlqueries.mcp_server.server import submit_feedback

        with patch("nlqueries.feedback.store.record_feedback") as mock_record:
            out = submit_feedback(
                question="How many orders?",
                agent_id="sales",
                generated_sql="SELECT COUNT(*) FROM orders",
                rating="up",
            )

        mock_record.assert_called_once()
        assert "sales" in out

    def test_thumbs_down_with_correction(self) -> None:
        from nlqueries.mcp_server.server import submit_feedback

        with patch("nlqueries.feedback.store.record_feedback") as mock_record:
            out = submit_feedback(
                question="How many orders?",
                agent_id="sales",
                generated_sql="SELECT COUNT(*) FROM order",  # wrong table
                rating="down",
                corrected_sql="SELECT COUNT(*) FROM orders",
            )

        mock_record.assert_called_once()
        assert "promote-feedback" in out or "Corrected" in out

    def test_invalid_rating_rejected(self) -> None:
        from nlqueries.mcp_server.server import submit_feedback

        out = submit_feedback(
            question="q",
            agent_id="sales",
            generated_sql="SELECT 1",
            rating="sideways",
        )

        assert "Invalid rating" in out or "must be" in out

    def test_record_feedback_exception_returns_error(self) -> None:
        from nlqueries.mcp_server.server import submit_feedback

        with patch(
            "nlqueries.feedback.store.record_feedback",
            side_effect=OSError("disk full"),
        ):
            out = submit_feedback(
                question="q",
                agent_id="sales",
                generated_sql="SELECT 1",
                rating="up",
            )

        assert "Failed" in out or "disk full" in out


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


class TestHealth:
    def _patches(
        self,
        *,
        llm_key: str = "sk-test",
        llm_ok: bool = True,
        qdrant_ok: bool = True,
        daemon_vec: list[float] | None = None,
        agents: list[str] | None = None,
    ):
        """Context manager stack for health() test isolation."""
        from contextlib import ExitStack

        stack = ExitStack()

        stack.enter_context(patch("nlqueries.mcp_server.server.config.ANTHROPIC_API_KEY", llm_key))
        stack.enter_context(patch("nlqueries.mcp_server.server.config.LLM_PROVIDER", "anthropic"))
        stack.enter_context(
            patch("nlqueries.mcp_server.server.config.LLM_MODEL", "claude-sonnet-4-5")
        )

        mock_llm = MagicMock()
        if not llm_ok:
            mock_llm.complete.side_effect = RuntimeError("API unreachable")
        stack.enter_context(patch("nlqueries.llm.get_llm_client", return_value=mock_llm))

        if qdrant_ok:
            mock_urlopen = MagicMock()
            stack.enter_context(patch("urllib.request.urlopen", return_value=mock_urlopen))
        else:
            stack.enter_context(
                patch(
                    "urllib.request.urlopen",
                    side_effect=OSError("connection refused"),
                )
            )

        stack.enter_context(
            patch(
                "nlqueries.embeddings.embedder._try_daemon_single",
                return_value=daemon_vec,
            )
        )

        stack.enter_context(
            patch(
                "nlqueries.mcp_server.server.list_agents",
                return_value=agents or [],
            )
        )

        return stack

    def test_all_green(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(daemon_vec=[0.1] * 384, agents=["sales"]):
            out = health()

        assert "✅" in out

    def test_missing_api_key_shows_error(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(llm_key=""):
            out = health()

        assert "❌" in out
        assert "ANTHROPIC_API_KEY" in out

    def test_qdrant_down_shown(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(qdrant_ok=False):
            out = health()

        assert "Qdrant" in out
        assert "❌" in out

    def test_embed_daemon_not_running_is_warning(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(daemon_vec=None):
            out = health()

        assert "⚠" in out
        assert "Embed daemon" in out

    def test_no_agents_is_warning(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(agents=[]):
            out = health()

        assert "⚠" in out

    def test_agents_listed_when_present(self) -> None:
        from nlqueries.mcp_server.server import health

        with self._patches(agents=["sales", "support"]):
            out = health()

        assert "sales" in out
        assert "support" in out


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------


class TestInvalidateCache:
    def test_success_message(self) -> None:
        from nlqueries.mcp_server.server import invalidate_cache

        with patch("nlqueries.cache.semantic_cache.SemanticCache") as MockCache:
            invalidate_cache("sales")

        MockCache.return_value.invalidate.assert_called_once_with("sales")

    def test_success_message_contains_agent_id(self) -> None:
        from nlqueries.mcp_server.server import invalidate_cache

        with patch("nlqueries.cache.semantic_cache.SemanticCache"):
            out = invalidate_cache("sales")

        assert "sales" in out
        assert "cleared" in out.lower() or "invalidat" in out.lower()

    def test_exception_returns_error_string(self) -> None:
        from nlqueries.mcp_server.server import invalidate_cache

        with patch("nlqueries.cache.semantic_cache.SemanticCache") as MockCache:
            MockCache.return_value.invalidate.side_effect = RuntimeError("Qdrant down")
            out = invalidate_cache("sales")

        assert "failed" in out.lower() or "Qdrant down" in out


# ---------------------------------------------------------------------------
# list_connectors
# ---------------------------------------------------------------------------


class TestListConnectors:
    def test_no_file_returns_guidance(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_connectors

        with patch(
            "nlqueries.mcp_server.server.config.CONNECTORS_FILE",
            tmp_path / "nonexistent.yaml",
        ):
            out = list_connectors()

        assert "No connectors" in out

    def test_empty_file_returns_guidance(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_connectors

        f = tmp_path / "connectors.yaml"
        f.write_text("{}")
        with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
            out = list_connectors()

        assert "No connectors" in out

    def test_connector_ids_and_types_shown(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_connectors

        f = tmp_path / "connectors.yaml"
        f.write_text(
            "prod_db:\n"
            "  db_type: postgres\n"
            "  host: db.example.com\n"
            "  database: mydb\n"
            "analytics:\n"
            "  db_type: redshift\n"
        )
        with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
            out = list_connectors()

        assert "prod_db" in out
        assert "postgres" in out
        assert "analytics" in out
        assert "redshift" in out

    def test_passwords_not_shown(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_connectors

        f = tmp_path / "connectors.yaml"
        f.write_text("prod_db:\n  db_type: postgres\n  password: super_secret_pass\n")
        with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
            out = list_connectors()

        assert "super_secret_pass" not in out

    def test_bad_yaml_returns_error(self, tmp_path: Path) -> None:
        from nlqueries.mcp_server.server import list_connectors

        f = tmp_path / "connectors.yaml"
        f.write_text("{{invalid")
        with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f):
            out = list_connectors()

        assert "Failed" in out or "error" in out.lower()


# ---------------------------------------------------------------------------
# get_query_history
# ---------------------------------------------------------------------------


class TestGetQueryHistory:
    def test_no_history_returns_message(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        with patch("nlqueries.feedback.store.load_feedback", return_value=[]):
            out = get_query_history("sales")

        assert "No query history" in out or "not found" in out.lower()

    def test_records_shown_newest_first(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        old = _feedback(
            question="Old question",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
        )
        new = _feedback(
            question="New question",
            ts=datetime(2026, 6, 1, tzinfo=UTC),
        )
        with patch("nlqueries.feedback.store.load_feedback", return_value=[old, new]):
            out = get_query_history("sales")

        new_idx = out.index("New question")
        old_idx = out.index("Old question")
        assert new_idx < old_idx  # newest first

    def test_rating_icons_present(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        records = [
            _feedback(rating="up"),
            _feedback(rating="down", question="Bad query"),
        ]
        with patch("nlqueries.feedback.store.load_feedback", return_value=records):
            out = get_query_history("sales")

        assert "👍" in out
        assert "👎" in out

    def test_limit_respected(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        records = [
            _feedback(
                question=f"Q{i}",
                ts=datetime(2026, 1, i + 1, tzinfo=UTC),
            )
            for i in range(10)
        ]
        with patch("nlqueries.feedback.store.load_feedback", return_value=records):
            out = get_query_history("sales", limit=3)

        shown = [f"Q{i}" in out for i in range(10)]
        assert sum(shown) == 3

    def test_load_exception_returns_error(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        with patch(
            "nlqueries.feedback.store.load_feedback",
            side_effect=OSError("file gone"),
        ):
            out = get_query_history("sales")

        assert "Failed" in out or "file gone" in out

    def test_corrected_sql_shown_for_down_votes(self) -> None:
        from nlqueries.mcp_server.server import get_query_history

        fb = _feedback(
            rating="down",
            corrected_sql="SELECT COUNT(*) FROM orders WHERE status='active'",
        )
        with patch("nlqueries.feedback.store.load_feedback", return_value=[fb]):
            out = get_query_history("sales")

        assert "Corrected" in out


# ---------------------------------------------------------------------------
# get_cache_stats
# ---------------------------------------------------------------------------


class TestGetCacheStats:
    def test_stats_shown(self) -> None:
        from nlqueries.mcp_server.server import get_cache_stats

        with patch("nlqueries.cache.semantic_cache.SemanticCache") as MockCache:
            MockCache.return_value.stats.return_value = {
                "total_entries": 42,
                "collection": "cache_sales",
            }
            out = get_cache_stats("sales")

        assert "42" in out
        assert "cache_sales" in out

    def test_agent_id_in_output(self) -> None:
        from nlqueries.mcp_server.server import get_cache_stats

        with patch("nlqueries.cache.semantic_cache.SemanticCache") as MockCache:
            MockCache.return_value.stats.return_value = {
                "total_entries": 0,
                "collection": "cache_sales",
            }
            out = get_cache_stats("sales")

        assert "sales" in out

    def test_exception_returns_error(self) -> None:
        from nlqueries.mcp_server.server import get_cache_stats

        with patch("nlqueries.cache.semantic_cache.SemanticCache") as MockCache:
            MockCache.return_value.stats.side_effect = RuntimeError("Qdrant gone")
            out = get_cache_stats("sales")

        assert "Could not" in out or "Qdrant gone" in out


# ---------------------------------------------------------------------------
# MCP server object — all 9 tools registered
# ---------------------------------------------------------------------------


class TestMcpServerObject:
    def test_server_name(self) -> None:
        from nlqueries.mcp_server.server import mcp

        assert mcp.name == "nlqueries"

    def _tool_names(self) -> set[str]:
        from nlqueries.mcp_server.server import mcp

        return {t.name for t in mcp._tool_manager.list_tools()}

    def test_list_agents_registered(self) -> None:
        assert "list_agents" in self._tool_names()

    def test_query_registered(self) -> None:
        assert "query" in self._tool_names()

    def test_get_agent_schema_registered(self) -> None:
        assert "get_agent_schema" in self._tool_names()

    def test_submit_feedback_registered(self) -> None:
        assert "submit_feedback" in self._tool_names()

    def test_health_registered(self) -> None:
        assert "health" in self._tool_names()

    def test_invalidate_cache_registered(self) -> None:
        assert "invalidate_cache" in self._tool_names()

    def test_list_connectors_registered(self) -> None:
        assert "list_connectors" in self._tool_names()

    def test_get_query_history_registered(self) -> None:
        assert "get_query_history" in self._tool_names()

    def test_get_cache_stats_registered(self) -> None:
        assert "get_cache_stats" in self._tool_names()

    def test_all_nine_tools_registered(self) -> None:
        expected = {
            "list_agents",
            "get_agent_schema",
            "query",
            "submit_feedback",
            "health",
            "invalidate_cache",
            "list_connectors",
            "get_query_history",
            "get_cache_stats",
        }
        assert expected <= self._tool_names()
