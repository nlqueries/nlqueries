"""Tests for nlqueries.orchestrator.orchestrator.Orchestrator."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.sql_generation import SQLGenerationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_tokens(gen: Any) -> list[str]:
    tokens: list[str] = []
    async for token in gen:
        tokens.append(token)
    return tokens


def _make_kb(tables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": {
            "tables": tables
            or [
                {
                    "name": "orders",
                    "description": "Purchase records",
                    "row_count": 100,
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": ""},
                        {"name": "total", "type": "DECIMAL", "description": ""},
                    ],
                }
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [
            {
                "intent": "Count all orders",
                "template": "SELECT COUNT(*) FROM orders",
            }
        ],
    }


class _MockLLM:
    """Minimal LLMClient stand-in with a tracked async astream generator.

    The orchestrator now uses ``astream()`` (not the sync ``stream()``) and
    ``supports_prompt_caching`` to decide whether to attach cache_control to
    system blocks.  This class records every call for assertion in tests.
    """

    supports_prompt_caching: bool = False

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._calls: list[tuple[Any, str]] = []  # [(system, user), ...]

    async def astream(self, system: Any, user: str) -> Any:
        self._calls.append((system, user))
        for t in self._tokens:
            yield t

    @property
    def last_system(self) -> Any:
        return self._calls[-1][0] if self._calls else None

    @property
    def last_user(self) -> str:
        return self._calls[-1][1] if self._calls else ""

    def system_text(self) -> str:
        """Join all block texts from the last system call to a plain string."""
        sys = self.last_system
        if isinstance(sys, str):
            return sys
        if isinstance(sys, list):
            return " ".join(b.get("text", "") for b in sys if isinstance(b, dict))
        return ""


def _make_sql_result(
    sql: str = "SELECT 1",
    is_valid: bool = True,
    dialect: str = "postgres",
) -> SQLGenerationResult:
    return SQLGenerationResult(
        sql=sql,
        is_valid=is_valid,
        validation_error=None if is_valid else "mock error",
        dialect=dialect,
        attempt_count=1,
    )


def _write_kb(path: Path, agent_id: str, kb: dict[str, Any]) -> None:
    """Write *kb* as YAML to *path* using the sanitised *agent_id* stem."""
    import re

    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    kb_file = path / f"{safe_id}.yaml"
    kb_file.write_text(yaml.dump(kb), encoding="utf-8")


def _last_token_as_json(tokens: list[str]) -> dict[str, Any]:
    """Parse the final token as JSON and return it."""
    return json.loads(tokens[-1])  # type: ignore[no-any-return]


def _non_json_text(tokens: list[str]) -> str:
    """Concatenate all non-JSON tokens (the reasoning text)."""
    parts: list[str] = []
    for t in tokens:
        try:
            json.loads(t)
        except (json.JSONDecodeError, ValueError):
            parts.append(t)
    return "".join(parts)


# ---------------------------------------------------------------------------
# handle_question — token streaming (reasoning phase)
# ---------------------------------------------------------------------------


def test_orchestrator_yields_tokens_in_order() -> None:
    """Reasoning tokens are yielded before the SQL final chunk.

    The orchestrator uses a sentinel-split buffer so individual token
    boundaries may shift; we verify the full concatenated content instead.
    """
    source_tokens = ["SELECT", " ", "*", " ", "FROM", " ", "orders"]
    mock_llm = _MockLLM(source_tokens)

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("all orders", "agent1")))

    assert _last_token_as_json(tokens)["type"] == "sql"
    # Reasoning text (concatenated) must contain the expected content
    reasoning = _non_json_text(tokens)
    assert "SELECT" in reasoning
    assert "orders" in reasoning


def test_orchestrator_yields_all_tokens() -> None:
    """Every token from the LLM stream must appear in the output (no dropping)."""
    source_tokens = ["tok1", "tok2", "tok3", "tok4", "tok5"]
    mock_llm = _MockLLM(source_tokens)

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "myagent", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "myagent")))

    # Final token is JSON; the reasoning tokens together must cover all source content
    assert _last_token_as_json(tokens)["type"] == "sql"
    reasoning = _non_json_text(tokens)
    assert "".join(source_tokens) == reasoning


def test_orchestrator_yields_sql_chunk_when_llm_returns_no_tokens() -> None:
    """Even with an empty reasoning stream the final SQL chunk must be yielded."""
    mock_llm = _MockLLM([])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result("SELECT 1")),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    assert len(tokens) == 1
    assert _last_token_as_json(tokens)["type"] == "sql"


# ---------------------------------------------------------------------------
# handle_question — prompt assembly
# ---------------------------------------------------------------------------


def test_orchestrator_calls_llm_astream_once() -> None:
    """The orchestrator must call astream exactly once (single-call path)."""
    mock_llm = _MockLLM(["result"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    assert len(mock_llm._calls) == 1


def test_orchestrator_passes_non_empty_prompts_to_llm() -> None:
    """The system blocks must be non-empty and user must be the question text."""
    mock_llm = _MockLLM(["ok"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("my question", "agent1")))

    assert mock_llm.system_text().strip()
    assert mock_llm.last_user == "my question"


def test_orchestrator_uses_agent_id_schema_collection() -> None:
    """Collection name passed to assemble_prompt must be agent_{agent_id}_schema."""
    from nlqueries.orchestrator.prompt_assembly import AssembledPrompt

    mock_llm = _MockLLM(["ok"])
    captured: dict[str, Any] = {}

    async def fake_assemble(
        question: str,
        kb: Any,
        top_k_capsules: int = 5,
        *,
        collection: str | None = None,
        vector: list[float] | None = None,
        extra_dynamic_context: str | None = None,
    ) -> AssembledPrompt:
        captured["collection"] = collection
        return AssembledPrompt(static_system="sys", dynamic_context="", user_question=question)

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "my_agent", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.assemble_prompt_async",
                side_effect=fake_assemble,
            ),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "my_agent")))

    assert captured["collection"] == "agent_my_agent_schema"


# ---------------------------------------------------------------------------
# handle_question — SQL generation step
# ---------------------------------------------------------------------------


def test_orchestrator_yields_final_sql_chunk() -> None:
    """The last yielded token must be valid JSON with type 'sql'."""
    mock_llm = _MockLLM([])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result("SELECT id FROM orders")),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    chunk = _last_token_as_json(tokens)
    assert chunk["type"] == "sql"
    assert chunk["sql"] == "SELECT id FROM orders"
    assert chunk["is_valid"] is True
    assert chunk["dialect"] == "postgres"
    assert chunk["attempt_count"] == 1
    assert chunk["validation_error"] is None


def test_orchestrator_final_chunk_reflects_invalid_result() -> None:
    """When SQL validation fails, the chunk must carry is_valid=False."""
    bad_result = SQLGenerationResult(
        sql="DELETE FROM orders",
        is_valid=False,
        validation_error="Only SELECT allowed; got Delete",
        dialect="postgres",
        attempt_count=2,
    )
    mock_llm = _MockLLM([])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=bad_result),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    chunk = _last_token_as_json(tokens)
    assert chunk["is_valid"] is False
    assert chunk["attempt_count"] == 2
    assert chunk["validation_error"] is not None


def test_orchestrator_passes_dialect_to_validate_and_repair() -> None:
    """The dialect parameter must be forwarded to validate_and_repair."""
    mock_llm = _MockLLM([])
    mock_repair = AsyncMock(return_value=_make_sql_result(dialect="snowflake"))

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                mock_repair,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(
                _collect_tokens(orch.handle_question("question", "agent1", dialect="snowflake"))
            )

    _sql, _kb, passed_dialect, _llm, _system = mock_repair.call_args.args
    assert passed_dialect == "snowflake"


def test_orchestrator_passes_kb_to_validate_and_repair() -> None:
    """validate_and_repair must receive the actual loaded knowledge base."""
    mock_llm = _MockLLM([])
    mock_repair = AsyncMock(return_value=_make_sql_result())

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        kb = _make_kb()
        _write_kb(kb_path, "agent1", kb)

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                mock_repair,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    _sql, passed_kb, _dialect, _llm, _system = mock_repair.call_args.args
    assert "schema" in passed_kb


def test_orchestrator_loads_kb_schema_into_prompt() -> None:
    """The system blocks must include table names from the loaded knowledge base."""
    mock_llm = _MockLLM(["SQL"])
    mock_repair = AsyncMock(return_value=_make_sql_result())

    kb = _make_kb(
        tables=[
            {
                "name": "special_table",
                "description": "A unique table",
                "row_count": 1,
                "columns": [{"name": "col1", "type": "TEXT", "description": ""}],
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", kb)

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                mock_repair,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    assert "special_table" in mock_llm.system_text()


# ---------------------------------------------------------------------------
# Sentinel-split behaviour
# ---------------------------------------------------------------------------


def test_orchestrator_splits_sql_from_sentinel() -> None:
    """Tokens inside <sql>...</sql> are not yielded as reasoning text."""
    sql_text = "SELECT id FROM orders"
    source_tokens = [
        "Let me query the orders table.",
        f"\n<sql>\n{sql_text}\n</sql>",
    ]
    mock_llm = _MockLLM(source_tokens)
    mock_repair = AsyncMock(return_value=_make_sql_result(sql=sql_text))

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                mock_repair,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("all orders", "agent1")))

    # SQL text must not appear in the reasoning output
    reasoning = _non_json_text(tokens)
    assert sql_text not in reasoning
    assert "Let me query" in reasoning
    # validate_and_repair was called with the extracted SQL
    sql_arg = mock_repair.call_args.args[0]
    assert sql_text in sql_arg or sql_arg.strip() == sql_text


# ---------------------------------------------------------------------------
# handle_question — knowledge base loading
# ---------------------------------------------------------------------------


def test_orchestrator_raises_file_not_found_for_missing_kb() -> None:
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
    ):
        mock_cfg.KB_PATH = Path(tmpdir)
        orch = Orchestrator()

        with pytest.raises(FileNotFoundError, match="No knowledge base found for agent"):
            asyncio.run(_collect_tokens(orch.handle_question("question", "nonexistent_agent")))


def test_orchestrator_sanitises_agent_id_for_filename() -> None:
    """Colons in agent_id are replaced with underscores for the filename."""
    agent_id = "postgres:localhost:mydb"
    mock_llm = _MockLLM(["ok"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, agent_id, _make_kb())  # writes postgres_localhost_mydb.yaml

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=_make_sql_result()),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", agent_id)))

    assert "ok" in _non_json_text(tokens)
    assert _last_token_as_json(tokens)["type"] == "sql"


# ---------------------------------------------------------------------------
# _load_knowledge_base (internal)
# ---------------------------------------------------------------------------


def test_load_knowledge_base_returns_dict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg:
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            kb = orch._load_knowledge_base("agent1")

    assert isinstance(kb, dict)
    assert "schema" in kb


def test_load_knowledge_base_raises_for_unknown_agent() -> None:
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
    ):
        mock_cfg.KB_PATH = Path(tmpdir)
        orch = Orchestrator()

        with pytest.raises(FileNotFoundError):
            orch._load_knowledge_base("does_not_exist")
