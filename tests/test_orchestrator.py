"""Tests for nlqueries.orchestrator.orchestrator.Orchestrator."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nlqueries.orchestrator.orchestrator import Orchestrator

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


def _make_mock_llm(tokens: list[str]) -> MagicMock:
    """Return a mock LLMClient whose stream() yields *tokens*."""
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter(tokens)
    return mock_llm


def _write_kb(path: Path, agent_id: str, kb: dict[str, Any]) -> None:
    """Write *kb* as YAML to *path* using the sanitised *agent_id* stem."""
    import re

    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    kb_file = path / f"{safe_id}.yaml"
    kb_file.write_text(yaml.dump(kb), encoding="utf-8")


# ---------------------------------------------------------------------------
# handle_question — token streaming
# ---------------------------------------------------------------------------


def test_orchestrator_yields_tokens_in_order() -> None:
    expected = ["SELECT", " ", "*", " ", "FROM", " ", "orders"]

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch(
                "nlqueries.orchestrator.orchestrator.get_llm_client",
                return_value=_make_mock_llm(expected),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("all orders", "agent1")))

    assert tokens == expected


def test_orchestrator_yields_all_tokens() -> None:
    all_tokens = ["tok1", "tok2", "tok3", "tok4", "tok5"]

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "myagent", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch(
                "nlqueries.orchestrator.orchestrator.get_llm_client",
                return_value=_make_mock_llm(all_tokens),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "myagent")))

    assert tokens == all_tokens


def test_orchestrator_yields_empty_when_llm_returns_no_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch(
                "nlqueries.orchestrator.orchestrator.get_llm_client",
                return_value=_make_mock_llm([]),
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    assert tokens == []


# ---------------------------------------------------------------------------
# handle_question — prompt assembly
# ---------------------------------------------------------------------------


def test_orchestrator_calls_llm_stream_once() -> None:
    mock_llm = _make_mock_llm(["result"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    mock_llm.stream.assert_called_once()


def test_orchestrator_passes_non_empty_prompts_to_llm() -> None:
    mock_llm = _make_mock_llm(["ok"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "agent1", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("my question", "agent1")))

    system_arg, user_arg = mock_llm.stream.call_args.args
    assert system_arg.strip()
    assert user_arg == "my question"


def test_orchestrator_uses_agent_id_schema_collection() -> None:
    """Collection name passed to assemble_prompt must be agent_{agent_id}_schema."""
    mock_llm = _make_mock_llm(["ok"])
    captured: dict[str, Any] = {}

    def fake_assemble(
        question: str,
        kb: Any,
        top_k_capsules: int = 5,
        *,
        collection: str | None = None,
    ) -> tuple[str, str]:
        captured["collection"] = collection
        return "sys", "usr"

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, "my_agent", _make_kb())

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch(
                "nlqueries.orchestrator.orchestrator.assemble_prompt",
                side_effect=fake_assemble,
            ),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "my_agent")))

    assert captured["collection"] == "agent_my_agent_schema"


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
    mock_llm = _make_mock_llm(["ok"])

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        _write_kb(kb_path, agent_id, _make_kb())  # writes postgres_localhost_mydb.yaml

        with (
            patch("nlqueries.orchestrator.orchestrator.config") as mock_cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=mock_llm),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            tokens = asyncio.run(_collect_tokens(orch.handle_question("question", agent_id)))

    assert tokens == ["ok"]


def test_orchestrator_loads_kb_schema_into_prompt() -> None:
    """The system prompt must include table names from the loaded knowledge base."""
    mock_llm = _make_mock_llm(["SQL"])

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
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        ):
            mock_cfg.KB_PATH = kb_path
            orch = Orchestrator()
            asyncio.run(_collect_tokens(orch.handle_question("question", "agent1")))

    system_arg, _ = mock_llm.stream.call_args.args
    assert "special_table" in system_arg


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
