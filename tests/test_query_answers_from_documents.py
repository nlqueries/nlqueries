"""`query` and the MCP `query` tool answer from an agent's listed document sources.

Routing to documents, or to documents and the database together, existed in the
library and was reached only from Python: both entry points passed "SQL only".
An agent that lists document sources (`nlqueries agent docs add`) now offers
document and hybrid answers from both, searching every listed source; an agent
that lists none answers from its database alone, exactly as before.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from nlqueries import config
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.orchestrator import document_retrieval, multi_agent_orchestrator
from nlqueries.orchestrator.document_retrieval import Citation
from nlqueries.orchestrator.sync_runner import AgentQueryResult, answer_types_for

_AGENT = "postgres:localhost:shop"


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "connectors.yaml"
    path.write_text(
        yaml.dump(
            {
                _AGENT: {
                    "db_type": "postgres",
                    "database": "shop",
                    "alias": "shop",
                    "document_sources": ["q1-report", "eng-wiki"],
                },
                "postgres:localhost:plain": {"db_type": "postgres", "database": "plain"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_main, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    return path


# ---------------------------------------------------------------------------
# What an agent offers, and where its documents are
# ---------------------------------------------------------------------------


def test_an_agent_with_document_sources_offers_documents_and_hybrid(
    connectors_file: Path,
) -> None:
    """`hybrid` is listed: the router downgrades it to `sql` when it is not."""
    assert answer_types_for(_AGENT) == ("sql", "document", "hybrid")
    assert answer_types_for("postgres_localhost_shop") == ("sql", "document", "hybrid")


def test_an_agent_without_them_answers_from_its_database_as_before(
    connectors_file: Path,
) -> None:
    assert answer_types_for("postgres:localhost:plain") == ("sql",)
    assert answer_types_for("not-registered") == ("sql",)


def test_an_unreadable_connectors_file_means_database_only(
    connectors_file: Path,
) -> None:
    connectors_file.write_text("[unclosed", encoding="utf-8")

    assert answer_types_for(_AGENT) == ("sql",)


def test_the_documents_searched_are_the_listed_sources(connectors_file: Path) -> None:
    assert multi_agent_orchestrator._document_collections(_AGENT) == [
        "doc_q1-report_chunks",
        "doc_eng-wiki_chunks",
    ]


def test_an_agent_listing_none_reads_the_collection_named_after_it(
    connectors_file: Path,
) -> None:
    """As before, and as an embedder whose agents are not in the connectors file gets."""
    assert multi_agent_orchestrator._document_collections("plain-agent") == (
        "doc_plain-agent_chunks"
    )


# ---------------------------------------------------------------------------
# Searching several collections
# ---------------------------------------------------------------------------


def _hit(source: str, n: int, score: float) -> Any:
    payload = {
        "chunk_id": f"{source}{n:06d}".ljust(32, "0")[:32],
        "source_id": source,
        "source_name": f"{source}.md",
        "page_number": n,
        "chunk_index": n,
        "text": f"{source} chunk {n}",
        "metadata": {},
    }
    return SimpleNamespace(payload=payload, score=score)


class _Collections:
    def __init__(self, hits: dict[str, list[Any]]) -> None:
        self.hits = hits
        self.searched: list[str] = []

    def query_points(self, collection_name: str, query: Any, limit: int, **_: Any) -> Any:
        self.searched.append(collection_name)
        if collection_name not in self.hits:
            raise RuntimeError(f"Collection {collection_name} not found")
        return SimpleNamespace(points=self.hits[collection_name][:limit])


@pytest.fixture
def collections(monkeypatch: pytest.MonkeyPatch) -> _Collections:
    store = _Collections(
        {
            "doc_a_chunks": [_hit("a", 1, 0.91), _hit("a", 2, 0.40)],
            "doc_b_chunks": [_hit("b", 1, 0.85), _hit("b", 2, 0.70)],
        }
    )
    monkeypatch.setattr("nlqueries.embeddings.qdrant_store._client", store)
    monkeypatch.setattr("nlqueries.embeddings.embedder.embed_text", lambda text: [0.1] * 384)
    document_retrieval._UNSEARCHABLE_WARNED.clear()
    return store


def test_several_sources_give_the_nearest_chunks_overall(collections: _Collections) -> None:
    result = document_retrieval.retrieve_across("q", ["doc_a_chunks", "doc_b_chunks"], top_k=3)

    assert [c.source_name for c in result.citations] == ["a.md", "b.md", "b.md"]
    assert [round(c.relevance_score, 2) for c in result.citations] == [0.91, 0.85, 0.70]
    assert result.collection == "doc_a_chunks, doc_b_chunks"


def test_a_source_not_ingested_is_skipped_and_reported_once(
    collections: _Collections, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            result = document_retrieval.retrieve_across(
                "q", ["doc_missing_chunks", "doc_a_chunks"], top_k=2
            )

    assert [c.source_name for c in result.citations] == ["a.md", "a.md"]
    assert caplog.text.count("Could not search 'doc_missing_chunks'") == 1


def test_the_document_agent_searches_a_list_across_and_a_name_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlqueries.orchestrator import document_orchestrator

    calls: list[tuple[str, Any]] = []
    empty = document_retrieval.DocumentRetrievalResult(chunks=[], citations=[], collection="x")

    def across(question: str, collections: list[str], top_k: int = 5) -> Any:
        calls.append(("across", collections))
        return empty

    def one(question: str, collection: str, top_k: int = 5, source_id_filter: Any = None) -> Any:
        calls.append(("one", collection))
        return empty

    llm = MagicMock()

    async def stream(*_: Any) -> Any:
        yield "answer"

    llm.astream = stream
    monkeypatch.setattr(document_orchestrator, "retrieve_across", across)
    monkeypatch.setattr(document_orchestrator, "retrieve_for_question", one)
    monkeypatch.setattr(document_orchestrator, "get_llm_client", lambda: llm)

    async def drain(collection: Any) -> None:
        async for _ in document_orchestrator.DocumentOrchestrator().handle_question(
            "q", collection
        ):
            pass

    asyncio.run(drain(["doc_a_chunks", "doc_b_chunks"]))
    asyncio.run(drain("doc_a_chunks"))

    assert calls == [("across", ["doc_a_chunks", "doc_b_chunks"]), ("one", "doc_a_chunks")]


# ---------------------------------------------------------------------------
# The two entry points
# ---------------------------------------------------------------------------


def _result(agent_type: str, **fields: Any) -> AgentQueryResult:
    base: dict[str, Any] = {
        "question": "q",
        "resolved_question": "q",
        "agent_type": agent_type,
        "answer": "",
        "sql": None,
        "sql_result": None,
        "sql_is_valid": True,
        "execution_mode": "execute_read_only",
        "citations": [],
        "merged_answer": None,
        "latency_ms": 5,
        "session_id": None,
    }
    base.update(fields)
    return AgentQueryResult(**base)


def test_query_offers_the_agents_answer_types(connectors_file: Path) -> None:
    seen: dict[str, Any] = {}

    def run(question: str, agent_id: str, **kwargs: Any) -> AgentQueryResult:
        seen.update(kwargs)
        return _result("document", answer="From the report.")

    with patch("nlqueries.orchestrator.sync_runner.run_query_sync", run):
        result = CliRunner().invoke(cli, ["query", "shop", "What does the report say?"])

    assert result.exit_code == 0, result.output
    assert seen["available_types"] == ("sql", "document", "hybrid")


def test_query_prints_a_hybrid_answer_and_names_its_sources(connectors_file: Path) -> None:
    """The merged answer was printed only when no rows came back, so a hybrid
    answer, which has rows, never showed it. Sources are listed, as text."""
    from nlqueries.connectors.base import QueryResult

    rows = QueryResult(columns=["n"], rows=[[3]], row_count=1, execution_time_ms=1.0, error=None)
    citation = Citation(
        chunk_id="c",
        source_name="Q1 [final].pdf",
        page_number=4,
        chunk_index=0,
        excerpt="...",
        relevance_score=0.9,
    )

    def run(question: str, agent_id: str, **kwargs: Any) -> AgentQueryResult:
        return _result(
            "hybrid",
            answer="Three orders, as the Q1 report also says.",
            sql="SELECT 3",
            sql_result=rows,
            citations=[citation],
        )

    with patch("nlqueries.orchestrator.sync_runner.run_query_sync", run):
        result = CliRunner().invoke(cli, ["query", "shop", "How many, and what does Q1 say?"])

    assert result.exit_code == 0, result.output
    assert "Three orders, as the Q1 report also says." in result.output
    assert "[1] Q1 [final].pdf, page 4" in result.output


def test_the_mcp_tool_offers_the_agents_answer_types(connectors_file: Path) -> None:
    from nlqueries.mcp_server.server import query

    seen: dict[str, Any] = {}

    async def run(question: str, agent_id: str, **kwargs: Any) -> AgentQueryResult:
        seen.update(kwargs)
        return _result("document", answer="From the wiki.")

    with patch("nlqueries.orchestrator.sync_runner.run_query", run):
        text = asyncio.run(query("What does the wiki say?", "postgres_localhost_shop"))

    assert "From the wiki." in text
    assert seen["available_types"] == ("sql", "document", "hybrid")


def test_the_mcp_tool_keeps_an_agent_without_sources_to_its_database(
    connectors_file: Path,
) -> None:
    from nlqueries.mcp_server.server import query

    seen: dict[str, Any] = {}

    async def run(question: str, agent_id: str, **kwargs: Any) -> AgentQueryResult:
        seen.update(kwargs)
        return _result("sql", sql="SELECT 1")

    with patch("nlqueries.orchestrator.sync_runner.run_query", run):
        asyncio.run(query("How many?", "postgres_localhost_plain"))

    assert seen["available_types"] == ("sql",)
