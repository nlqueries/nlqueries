"""Every store keyed by agent uses one name for it: ``agent_storage_id``.

The knowledge base, the feedback and session files, and the cache, document and
schema collections each derived the name themselves, and not all the same way:
the cache turned ``.`` into ``_`` as well, so ``sales.eu`` and ``sales_eu``
shared one cache. Document and schema collections used the raw ID, so the CLI
(``postgres:localhost:mydb``) and the MCP server (``postgres_localhost_mydb``,
from the knowledge-base file name) wrote and read different collections for the
same agent.
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from nlqueries import config
from nlqueries.cache.semantic_cache import SemanticCache, cache_collection_name
from nlqueries.embeddings import qdrant_store
from nlqueries.state_files import agent_storage_id


@pytest.fixture(autouse=True)
def _no_remembered_resolutions():
    qdrant_store._resolved.clear()
    yield
    qdrant_store._resolved.clear()


def _client_with(*names: str) -> Any:
    """A Qdrant client stand-in that holds collections called *names*."""
    collections = [SimpleNamespace(name=n) for n in names]
    calls: list[str] = []

    def get_collections() -> Any:
        calls.append("get_collections")
        return SimpleNamespace(collections=collections)

    return SimpleNamespace(get_collections=get_collections, calls=calls)


class TestTheName:
    @pytest.mark.parametrize(
        ("agent_id", "expected"),
        [
            ("sales", "sales"),
            ("sales.eu", "sales.eu"),
            ("sales-eu_2", "sales-eu_2"),
            ("postgres:localhost:mydb", "postgres_localhost_mydb"),
            ("a/b c", "a_b_c"),
        ],
    )
    def test_it_keeps_letters_digits_dots_dashes_and_underscores(
        self, agent_id: str, expected: str
    ) -> None:
        assert agent_storage_id(agent_id) == expected

    def test_a_connector_id_and_its_sanitised_form_name_one_agent(self) -> None:
        # Deliberate: the CLI uses the connector ID, the MCP server the
        # knowledge-base file name, and both must reach the same agent.
        assert agent_storage_id("postgres:localhost:mydb") == agent_storage_id(
            "postgres_localhost_mydb"
        )


class TestTheCache:
    def test_a_dotted_id_and_its_underscore_twin_get_different_collections(self) -> None:
        assert cache_collection_name("sales.eu") != cache_collection_name("sales_eu")

    def test_the_cache_reads_and_writes_that_collection(self) -> None:
        assert SemanticCache("sales.eu")._collection == "cache_sales.eu"
        assert SemanticCache("sales_eu")._collection == "cache_sales_eu"

    def test_kb_stats_counts_the_collection_the_cache_writes(self, tmp_path) -> None:
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb_file = tmp_path / "kb.yaml"
        kb_file.write_text(yaml.safe_dump({"schema": {"tables": []}}), encoding="utf-8")
        urls: list[str] = []

        def urlopen(url: str, timeout: float) -> Any:
            urls.append(url)
            return io.BytesIO(json.dumps({"result": {"vectors_count": 3}}).encode())

        with patch("urllib.request.urlopen", urlopen):
            stats = compute_kb_stats("pg:h:sales.eu", kb_file)

        assert urls == [f"{config.QDRANT_URL}/collections/cache_pg_h_sales.eu"]
        assert stats.cache_entries == 3


class TestTheKnowledgeBase:
    @pytest.mark.parametrize(
        ("agent_id", "file_name"),
        [
            ("postgres:localhost:mydb", "postgres_localhost_mydb.yaml"),
            ("postgres_localhost_mydb", "postgres_localhost_mydb.yaml"),
            ("sales.eu", "sales.eu.yaml"),
        ],
    )
    def test_an_existing_knowledge_base_is_still_loaded(
        self, agent_id: str, file_name: str
    ) -> None:
        from nlqueries.orchestrator.orchestrator import Orchestrator

        config.KB_PATH.mkdir(parents=True, exist_ok=True)
        (config.KB_PATH / file_name).write_text(
            yaml.safe_dump({"meta": {"marker": file_name}}), encoding="utf-8"
        )

        kb = Orchestrator()._load_knowledge_base(agent_id)

        assert kb["meta"]["marker"] == file_name


class TestCollectionsToRead:
    def test_an_id_its_name_keeps_whole_never_asks_qdrant(self) -> None:
        # Records rather than raises: the helper treats any error from Qdrant
        # as out of reach, so a raising stand-in would be swallowed.
        client = _client_with("doc_sales.eu_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            name = qdrant_store.agent_collection_to_read("doc_", "sales.eu", "_chunks")
        assert name == "doc_sales.eu_chunks"
        assert client.calls == []

    def test_the_new_name_is_read_when_it_exists(self) -> None:
        client = _client_with("doc_pg_h_db_chunks", "doc_pg:h:db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            name = qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
        assert name == "doc_pg_h_db_chunks"

    def test_the_old_raw_name_is_read_when_only_it_exists(self) -> None:
        client = _client_with("doc_pg:h:db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            name = qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
        assert name == "doc_pg:h:db_chunks"

    def test_the_new_name_when_neither_exists(self) -> None:
        client = _client_with()
        with patch.object(qdrant_store, "_get_client", return_value=client):
            name = qdrant_store.agent_collection_to_read("agent_", "pg:h:db", "_schema")
        assert name == "agent_pg_h_db_schema"

    def test_the_new_name_when_qdrant_is_out_of_reach(self) -> None:
        def unreachable() -> Any:
            raise ConnectionError("refused")

        with patch.object(qdrant_store, "_get_client", unreachable):
            name = qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
        assert name == "doc_pg_h_db_chunks"

    def test_the_choice_is_reused_rather_than_asked_per_question(self) -> None:
        client = _client_with("doc_pg:h:db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            first = qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
            second = qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
        assert first == second == "doc_pg:h:db_chunks"
        assert client.calls == ["get_collections"]

    def test_the_choice_is_asked_again_once_it_is_old(self) -> None:
        client = _client_with("doc_pg:h:db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
            name, (stamp, chosen) = next(iter(qdrant_store._resolved.items()))
            qdrant_store._resolved[name] = (stamp - qdrant_store._RESOLVED_TTL_SECONDS - 1, chosen)
            qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks")
        assert client.calls == ["get_collections", "get_collections"]


class TestWhereTheNamesAreUsed:
    def test_a_document_agent_reads_its_storage_name(self) -> None:
        from nlqueries.orchestrator import multi_agent_orchestrator as mao

        seen: list[str] = []

        async def handle_question(self: Any, question: str, collection: str) -> Any:
            seen.append(collection)
            if False:  # pragma: no cover - makes this an async generator
                yield ""

        client = _client_with("doc_pg_h_db_chunks")
        with (
            patch.object(mao.DocumentOrchestrator, "handle_question", handle_question),
            patch.object(qdrant_store, "_get_client", return_value=client),
        ):
            asyncio.run(mao._run_document("q", "pg:h:db"))

        assert seen == ["doc_pg_h_db_chunks"]

    def test_the_sql_agent_reads_its_schema_collection_by_storage_name(self) -> None:
        from nlqueries.orchestrator import orchestrator as orch_module

        seen: list[str] = []

        class _Stop(Exception):
            pass

        async def assemble(*args: Any, collection: str, **kwargs: Any) -> Any:
            seen.append(collection)
            raise _Stop

        async def drain() -> None:
            orchestrator = orch_module.Orchestrator()
            async for _ in orchestrator.handle_question("q", "pg:h:db", dialect="postgres"):
                pass

        client = _client_with("agent_pg_h_db_schema")
        with (
            patch.object(orch_module.Orchestrator, "_load_knowledge_base", return_value={}),
            patch.object(orch_module, "assemble_prompt_async", assemble),
            patch.object(qdrant_store, "_get_client", return_value=client),
            pytest.raises(_Stop),
        ):
            asyncio.run(drain())

        assert seen == ["agent_pg_h_db_schema"]

    def test_doc_ingest_writes_the_storage_name(self) -> None:
        from nlqueries.cli import main as cli_main

        written: list[str] = []

        def replace_chunks(collection: str, source_id: str, chunks: list[Any]) -> None:
            written.append(collection)

        with patch.object(qdrant_store, "replace_chunks", replace_chunks):
            cli_main._store_document_chunks("pg:h:db", ["chunk"], allow_empty=False)

        assert written == ["doc_pg_h_db_chunks"]

    def test_the_schema_is_embedded_under_the_storage_name(self) -> None:
        from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
        from nlqueries.knowledge.kb_generator import generate_knowledge_base

        created: list[str] = []
        upserted: list[str] = []
        column = ColumnSpec("id", "INT", False, True, False, None, None)
        schema = SchemaSpec(
            database="shop",
            tables=[TableSpec("orders", "public", None, [column], None)],
            extracted_at="2026-10-22T00:00:00Z",
        )
        with (
            patch.object(
                qdrant_store, "ensure_collection", lambda name, *a, **k: created.append(name)
            ),
            patch.object(
                qdrant_store, "upsert_schema", lambda name, *a, **k: upserted.append(name)
            ),
        ):
            generate_knowledge_base(schema, [], "pg:h:db", embed=True)

        assert created == upserted == ["agent_pg_h_db_schema"]
