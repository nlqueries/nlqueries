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


def _client_with(*names: str, empty: tuple[str, ...] = ()) -> Any:
    """A Qdrant client stand-in holding collections called *names*.

    Each holds one point, except those named in *empty*.
    """
    held = list(names)
    calls: list[str] = []

    def get_collections() -> Any:
        calls.append("get_collections")
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in held])

    def count(collection_name: str, exact: bool = True) -> Any:
        return SimpleNamespace(count=0 if collection_name in empty else 1)

    def delete_collection(name: str) -> None:
        calls.append(f"delete {name}")
        held.remove(name)

    return SimpleNamespace(
        get_collections=get_collections,
        count=count,
        delete_collection=delete_collection,
        calls=calls,
        held=held,
    )


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
        self, agent_id: str, file_name: str, tmp_path, monkeypatch
    ) -> None:
        from nlqueries.orchestrator.orchestrator import Orchestrator

        # A directory of its own: the shared KB_PATH is what the MCP server
        # lists agents from, so files left there show up in later tests.
        monkeypatch.setattr(config, "KB_PATH", tmp_path)
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

    def test_the_old_name_is_read_while_the_new_one_holds_nothing(self) -> None:
        """An ingest that failed before its first batch still created the new
        collection; an empty one must not hide the copy that has the text."""
        client = _client_with(
            "doc_pg_h_db_chunks", "doc_pg:h:db_chunks", empty=("doc_pg_h_db_chunks",)
        )
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


class TestRetiringTheOldCollection:
    def test_the_raw_id_collection_is_deleted(self) -> None:
        client = _client_with("doc_pg_h_db_chunks", "doc_pg:h:db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            retired = qdrant_store.retire_legacy_collection("doc_", "pg:h:db", "_chunks")
        assert retired == "doc_pg:h:db_chunks"
        assert client.held == ["doc_pg_h_db_chunks"]

    def test_nothing_is_deleted_when_the_name_is_unchanged(self) -> None:
        client = _client_with("doc_sales.eu_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            assert qdrant_store.retire_legacy_collection("doc_", "sales.eu", "_chunks") is None
        assert client.held == ["doc_sales.eu_chunks"]
        assert client.calls == []

    def test_nothing_is_deleted_when_there_is_no_old_collection(self) -> None:
        client = _client_with("doc_pg_h_db_chunks")
        with patch.object(qdrant_store, "_get_client", return_value=client):
            assert qdrant_store.retire_legacy_collection("doc_", "pg:h:db", "_chunks") is None
        assert client.held == ["doc_pg_h_db_chunks"]

    def test_a_reader_stops_reading_it_at_once(self) -> None:
        client = _client_with(
            "doc_pg_h_db_chunks", "doc_pg:h:db_chunks", empty=("doc_pg_h_db_chunks",)
        )
        with patch.object(qdrant_store, "_get_client", return_value=client):
            assert qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks") == (
                "doc_pg:h:db_chunks"
            )
            qdrant_store.retire_legacy_collection("doc_", "pg:h:db", "_chunks")
            assert qdrant_store.agent_collection_to_read("doc_", "pg:h:db", "_chunks") == (
                "doc_pg_h_db_chunks"
            )


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

    def test_the_sql_agent_keeps_the_raw_id_in_its_schema_collection(self) -> None:
        """Deliberately unchanged: `_search_verified` takes the agent ID back out
        of this name and filters the verified examples on it, and
        `promote-feedback` stores them under the ID as given. A storage name
        here would hand it `pg_h_db` and match no promoted example."""
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

        with (
            patch.object(orch_module.Orchestrator, "_load_knowledge_base", return_value={}),
            patch.object(orch_module, "assemble_prompt_async", assemble),
            pytest.raises(_Stop),
        ):
            asyncio.run(drain())

        assert seen == ["agent_pg:h:db_schema"]

    def test_the_hybrid_merge_records_the_collection_read_without_asking_again(self) -> None:
        """`_merge_hybrid` runs on the event loop; the collection `_run_hybrid`
        resolved is passed in rather than resolved again with a Qdrant call."""
        from nlqueries.orchestrator import multi_agent_orchestrator as mao
        from nlqueries.orchestrator.document_retrieval import Citation

        client = _client_with("doc_pg:h:db_chunks")
        citation = Citation(
            chunk_id="c",
            source_name="Doc",
            page_number=None,
            chunk_index=0,
            excerpt="x",
            relevance_score=1.0,
        )
        passed: list[Any] = []

        def merge_results(question: str, **kwargs: Any) -> Any:
            passed.append(kwargs["document_result"])
            return "merged"

        with (
            patch.object(qdrant_store, "_get_client", return_value=client),
            patch.object(mao, "merge_results", merge_results),
        ):
            mao._merge_hybrid("q", "pg:h:db", [], [citation], "doc_pg:h:db_chunks")

        assert client.calls == []
        (document_result,) = passed
        assert document_result.collection == "doc_pg:h:db_chunks"

    def test_the_hybrid_run_returns_the_collection_it_read(self) -> None:
        from nlqueries.orchestrator import multi_agent_orchestrator as mao

        async def no_tokens(*args: Any, **kwargs: Any) -> Any:
            if False:  # pragma: no cover - makes this an async generator
                yield ""

        client = _client_with("doc_pg:h:db_chunks")
        with (
            patch.object(mao.Orchestrator, "handle_question", no_tokens),
            patch.object(mao.DocumentOrchestrator, "handle_question", no_tokens),
            patch.object(qdrant_store, "_get_client", return_value=client),
        ):
            *_, collection = asyncio.run(mao._run_hybrid("q", "pg:h:db", "postgres"))

        assert collection == "doc_pg:h:db_chunks"

    @pytest.mark.parametrize(("chunks", "allow_empty"), [(["chunk"], False), ([], True)])
    def test_doc_ingest_writes_the_storage_name_and_removes_the_old_copy(
        self, chunks: list[str], allow_empty: bool
    ) -> None:
        """Including a deliberate --allow-empty removal, which would otherwise
        leave the source's text in the collection an earlier release wrote."""
        from nlqueries.cli import main as cli_main

        written: list[str] = []

        def replace_chunks(collection: str, source_id: str, chunks: list[Any]) -> None:
            written.append(collection)

        client = _client_with("doc_pg_h_db_chunks", "doc_pg:h:db_chunks")
        with (
            patch.object(qdrant_store, "replace_chunks", replace_chunks),
            patch.object(qdrant_store, "_get_client", return_value=client),
        ):
            cli_main._store_document_chunks("pg:h:db", chunks, allow_empty=allow_empty)

        assert written == ["doc_pg_h_db_chunks"]
        assert client.held == ["doc_pg_h_db_chunks"]

    def test_doc_ingest_fails_when_the_old_copy_cannot_be_removed(self) -> None:
        from nlqueries.cli import main as cli_main

        def unreachable() -> Any:
            raise ConnectionError("refused")

        with (
            patch.object(qdrant_store, "replace_chunks", lambda *a: None),
            patch.object(qdrant_store, "_get_client", unreachable),
            pytest.raises(SystemExit) as exited,
        ):
            cli_main._store_document_chunks("pg:h:db", ["chunk"], allow_empty=False)

        assert exited.value.code == 1

    def test_doc_ingest_keeps_the_old_copy_when_the_store_fails(self) -> None:
        from nlqueries.cli import main as cli_main

        def replace_chunks(*args: Any) -> None:
            raise ConnectionError("embedder down")

        client = _client_with("doc_pg:h:db_chunks")
        with (
            patch.object(qdrant_store, "replace_chunks", replace_chunks),
            patch.object(qdrant_store, "_get_client", return_value=client),
            pytest.raises(SystemExit),
        ):
            cli_main._store_document_chunks("pg:h:db", ["chunk"], allow_empty=False)

        assert client.held == ["doc_pg:h:db_chunks"]
