"""The schema collection is named with the agent's storage name, like every other store.

It used the raw agent ID, so for a connector ID with `:` the CLI
(`postgres:localhost:mydb`) and the MCP server (`postgres_localhost_mydb`, from
the knowledge-base file name) read different collections, and capsules embedded
from the CLI helped only the CLI's questions. Verified pairs were matched on the
raw ID too, so the MCP server never saw pairs promoted from the CLI. An ID and
its sanitised form are now one agent throughout. A collection an earlier
release wrote under the raw ID is still read, for each kind of point, until the
new name holds that kind.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.embeddings import qdrant_store
from nlqueries.orchestrator import prompt_assembly
from nlqueries.processing.parameterizer import QueryCapsule

_VECTOR = [0.1] * 384
_CLI_ID, _MCP_ID = "pg:h:db", "pg_h_db"


class _Hit:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.score = 0.9


class _Store:
    """Collections of points in memory, with what the code asks of Qdrant."""

    def __init__(self) -> None:
        self.points: dict[str, dict[int, dict[str, Any]]] = {}
        self.deleted: list[str] = []
        self.asked = 0

    def add(self, collection: str, *payloads: dict[str, Any]) -> None:
        stored = self.points.setdefault(collection, {})
        for payload in payloads:
            stored[len(stored) + 1000] = payload

    def get_collections(self) -> Any:
        self.asked += 1
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.points])

    def count(self, collection_name: str, count_filter: Any, exact: bool) -> Any:
        (condition,) = count_filter.must
        n = sum(
            p.get("type") in condition.match.any
            for p in self.points.get(collection_name, {}).values()
        )
        return SimpleNamespace(count=n)

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        stored = self.points.setdefault(collection_name, {})
        for point in points:
            stored[point.id] = dict(point.payload)

    def query_points(
        self, collection_name: str, query: Any, query_filter: Any = None, limit: int = 10
    ) -> Any:
        must = query_filter.must if query_filter is not None else []
        found = [
            _Hit(p)
            for p in self.points.get(collection_name, {}).values()
            if all(p.get(c.key) == c.match.value for c in must)
        ]
        return MagicMock(points=found[:limit])

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.points.pop(name, None)


@pytest.fixture()
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    fake = _Store()
    monkeypatch.setattr(qdrant_store, "_client", fake)
    monkeypatch.setattr(
        "nlqueries.embeddings.embedder.embed_batch", lambda texts: [_VECTOR for _ in texts]
    )
    qdrant_store._resolved.clear()
    qdrant_store._UNSIGNED_CAPSULES_WARNED.clear()
    qdrant_store._REPOINTED_CAPSULES_WARNED.clear()
    return fake


_TABLE = {"type": "table", "table_name": "orders"}


def _capsule(intent: str) -> QueryCapsule:
    return QueryCapsule(
        template_sql=f"SELECT 1 -- {intent}",
        placeholders=[],
        tables=["orders"],
        columns=[],
        frequency=3,
        auto_description="",
        intent=intent,
    )


# ---------------------------------------------------------------------------
# The name, and reading it during the changeover
# ---------------------------------------------------------------------------


def test_both_forms_of_an_id_name_one_collection() -> None:
    assert qdrant_store.schema_collection(_CLI_ID) == "agent_pg_h_db_schema"
    assert qdrant_store.schema_collection(_MCP_ID) == "agent_pg_h_db_schema"


def test_an_id_the_storage_name_keeps_needs_no_lookup(store: _Store) -> None:
    assert qdrant_store.schema_collection_to_read("sales", ("table",)) == "agent_sales_schema"
    assert store.asked == 0


def test_tables_are_read_from_the_raw_id_collection_until_rewritten(store: _Store) -> None:
    """Capsules moved to the new name with `process-history --embed`; tables
    move only when the knowledge base is embedded again. Each is read from
    where it is."""
    store.add("agent_pg:h:db_schema", _TABLE)
    store.add("agent_pg_h_db_schema", {"type": "capsule"})

    tables = qdrant_store.schema_collection_to_read(_CLI_ID, qdrant_store.SCHEMA_POINT_TYPES)
    capsules = qdrant_store.schema_collection_to_read(_CLI_ID, (qdrant_store.CAPSULE_TYPE,))

    assert tables == "agent_pg:h:db_schema"
    assert capsules == "agent_pg_h_db_schema"


def test_the_new_name_is_read_once_it_holds_the_kind(store: _Store) -> None:
    store.add("agent_pg:h:db_schema", _TABLE)
    store.add("agent_pg_h_db_schema", _TABLE)

    found = qdrant_store.schema_collection_to_read(_CLI_ID, qdrant_store.SCHEMA_POINT_TYPES)

    assert found == "agent_pg_h_db_schema"


def test_qdrant_out_of_reach_gives_the_new_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def down() -> Any:
        raise ConnectionError("refused")

    monkeypatch.setattr(qdrant_store, "_get_client", down)
    qdrant_store._resolved.clear()

    assert qdrant_store.schema_collection_to_read(_CLI_ID, ("table",)) == "agent_pg_h_db_schema"


# ---------------------------------------------------------------------------
# The generator writes the new name and removes the old one
# ---------------------------------------------------------------------------


def test_rewriting_the_tables_retires_the_raw_id_collection(store: _Store) -> None:
    store.add("agent_pg:h:db_schema", _TABLE)
    qdrant_store.schema_collection_to_read(_CLI_ID, qdrant_store.SCHEMA_POINT_TYPES)

    assert qdrant_store.retire_legacy_schema_collection(_CLI_ID) == "agent_pg:h:db_schema"
    assert store.deleted == ["agent_pg:h:db_schema"]
    # The answer kept for a minute named the deleted collection; it is dropped.
    assert not [k for k in qdrant_store._resolved if k.startswith("agent_pg_h_db_schema|")]


def test_the_generator_writes_the_storage_name_then_retires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
    from nlqueries.knowledge import kb_generator

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(qdrant_store, "ensure_collection", lambda name, *a: None)
    monkeypatch.setattr(
        qdrant_store, "upsert_schema", lambda name, schema, agent_id: calls.append(("upsert", name))
    )
    monkeypatch.setattr(
        qdrant_store,
        "retire_legacy_schema_collection",
        lambda agent: calls.append(("retire", agent)),
    )
    column = ColumnSpec("id", "integer", False, True, False, None, None)
    table = TableSpec("orders", "public", 10, [column], None)
    schema = SchemaSpec("db", [table], "2026-10-25T00:00:00Z")

    kb_generator.generate_knowledge_base(schema, [], agent_name=_CLI_ID, embed=True)

    assert calls == [("upsert", "agent_pg_h_db_schema"), ("retire", _CLI_ID)]


# ---------------------------------------------------------------------------
# One agent, whichever form of its ID asks
# ---------------------------------------------------------------------------


def _examples(agent_id: str) -> str:
    with (
        patch.object(qdrant_store, "search_schema", return_value=[]),
        patch.object(prompt_assembly, "_search_verified", return_value=[]),
    ):
        return prompt_assembly._build_dynamic_context(
            "orders",
            {},
            5,
            qdrant_store.schema_collection(agent_id),
            _VECTOR,
            agent_id,
        )


@pytest.mark.parametrize(("embedded_as", "asked_as"), [(_CLI_ID, _MCP_ID), (_MCP_ID, _CLI_ID)])
def test_a_capsule_embedded_under_one_form_reaches_the_other(
    store: _Store, embedded_as: str, asked_as: str
) -> None:
    qdrant_store.upsert_capsules(
        qdrant_store.schema_collection(embedded_as), [_capsule("monthly orders")], embedded_as
    )

    assert "Intent: monthly orders" in _examples(asked_as)


def test_a_capsule_of_an_agent_with_another_storage_name_is_not_shown(store: _Store) -> None:
    qdrant_store.upsert_capsules(
        qdrant_store.schema_collection(_CLI_ID), [_capsule("someone else's")], "billing"
    )

    assert "someone else's" not in _examples(_CLI_ID)


def test_a_caller_passing_only_a_collection_still_gets_its_capsules(store: _Store) -> None:
    """Library callers that pass `collection=` alone, as before."""
    collection = qdrant_store.schema_collection("sales")
    qdrant_store.upsert_capsules(collection, [_capsule("by region")], "sales")

    with (
        patch.object(qdrant_store, "search_schema", return_value=[]),
        patch.object(prompt_assembly, "_search_verified", return_value=[]),
    ):
        text = prompt_assembly._build_dynamic_context("q", {}, 5, collection, _VECTOR)

    assert "Intent: by region" in text


def test_the_question_path_reads_each_kind_from_where_it_is(store: _Store) -> None:
    """Through prompt assembly, as the orchestrator calls it: the tables hint
    from the raw-ID collection that still holds the tables, the capsules from
    the new name that holds them."""
    store.add("agent_pg:h:db_schema", _TABLE)
    qdrant_store.upsert_capsules(
        qdrant_store.schema_collection(_CLI_ID), [_capsule("monthly orders")], _CLI_ID
    )
    searched: list[str] = []

    def search_schema(collection: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        searched.append(collection)
        return [{"table_name": "orders"}]

    with (
        patch.object(qdrant_store, "search_schema", search_schema),
        patch.object(prompt_assembly, "_search_verified", return_value=[]),
    ):
        text = prompt_assembly._build_dynamic_context(
            "q", {}, 5, qdrant_store.schema_collection(_CLI_ID), _VECTOR, _CLI_ID
        )

    assert searched == ["agent_pg:h:db_schema"]
    assert "orders" in text and "Intent: monthly orders" in text
