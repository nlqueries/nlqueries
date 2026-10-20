"""Promoted feedback reaches the prompt for an ordinary connector ID.

`promote-feedback` (and `export-kb`, which runs it) writes confirmed pairs to
``agent_<safe id>_verified``, with every character outside ``[A-Za-z0-9_-]``
replaced by ``_``. The question path derived the name from the raw connector
ID instead, so for ``postgres:localhost:mydb`` -- the shape every `connect`
gives -- it looked for ``agent_postgres:localhost:mydb_verified``, found
nothing, and the promoted examples were never used.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from nlqueries import config
from nlqueries.feedback.promoter import _verified_collection
from nlqueries.orchestrator.prompt_assembly import _search_verified
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_AGENT_ID = "postgres:localhost:mydb"


def _unit_vector() -> list[float]:
    vec = [0.0] * config.EMBED_DIMENSIONS
    vec[0] = 1.0
    return vec


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    # Named exactly as promote-feedback names it.
    collection = _verified_collection(_AGENT_ID)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[
            PointStruct(
                id=1,
                vector=_unit_vector(),
                payload={
                    "question": "How many orders last month?",
                    "sql": "SELECT COUNT(*) FROM orders",
                    "agent_id": _AGENT_ID,
                    "verified": True,
                },
            )
        ],
    )
    with patch("qdrant_client.QdrantClient", return_value=client):
        yield client
    client.close()


def test_a_promoted_pair_is_found_for_a_connector_id_with_colons(qdrant: QdrantClient) -> None:
    # The collection name the orchestrator passes in, built from the raw ID.
    hits = _search_verified(
        f"agent_{_AGENT_ID}_schema", "How many orders last month?", vector=_unit_vector()
    )

    assert [h["sql"] for h in hits] == ["SELECT COUNT(*) FROM orders"]


def test_an_agent_whose_id_sanitises_the_same_does_not_get_anothers_pairs(
    qdrant: QdrantClient,
) -> None:
    """`postgres:db.prod:sales` and `postgres:db_prod:sales` both become
    `agent_postgres_db_prod_sales_verified`, so their pairs share a collection
    and only the `agent_id` the promoter writes into each payload tells them
    apart."""
    first, second = "postgres:db.prod:sales", "postgres:db_prod:sales"
    collection = _verified_collection(first)
    assert collection == _verified_collection(second)
    qdrant.create_collection(
        collection,
        vectors_config=VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE),
    )
    qdrant.upsert(
        collection,
        points=[
            PointStruct(
                id=10 + n,
                vector=_unit_vector(),
                payload={"question": "q", "sql": sql, "agent_id": agent, "verified": True},
            )
            for n, (agent, sql) in enumerate(
                [(first, "SELECT 1 FROM prod_sales"), (second, "SELECT 1 FROM other_sales")]
            )
        ],
    )

    hits = _search_verified(f"agent_{first}_schema", "q", vector=_unit_vector())

    assert [h["sql"] for h in hits] == ["SELECT 1 FROM prod_sales"]
