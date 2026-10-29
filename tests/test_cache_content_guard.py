"""The semantic cache serves a hit only when it asks for what the question asks.

Similarity alone could not tell "Which five customers have spent the most, and
in which region are they?" from "... how much, and in which region?": the two
score 0.985 with the cache's embedding model, above most genuine rephrasings,
and the second was answered with the first's SQL, without the amount. These
pin the check (`_asks_the_same`) and its place in both tiers of `get()`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.cache.envelope import sign
from nlqueries.cache.semantic_cache import SemanticCache, _asks_the_same, _question_content

from tests.test_semantic_cache import (
    TEST_BINDING,
    TEST_KEY,
    _fresh_payload,
    _make_query_response,
    _make_scored_point,
)


@pytest.fixture(autouse=True)
def _use_the_test_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: TEST_KEY)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cached,incoming",
    [
        # The pair that was served.
        (
            "Which five customers have spent the most, and in which region are they?",
            "Which five customers have spent the most, how much, and in which region?",
        ),
        ("Top 5 customers by spend", "Top 10 customers by spend"),
        ("Top five customers by spend", "Top ten customers by spend"),
        ("List the products out of stock", "List the products not out of stock"),
        ("Customers who spent the most", "Customers who spent the least"),
        ("Revenue by region", "Revenue and margin by region"),
        ("Orders shipped to Germany", "Orders shipped to France"),
        ("Orders placed last month", "Orders placed this month"),
    ],
)
def test_a_question_that_asks_for_something_else_is_not_the_same(
    cached: str, incoming: str
) -> None:
    assert not _asks_the_same(cached, incoming)


@pytest.mark.parametrize(
    "cached,incoming",
    [
        (
            "Which five customers have spent the most, and in which region are they?",
            "Which five customers have spent the most, and which region are they in?",
        ),
        ("Which 5 customers spent the most", "Which 5 customers have spent the most"),
        ("What is the average order value?", "What's the average order value?"),
        ("List the products out of stock", "List all products out of stock"),
        ("Total revenue by region", "Total revenue per region"),
        ("Show the orders", "Show me the order"),
        ("How many orders were there?", "Roughly how many orders were there?"),
    ],
)
def test_a_rephrasing_is_the_same(cached: str, incoming: str) -> None:
    assert _asks_the_same(cached, incoming)


def test_numbers_and_their_words_are_one_value() -> None:
    assert _question_content("top 5 customers") == _question_content("top five customers")


def test_how_much_is_a_request_of_its_own() -> None:
    assert "AMOUNT" in _question_content("how much did they spend")
    assert "AMOUNT" not in _question_content("how many did they buy")


def test_an_entry_without_its_question_is_never_the_same() -> None:
    assert not _asks_the_same(None, "How many orders?")
    assert not _asks_the_same("", "How many orders?")


# ---------------------------------------------------------------------------
# In get()
# ---------------------------------------------------------------------------


def _client(*responses: Any) -> MagicMock:
    coll = MagicMock()
    coll.name = "cache_agent1"
    client = MagicMock()
    client.get_collections.return_value.collections = [coll]
    client.retrieve.return_value = []  # Tier 0 miss
    client.query_points.side_effect = [_make_query_response(list(r)) for r in responses]
    return client


def _get(client: MagicMock, question: str) -> Any:
    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        return SemanticCache("agent1", binding=TEST_BINDING).get(question)


ASKED = "Which five customers have spent the most, how much, and in which region?"


def test_tier_1_skips_a_near_neighbour_that_asks_for_less() -> None:
    """The nearest entry asks for less than the question; a rephrasing of the
    question itself ranks below it and is served instead."""
    narrower = _make_scored_point(
        0.985,
        _fresh_payload(
            question="Which five customers have spent the most, and in which region are they?",
            answer="name and region",
        ),
        point_id=1,
    )
    same = _make_scored_point(
        0.98,
        _fresh_payload(
            question="Which 5 customers have spent the most, how much, and in which region?",
            answer="name, region and total",
        ),
        point_id=2,
    )

    entry = _get(_client([narrower, same]), ASKED)

    assert entry is not None and entry.answer == "name, region and total"


def test_tier_1_misses_when_only_the_narrower_entry_is_there() -> None:
    narrower = _make_scored_point(
        0.985,
        _fresh_payload(
            question="Which five customers have spent the most, and in which region are they?"
        ),
    )
    # Tier 1 misses; the question has no entity Tier 2 could mask ("five" is a word).
    assert _get(_client([narrower]), ASKED) is None


def test_the_guard_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CACHE_CONTENT_GUARD", False)
    narrower = _make_scored_point(
        0.985,
        _fresh_payload(
            question="Which five customers have spent the most, and in which region are they?",
            answer="name and region",
        ),
    )
    entry = _get(_client([narrower]), ASKED)
    assert entry is not None and entry.answer == "name and region"


def _template(question: str, sql: str) -> dict[str, Any]:
    return sign(
        {
            "question": question,
            "resolved_question": question,
            "agent_type": "sql",
            "answer": "Found results.",
            "sql": sql,
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "template",
        },
        TEST_BINDING,
        TEST_KEY,
    )


def test_tier_2_still_rebinds_a_different_value() -> None:
    """Compared on the masked forms: a different date is what Tier 2 is for."""
    tmpl = _make_scored_point(
        0.95,
        _template("orders after <DATE>", "SELECT * FROM orders WHERE d >= '[d:DATE]'"),
        point_id=9,
    )
    entry = _get(_client([], [tmpl]), "orders after 2024-06-01")
    assert entry is not None and entry.sql == "SELECT * FROM orders WHERE d >= '2024-06-01'"


def test_tier_2_skips_a_template_that_asks_for_something_else() -> None:
    """Masked, "shipped orders after <DATE>" and "orders after <DATE>" still
    differ in what they ask for, so the template is not bound to the question."""
    tmpl = _make_scored_point(
        0.95,
        _template("orders after <DATE>", "SELECT * FROM orders WHERE d >= '[d:DATE]'"),
    )
    assert _get(_client([], [tmpl]), "shipped orders after 2024-06-01") is None
