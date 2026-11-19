"""
tests.test_prompt_values
~~~~~~~~~~~~~~~~~~~~~~~~
The value index's values in the prompt (F20, part 2d): ``LookupSource.
prompt_values`` and how prompt assembly shows what it returns.

Core supplies no provider. Where one is given and the request's
``value_index`` setting is on, a short column's every value goes in the
schema (the cached part) in place of its samples, and a longer column's values
matching the question go in the per-question part. A provider that fails, is
slow or answers with something else leaves the prompt as it would be without.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from nlqueries.connectors import loader
from nlqueries.connectors.loader import LookupSource, PromptValues
from nlqueries.orchestrator import prompt_values as pv
from nlqueries.orchestrator.prompt_assembly import assemble_prompt, assemble_prompt_async
from nlqueries.query_settings import QuerySettings, use_query_settings

KB: dict[str, Any] = {
    "db_name": "shop",
    "schema": {
        "tables": [
            {
                "name": "stores",
                "schema": "shop",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_primary_key": True},
                    {"name": "region", "type": "TEXT", "samples": ["EU", "US"]},
                    {"name": "city", "type": "TEXT", "samples": ["Paris"]},
                ],
            }
        ]
    },
}

REGION = ("shop", "stores", "region")
CITY = ("shop", "stores", "city")


@pytest.fixture
def provide(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Any], list[str]]]:
    """Give the agent a lookup source whose prompt_values is *fn*; return the
    questions it was asked."""
    asked: list[str] = []

    def install(fn: Any) -> list[str]:
        def counted(question: str) -> Any:
            asked.append(question)
            return fn(question)

        monkeypatch.setattr(
            loader,
            "lookup_source",
            lambda _agent: LookupSource(key="k", open=lambda: None, prompt_values=counted),
        )
        return asked

    yield install


def _on() -> Any:
    return use_query_settings(QuerySettings(value_index=True))


def _prompt(question: str = "stores in dallas fort worth") -> Any:
    return assemble_prompt(question, KB, agent_id="agent-1")


def test_with_the_setting_off_the_provider_is_not_asked(provide: Any) -> None:
    asked = provide(lambda _q: PromptValues(lists={REGION: ("EU", "US", "APAC")}))
    with use_query_settings(QuerySettings(value_index=False)):
        prompt = _prompt()
    assert asked == []
    assert "APAC" not in prompt.static_system


def test_a_short_column_s_list_replaces_its_samples_in_the_schema(provide: Any) -> None:
    provide(lambda _q: PromptValues(lists={REGION: ("EU", "US", "APAC")}))
    with _on():
        prompt = _prompt()
    assert "region:TEXT, values: ['EU', 'US', 'APAC']" in prompt.static_system
    assert "samples: ['EU', 'US']" not in prompt.static_system
    # A column the index does not list keeps its samples.
    assert "city:TEXT, samples: ['Paris']" in prompt.static_system


def test_the_schema_with_its_lists_is_the_same_for_every_question(provide: Any) -> None:
    provide(lambda _q: PromptValues(lists={REGION: ("EU", "US")}))
    with _on():
        first, second = _prompt("stores in paris"), _prompt("how many stores")
    assert first.static_system == second.static_system


def test_a_longer_column_s_matches_go_in_the_per_question_part(provide: Any) -> None:
    provide(
        lambda _q: PromptValues(
            matching={CITY: ("Dallas", "Fort Worth"), ("shop", "stores", "nope"): ("X",)}
        )
    )
    with _on():
        prompt = _prompt()
    assert "【Stored values matching the question】" in prompt.dynamic_context
    assert "shop.stores.city: 'Dallas', 'Fort Worth'" in prompt.dynamic_context
    # A column the knowledge base does not have is not added.
    assert "nope" not in prompt.dynamic_context
    assert "'X'" not in prompt.dynamic_context
    assert "Dallas" not in prompt.static_system


def test_the_verbose_schema_shows_the_list_too(
    provide: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries import config  # noqa: PLC0415

    monkeypatch.setattr(config, "SCHEMA_FORMAT", "verbose")
    provide(lambda _q: PromptValues(lists={REGION: ("EU", "US")}))
    with _on():
        prompt = _prompt()
    assert "  - region (TEXT); values: ['EU', 'US']" in prompt.static_system


def test_what_is_shown_is_capped_and_kept_to_a_line(provide: Any) -> None:
    long = "x" * 500
    provide(
        lambda _q: PromptValues(
            lists={REGION: tuple(f"r{i}" for i in range(pv.MAX_LIST + 10)) + (long,)},
            matching={CITY: tuple(f"c{i}" for i in range(pv.MAX_MATCHING + 5))},
        )
    )
    with _on():
        prompt = _prompt()
    # Trimmed, so no longer every value: shown as samples (from #422's review).
    assert "region:TEXT, samples: ['r0'" in prompt.static_system
    assert "region:TEXT, values:" not in prompt.static_system
    assert f"'r{pv.MAX_LIST - 1}'" in prompt.static_system
    assert f"'r{pv.MAX_LIST}'" not in prompt.static_system
    assert long not in prompt.static_system
    assert f"'c{pv.MAX_MATCHING - 1}'" in prompt.dynamic_context
    assert f"'c{pv.MAX_MATCHING}'" not in prompt.dynamic_context


@pytest.mark.parametrize(
    "answer",
    [
        lambda _q: (_ for _ in ()).throw(RuntimeError("the index is down")),
        lambda _q: {"lists": {REGION: ("APAC",)}},
        lambda _q: None,
    ],
    ids=["raises", "wrong-type", "none"],
)
def test_a_provider_that_gives_nothing_usable_leaves_the_prompt_as_without(
    provide: Any, answer: Any
) -> None:
    with use_query_settings(QuerySettings(value_index=False)):
        without = _prompt()
    provide(answer)
    with _on():
        prompt = _prompt()
    assert prompt.static_system == without.static_system
    assert prompt.dynamic_context == without.dynamic_context


def test_a_slow_provider_is_given_up_on(provide: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pv, "TIMEOUT_S", 0.2)

    def slow(_q: str) -> PromptValues:
        time.sleep(2)
        return PromptValues(lists={REGION: ("APAC",)})

    provide(slow)
    started = time.monotonic()
    with _on():
        prompt = _prompt()
    assert time.monotonic() - started < 1.5
    assert "APAC" not in prompt.static_system


def test_the_provider_runs_in_the_request_s_context(provide: Any) -> None:
    """A wrapper reads request-bound state, such as the row filters its index
    must have been built under, from context variables."""
    bound: contextvars.ContextVar[str | None] = contextvars.ContextVar("bound", default=None)
    seen: list[str | None] = []

    def reads(_q: str) -> PromptValues:
        seen.append(bound.get())
        return PromptValues()

    provide(reads)
    token = bound.set("filters-for-this-request")
    try:
        with _on():
            _prompt()
            asyncio.run(assemble_prompt_async("q", KB, agent_id="agent-1"))
    finally:
        bound.reset(token)
    assert seen == ["filters-for-this-request", "filters-for-this-request"]


def test_the_async_prompt_shows_the_same(provide: Any) -> None:
    provide(lambda _q: PromptValues(lists={REGION: ("EU", "US")}, matching={CITY: ("Dallas",)}))
    with _on():
        sync = _prompt()
        async_ = asyncio.run(
            assemble_prompt_async("stores in dallas fort worth", KB, agent_id="agent-1")
        )
    assert async_.static_system == sync.static_system
    assert async_.dynamic_context == sync.dynamic_context


def test_without_an_agent_or_a_provider_nothing_is_asked(
    provide: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = provide(lambda _q: PromptValues(lists={REGION: ("APAC",)}))
    with _on():
        assert assemble_prompt("q", KB).static_system.count("APAC") == 0
    assert asked == []
    monkeypatch.setattr(
        loader, "lookup_source", lambda _agent: LookupSource(key="k", open=lambda: None)
    )
    with _on():
        assert pv.prompt_values_for("agent-1", "q") is pv.EMPTY


def test_a_list_that_loses_a_value_too_long_for_a_line_is_shown_as_samples(
    provide: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries import config  # noqa: PLC0415

    provide(lambda _q: PromptValues(lists={REGION: ("EU", "x" * 500)}))
    with _on():
        assert "region:TEXT, samples: ['EU']" in _prompt().static_system
    monkeypatch.setattr(config, "SCHEMA_FORMAT", "verbose")
    with _on():
        assert "  - region (TEXT); samples: ['EU']" in _prompt().static_system
