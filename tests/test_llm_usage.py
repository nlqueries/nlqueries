# nlqueries-core — OSS (BSL 1.1)
"""Tests for the LLM usage-observation seam (nlqueries.llm.usage) and the
per-client usage recording. No network: SDK calls are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nlqueries.llm.usage import (
    UsageRecord,
    current_usage_sink,
    estimate_tokens,
    record_usage,
    use_usage_sink,
)

# ---------------------------------------------------------------------------
# Seam mechanics
# ---------------------------------------------------------------------------


def test_record_usage_appends_when_sink_bound() -> None:
    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        assert current_usage_sink() is sink
        record_usage(UsageRecord(model="m", input_tokens=3, output_tokens=1))
    assert len(sink) == 1
    assert sink[0].model == "m"
    assert sink[0].input_tokens == 3


def test_record_usage_is_noop_when_unbound() -> None:
    # No sink bound → nothing happens, no error.
    assert current_usage_sink() is None
    record_usage(UsageRecord(model="m", input_tokens=99))  # must not raise


def test_sink_resets_after_block() -> None:
    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        pass
    assert current_usage_sink() is None
    record_usage(UsageRecord(model="m"))
    assert sink == []  # nothing recorded outside the block


def test_none_binding_is_a_noop() -> None:
    with use_usage_sink(None):
        record_usage(UsageRecord(model="m"))  # goes nowhere, no error
    assert current_usage_sink() is None


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1  # max(1, ...)
    assert estimate_tokens("x" * 40) == 10


# ---------------------------------------------------------------------------
# Anthropic usage mapping
# ---------------------------------------------------------------------------


def test_anthropic_usage_maps_all_fields() -> None:
    from nlqueries.llm.anthropic_client import _record_anthropic_usage

    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=40,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=50,
    )
    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        _record_anthropic_usage("claude", usage)
    assert sink == [
        UsageRecord(
            model="claude",
            input_tokens=100,
            output_tokens=40,
            cache_read_tokens=200,
            cache_write_tokens=50,
            estimated=False,
        )
    ]


def test_anthropic_usage_none_is_noop() -> None:
    from nlqueries.llm.anthropic_client import _record_anthropic_usage

    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        _record_anthropic_usage("claude", None)
    assert sink == []


def test_anthropic_complete_records_usage() -> None:
    from nlqueries.llm.anthropic_client import AnthropicClient

    client = AnthropicClient(model="claude-x", api_key="test-key")
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=3,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    sink: list[UsageRecord] = []
    with (
        patch.object(client._client.messages, "create", return_value=fake_response),
        use_usage_sink(sink),
    ):
        out = client.complete("system", "user")
    assert out == "hello"
    assert len(sink) == 1
    assert (sink[0].input_tokens, sink[0].output_tokens) == (12, 3)
    assert sink[0].estimated is False


# ---------------------------------------------------------------------------
# LiteLLM usage mapping
# ---------------------------------------------------------------------------


def test_litellm_usage_splits_cached_tokens() -> None:
    from nlqueries.llm.litellm_client import _record_litellm_usage

    usage = SimpleNamespace(
        prompt_tokens=100,  # includes cached
        completion_tokens=25,
        prompt_tokens_details=SimpleNamespace(cached_tokens=30),
    )
    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        _record_litellm_usage("openai/gpt", usage)
    assert sink[0].input_tokens == 70  # 100 - 30 cached
    assert sink[0].cache_read_tokens == 30
    assert sink[0].output_tokens == 25
    assert sink[0].estimated is False


def test_litellm_estimated_record_is_flagged() -> None:
    from nlqueries.llm.litellm_client import _record_estimated

    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        _record_estimated("openai/gpt", "x" * 40, "y" * 20)
    assert sink[0].estimated is True
    assert sink[0].input_tokens == 10
    assert sink[0].output_tokens == 5


def test_litellm_complete_records_exact_usage() -> None:
    from nlqueries.llm import litellm_client

    client = litellm_client.LiteLLMClient(model="openai/gpt-4o")
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, prompt_tokens_details=None),
    )
    sink: list[UsageRecord] = []
    with (
        patch.object(litellm_client.litellm, "completion", MagicMock(return_value=fake_response)),
        use_usage_sink(sink),
    ):
        out = client.complete("system", "user")
    assert out == "hi"
    assert (sink[0].input_tokens, sink[0].output_tokens) == (8, 2)
    assert sink[0].estimated is False


def test_litellm_stream_records_estimate() -> None:
    from nlqueries.llm import litellm_client

    client = litellm_client.LiteLLMClient(model="openai/gpt-4o")

    def _chunks() -> object:
        for piece in ("he", "llo"):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    sink: list[UsageRecord] = []
    with (
        patch.object(litellm_client.litellm, "completion", MagicMock(return_value=_chunks())),
        use_usage_sink(sink),
    ):
        out = "".join(client.stream("system", "user"))
    assert out == "hello"
    assert len(sink) == 1
    assert sink[0].estimated is True
    assert sink[0].output_tokens == estimate_tokens("hello")
