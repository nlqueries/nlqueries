"""
tests.test_bedrock_prompt_caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt caching for Claude on Amazon Bedrock (owner, 2026-10-05).

The SQL prompt's stable block -- instructions and the full schema, identical for
every question to an agent -- is marked with ``cache_control``. The Anthropic
client always sent that; the LiteLLM client, which carries Bedrock, flattened
the system prompt and dropped it, so Bedrock never cached. It now keeps the
blocks for Claude on Bedrock, and LiteLLM turns the marker into Bedrock's
``cachePoint``. Nothing here reaches the network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.llm import litellm_client
from nlqueries.llm.litellm_client import LiteLLMClient, _caches_prompts, _system_message
from nlqueries.llm.usage import use_usage_sink

_SONNET = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
_BLOCKS: list[dict[str, Any]] = [
    {"type": "text", "text": "STATIC schema and rules", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "dynamic capsules"},
]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_SONNET, True),
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0", True),
        # Not Claude on Bedrock: the flattened prompt they always had.
        ("bedrock/us.amazon.nova-pro-v1:0", False),
        ("bedrock/us.meta.llama3-1-70b-instruct-v1:0", False),
        ("openai/gpt-4o", False),
        ("anthropic/claude-sonnet-4-5", False),
    ],
)
def test_only_claude_on_bedrock_takes_cache_markers(model: str, expected: bool) -> None:
    assert _caches_prompts(model) is expected
    assert LiteLLMClient(model=model).supports_prompt_caching is expected


def test_a_model_litellm_does_not_know_to_cache_is_left_alone() -> None:
    """The family check is not enough on its own: LiteLLM must also list the
    model as supporting prompt caching."""
    with patch.object(litellm_client.litellm.utils, "supports_prompt_caching", return_value=False):
        assert _caches_prompts(_SONNET) is False


def _response(**usage: Any) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="SELECT 1"), finish_reason="stop")
        ],
        usage=SimpleNamespace(**usage) if usage else None,
    )


def test_bedrock_claude_sends_the_blocks_with_their_marker() -> None:
    call = MagicMock(return_value=_response())
    with patch.object(litellm_client.litellm, "completion", call):
        LiteLLMClient(model=_SONNET).complete(_BLOCKS, "how many orders?")
    system = call.call_args.kwargs["messages"][0]
    assert system == {"role": "system", "content": _BLOCKS}


def test_bedrock_claude_sends_the_blocks_on_the_async_path_too() -> None:
    call = AsyncMock(return_value=_response())
    with patch.object(litellm_client.litellm, "acompletion", call):
        asyncio.run(LiteLLMClient(model=_SONNET).acomplete(_BLOCKS, "how many orders?"))
    assert call.call_args.kwargs["messages"][0]["content"] == _BLOCKS


def test_other_models_still_get_the_flattened_prompt() -> None:
    call = MagicMock(return_value=_response())
    with patch.object(litellm_client.litellm, "completion", call):
        LiteLLMClient(model="openai/gpt-4o").complete(_BLOCKS, "q")
    assert call.call_args.kwargs["messages"][0] == {
        "role": "system",
        "content": "STATIC schema and rules\ndynamic capsules",
    }


def test_a_plain_string_system_prompt_is_unchanged() -> None:
    assert _system_message("sys", keep_blocks=True) == {"role": "system", "content": "sys"}


def test_litellm_turns_the_marker_into_a_bedrock_cache_point() -> None:
    """The contract this relies on, checked against the pinned LiteLLM without
    sending anything: a cachePoint right after the stable block, and the
    per-question block after it -- outside the cached prefix."""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    request = AmazonConverseConfig()._transform_request(
        model=_SONNET.removeprefix("bedrock/"),
        messages=[_system_message(_BLOCKS, keep_blocks=True), {"role": "user", "content": "q"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["system"] == [
        {"text": "STATIC schema and rules"},
        {"cachePoint": {"type": "default"}},
        {"text": "dynamic capsules"},
    ]


def test_cache_writes_are_recorded_apart_from_plain_input() -> None:
    """Bedrock reports tokens written to the cache, and prompt_tokens counts
    them; they are billed at their own rate, so they come out of plain input
    and into cache_write_tokens -- as reads already did."""
    records: list[Any] = []
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_creation_tokens=1500),
    )
    with use_usage_sink(records):
        litellm_client._record_litellm_usage(_SONNET, usage)
    (record,) = records
    assert (record.input_tokens, record.cache_write_tokens, record.cache_read_tokens) == (
        500,
        1500,
        0,
    )

    records.clear()
    usage.prompt_tokens_details = SimpleNamespace(cached_tokens=1500, cache_creation_tokens=0)
    with use_usage_sink(records):
        litellm_client._record_litellm_usage(_SONNET, usage)
    (record,) = records
    assert (record.input_tokens, record.cache_write_tokens, record.cache_read_tokens) == (
        500,
        0,
        1500,
    )
