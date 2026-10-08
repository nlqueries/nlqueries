"""What reaches the provider for Claude 5 and later, and what does not.

Claude Sonnet 5.5 and Haiku 5.5 reject a temperature with a 400, and take an
``output_config.effort``. Both rules are keyed on the model id, so each is
checked on both clients for a 5.x model and for the models before it, whose
requests must not change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.claude import accepts_temperature, claude_generation, effort_for
from nlqueries.llm.litellm_client import LiteLLMClient

# ---------------------------------------------------------------------------
# Reading the generation from the id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "generation"),
    [
        ("claude-sonnet-5-5", 5),
        ("claude-haiku-5-5", 5),
        ("claude-opus-5-5", 5),
        ("anthropic/claude-sonnet-5-5", 5),
        ("bedrock/us.anthropic.claude-sonnet-5-5", 5),
        ("openrouter/anthropic/claude-sonnet-5.5", 5),
        ("CLAUDE-SONNET-5-5", 5),
        ("claude-sonnet-4-6", 4),
        ("claude-haiku-4-5-20251001", 4),
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", 4),
        # The pre-4 names put the version first, and neither rule applies.
        ("claude-3-5-sonnet-20241022", None),
        ("openai/gpt-4o", None),
        ("deepseek/deepseek-chat", None),
    ],
)
def test_the_generation_is_read_from_the_id(model: str, generation: int | None) -> None:
    assert claude_generation(model) == generation


def test_only_claude_5_and_later_lose_the_temperature() -> None:
    assert not accepts_temperature("claude-sonnet-5-5")
    assert not accepts_temperature("bedrock/us.anthropic.claude-haiku-5-5")
    assert accepts_temperature("claude-haiku-4-5-20251001")
    assert accepts_temperature("claude-3-5-sonnet-20241022")
    assert accepts_temperature("openai/gpt-4o")


def test_only_claude_5_and_later_get_an_effort() -> None:
    with patch.object(config, "LLM_EFFORT", "low"):
        assert effort_for("claude-sonnet-5-5") == "low"
        assert effort_for("claude-haiku-4-5-20251001") is None
        assert effort_for("openai/gpt-4o") is None
    with patch.object(config, "LLM_EFFORT", None):
        assert effort_for("claude-sonnet-5-5") is None


# ---------------------------------------------------------------------------
# LLM_EFFORT
# ---------------------------------------------------------------------------


def test_unset_effort_is_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    assert config._effort() == "low"


def test_a_blank_effort_sends_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blanking a value is how people switch one off: the model's own default."""
    for written in ("", "  "):
        monkeypatch.setenv("LLM_EFFORT", written)
        assert config._effort() is None, repr(written)


def test_an_effort_level_is_read_case_and_space_blind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EFFORT", " HIGH ")
    assert config._effort() == "high"


def test_a_misspelt_effort_falls_back_to_low_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Not to "nothing", which would quietly raise every request to the model's default."""
    monkeypatch.setenv("LLM_EFFORT", "lowest")
    with caplog.at_level(logging.WARNING):
        assert config._effort() == "low"
    assert "LLM_EFFORT" in caplog.text


def test_the_defaults_are_the_5_5_models_with_room_to_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL_FAST", raising=False)
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config._detect_model("anthropic") == "claude-sonnet-5-5"
    with patch.object(config, "_configured_model", return_value="claude-sonnet-5-5"):
        assert config._detect_fast_model("anthropic") == "claude-haiku-5-5"
    assert config._output_budget() == 4096


# ---------------------------------------------------------------------------
# AnthropicClient: both ride in extra_body
# ---------------------------------------------------------------------------


def _anthropic_response() -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = "SELECT 1"
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def _anthropic_acomplete(model: str, temperature: float | None) -> dict[str, Any]:
    sdk = MagicMock()
    sdk.messages.create = AsyncMock(return_value=_anthropic_response())
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic", return_value=sdk),
    ):
        client = AnthropicClient(model=model, api_key="k")
        asyncio.run(client.acomplete("sys", "user", temperature=temperature))
    kwargs: dict[str, Any] = dict(sdk.messages.create.call_args.kwargs)
    return kwargs


def test_anthropic_sends_claude_5_an_effort_and_no_temperature() -> None:
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _anthropic_acomplete("claude-sonnet-5-5", 0.4)
    assert kwargs["extra_body"] == {"output_config": {"effort": "low"}}
    assert "temperature" not in kwargs


def test_anthropic_keeps_a_temperature_for_claude_4_in_extra_body() -> None:
    """In extra_body, because anthropic 1.x refuses it as an argument (TypeError)."""
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _anthropic_acomplete("claude-haiku-4-5-20251001", 0.4)
    assert kwargs["extra_body"] == {"temperature": 0.4}
    assert "temperature" not in kwargs


def test_anthropic_requests_before_claude_5_are_unchanged() -> None:
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _anthropic_acomplete("claude-haiku-4-5-20251001", None)
    assert "extra_body" not in kwargs


def test_anthropic_sends_no_effort_when_it_is_switched_off() -> None:
    with patch.object(config, "LLM_EFFORT", None):
        kwargs = _anthropic_acomplete("claude-sonnet-5-5", 0.4)
    assert "extra_body" not in kwargs


def test_anthropic_sends_the_effort_on_every_path() -> None:
    """complete, stream and astream, not only the async completion."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _anthropic_response()
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    stream_ctx.text_stream = iter(["SELECT 1"])
    sdk.messages.stream.return_value = stream_ctx

    async_stream = MagicMock()

    async def _tokens() -> Any:
        yield "SELECT 1"

    async_stream.text_stream = _tokens()
    async_stream.get_final_message = AsyncMock(return_value=_anthropic_response())
    async_ctx = MagicMock()
    async_ctx.__aenter__ = AsyncMock(return_value=async_stream)
    async_ctx.__aexit__ = AsyncMock(return_value=False)
    asdk = MagicMock()
    asdk.messages.stream.return_value = async_ctx

    async def _drain(client: AnthropicClient) -> None:
        async for _ in client.astream("sys", "user"):
            pass

    with (
        patch.object(config, "LLM_EFFORT", "low"),
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=sdk),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic", return_value=asdk),
    ):
        client = AnthropicClient(model="claude-haiku-5-5", api_key="k")
        client.complete("sys", "user")
        list(client.stream("sys", "user"))
        asyncio.run(_drain(client))

    want = {"output_config": {"effort": "low"}}
    assert sdk.messages.create.call_args.kwargs["extra_body"] == want
    assert sdk.messages.stream.call_args.kwargs["extra_body"] == want
    assert asdk.messages.stream.call_args.kwargs["extra_body"] == want


# ---------------------------------------------------------------------------
# LiteLLMClient: output_config as an argument, and only towards Anthropic
# ---------------------------------------------------------------------------


def _litellm_response() -> MagicMock:
    choice = MagicMock()
    choice.message.content = "SELECT 1"
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


def _litellm_acomplete(model: str, temperature: float | None) -> dict[str, Any]:
    call = AsyncMock(return_value=_litellm_response())
    with patch("nlqueries.llm.litellm_client.litellm.acompletion", call):
        asyncio.run(LiteLLMClient(model=model).acomplete("sys", "user", temperature=temperature))
    kwargs: dict[str, Any] = dict(call.call_args.kwargs)
    return kwargs


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5-5", "claude-haiku-5-5"])
def test_litellm_sends_claude_5_an_effort_and_no_temperature(model: str) -> None:
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _litellm_acomplete(model, 0.4)
    assert kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in kwargs
    assert "extra_body" not in kwargs


def test_litellm_sends_bedrock_claude_5_no_effort_and_no_temperature() -> None:
    """Bedrock's effort shape is unchecked, so nothing; the 400 on temperature is the model's."""
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _litellm_acomplete("bedrock/us.anthropic.claude-sonnet-5-5", 0.4)
    assert "output_config" not in kwargs
    assert "temperature" not in kwargs


def test_litellm_requests_to_other_models_are_unchanged() -> None:
    with patch.object(config, "LLM_EFFORT", "low"):
        kwargs = _litellm_acomplete("openai/gpt-4o", 0.4)
        older = _litellm_acomplete("anthropic/claude-haiku-4-5-20251001", None)
    assert kwargs["temperature"] == 0.4
    assert "output_config" not in kwargs
    assert "output_config" not in older


def test_litellm_sends_the_effort_on_every_path() -> None:
    chunk = MagicMock()
    chunk.choices[0].delta.content = "SELECT 1"
    chunk.choices[0].finish_reason = "stop"

    async def _achunks() -> Any:
        yield chunk

    sync_call = MagicMock(side_effect=[_litellm_response(), iter([chunk])])
    async_call = AsyncMock(return_value=_achunks())

    async def _drain(client: LiteLLMClient) -> None:
        async for _ in client.astream("sys", "user"):
            pass

    with (
        patch.object(config, "LLM_EFFORT", "medium"),
        patch("nlqueries.llm.litellm_client.litellm.completion", sync_call),
        patch("nlqueries.llm.litellm_client.litellm.acompletion", async_call),
    ):
        client = LiteLLMClient(model="anthropic/claude-sonnet-5-5")
        client.complete("sys", "user")
        list(client.stream("sys", "user"))
        asyncio.run(_drain(client))

    want = {"effort": "medium"}
    assert [c.kwargs["output_config"] for c in sync_call.call_args_list] == [want, want]
    assert async_call.call_args.kwargs["output_config"] == want
