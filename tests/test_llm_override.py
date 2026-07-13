"""Tests for the task-local LLM override (get_llm_client + use_llm_override)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from nlqueries.llm import (
    LLMOverride,
    current_llm_override,
    get_llm_client,
    use_llm_override,
)
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.litellm_client import LiteLLMClient


def test_no_override_returns_default_client() -> None:
    assert current_llm_override() is None
    client = get_llm_client()
    # Default construction reads config; no explicit key is attached.
    if isinstance(client, LiteLLMClient):
        assert client._api_key is None


def test_override_selects_provider_model_and_key() -> None:
    override = LLMOverride(
        provider="litellm", model="openai/gpt-4o-mini", api_key="sk-tenant", api_base="https://x"
    )
    with use_llm_override(override):
        client = get_llm_client()
        assert isinstance(client, LiteLLMClient)
        assert client._model == "openai/gpt-4o-mini"
        assert client._api_key == "sk-tenant"
        assert client._api_base == "https://x"


def test_override_is_reset_after_context() -> None:
    with use_llm_override(LLMOverride(provider="litellm", model="openai/gpt-4o", api_key="k")):
        assert current_llm_override() is not None
    assert current_llm_override() is None


def test_fast_tier_prefers_fast_model_from_override() -> None:
    override = LLMOverride(provider="litellm", model="m-default", fast_model="m-fast")
    with use_llm_override(override):
        assert get_llm_client(tier="fast")._model == "m-fast"
        assert get_llm_client()._model == "m-default"


def test_anthropic_override_passes_api_key_to_sdk() -> None:
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic") as sync_cls,
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic") as async_cls,
    ):
        sync_cls.return_value = MagicMock()
        async_cls.return_value = MagicMock()
        AnthropicClient(model="claude-x", api_key="sk-ant-tenant")
    assert sync_cls.call_args.kwargs["api_key"] == "sk-ant-tenant"
    assert async_cls.call_args.kwargs["api_key"] == "sk-ant-tenant"


def test_override_is_task_local() -> None:
    """Concurrent tasks must not see each other's override (contextvar isolation)."""

    async def _run() -> tuple[str | None, str | None]:
        async def worker(key: str) -> str | None:
            with use_llm_override(LLMOverride(provider="litellm", model="m", api_key=key)):
                await asyncio.sleep(0.01)
                ov = current_llm_override()
                return ov.api_key if ov else None

        return await asyncio.gather(worker("a"), worker("b"))  # type: ignore[return-value]

    a, b = asyncio.run(_run())
    assert {a, b} == {"a", "b"}  # each task kept its own value


def test_litellm_auth_kwargs_omitted_when_unset() -> None:
    client = LiteLLMClient(model="openai/gpt-4o")
    assert client._auth_kwargs() == {}
    client_with = LiteLLMClient(model="openai/gpt-4o", api_key="k", api_base="b")
    assert client_with._auth_kwargs() == {"api_key": "k", "api_base": "b"}
