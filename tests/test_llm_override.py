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


# ----------------------------------------------------------------------
# extra: provider-specific kwargs (Bedrock's AWS region and credentials)
# ----------------------------------------------------------------------


def test_extra_is_forwarded_as_completion_kwargs() -> None:
    """The only path AWS settings have to the SDK; nothing else carries them."""
    client = LiteLLMClient(model="bedrock/x", extra={"aws_region_name": "eu-west-1"})
    assert client._auth_kwargs() == {"aws_region_name": "eu-west-1"}


def test_an_explicit_api_key_wins_over_one_inside_extra() -> None:
    """extra is applied first precisely so the named arguments can override it."""
    client = LiteLLMClient(model="m", api_key="explicit", extra={"api_key": "from-extra"})
    assert client._auth_kwargs()["api_key"] == "explicit"


def test_auth_kwargs_does_not_mutate_the_caller_dict() -> None:
    """One client serves many calls, and the dict belongs to whoever built it."""
    extra = {"aws_region_name": "us-east-1"}
    client = LiteLLMClient(model="m", api_key="k", api_base="b", extra=extra)
    client._auth_kwargs()
    client._auth_kwargs()
    assert extra == {"aws_region_name": "us-east-1"}


def test_extra_reaches_the_client_through_an_override() -> None:
    override = LLMOverride(
        provider="litellm",
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        extra={"aws_region_name": "ap-south-1", "aws_access_key_id": "AKIA"},
    )
    with use_llm_override(override):
        client = get_llm_client()
    assert isinstance(client, LiteLLMClient)
    assert client._auth_kwargs() == {
        "aws_region_name": "ap-south-1",
        "aws_access_key_id": "AKIA",
    }


def test_extra_is_not_offered_to_a_provider_that_cannot_accept_it() -> None:
    """AnthropicClient has a narrow constructor: passing extra would be a TypeError.

    Dropping it is the intended outcome rather than a limitation. An override
    carrying AWS credentials describes a Bedrock call, and must not quietly
    become an Anthropic one because the provider field says so.
    """
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        use_llm_override(
            LLMOverride(
                provider="anthropic", model="claude-x", extra={"aws_region_name": "eu-west-1"}
            )
        ),
    ):
        client = get_llm_client()
    assert isinstance(client, AnthropicClient)


def test_a_None_inside_extra_is_forwarded_rather_than_dropped() -> None:
    """LiteLLM reads a missing AWS credential as "use the boto3 chain".

    Whether to send one is the caller's decision, so core carries the value it
    was given instead of second-guessing it.
    """
    client = LiteLLMClient(model="bedrock/x", extra={"aws_session_token": None})
    assert client._auth_kwargs() == {"aws_session_token": None}
