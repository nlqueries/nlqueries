"""Tests for the task-local LLM override (get_llm_client + use_llm_override)."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
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


def test_extra_with_a_provider_that_cannot_accept_it_raises() -> None:
    """Dropping it silently is the dangerous outcome, not the safe one.

    An override carrying AWS credentials describes a Bedrock call. If the
    provider resolves to anthropic, those credentials are meaningless there and
    the request would still go out — to the public Anthropic API, under the
    process-level key, which is the exact egress a deployment chose Bedrock to
    avoid. Nothing reports it, because the answer comes back correct from the
    wrong place. So it fails instead, naming the fix.
    """
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        use_llm_override(
            LLMOverride(
                provider="anthropic", model="claude-x", extra={"aws_region_name": "eu-west-1"}
            )
        ),
        pytest.raises(ValueError, match="cannot accept it"),
    ):
        get_llm_client()


def test_extra_left_unset_still_reaches_a_non_litellm_provider_normally() -> None:
    """The negative control: only a populated extra is an error, not the field."""
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        use_llm_override(LLMOverride(provider="anthropic", model="claude-x")),
    ):
        assert isinstance(get_llm_client(), AnthropicClient)


def test_an_override_with_extra_but_no_provider_raises_rather_than_defaulting() -> None:
    """The case that prompted this: a partial override, which is a supported shape.

    `provider` falls back to `config.LLM_PROVIDER` — `anthropic` on a default
    install — so an override built only from `model` and `extra` is precisely
    how the silent-Anthropic path is reached.
    """
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        use_llm_override(LLMOverride(model="bedrock/x", extra={"aws_region_name": "eu-west-1"})),
        pytest.raises(ValueError, match="cannot accept it"),
    ):
        get_llm_client()


def test_a_None_inside_extra_is_forwarded_rather_than_dropped() -> None:
    """LiteLLM reads a missing AWS credential as "use the boto3 chain".

    Whether to send one is the caller's decision, so core carries the value it
    was given instead of second-guessing it.
    """
    client = LiteLLMClient(model="bedrock/x", extra={"aws_session_token": None})
    assert client._auth_kwargs() == {"aws_session_token": None}


def test_extra_may_not_carry_a_kwarg_the_client_already_passes() -> None:
    """Rejected at construction, because the collision is otherwise inconsistent.

    `_auth_kwargs()` is spread into the completion call directly in the sync and
    streaming paths, where a duplicate keyword raises TypeError. In `acomplete`
    it is merged into a dict literal after `model`/`messages`/`max_tokens`, where
    it silently wins instead. A host that put `max_tokens` in `extra` would get
    an exception from one method and a quietly capped answer from the other.
    """
    for reserved in ("model", "messages", "max_tokens", "stream", "temperature"):
        with pytest.raises(ValueError, match="may not contain"):
            LiteLLMClient(model="bedrock/x", extra={reserved: "anything"})


def test_an_ordinary_extra_key_is_still_accepted() -> None:
    """The negative control: only the names this class passes are refused."""
    client = LiteLLMClient(model="bedrock/x", extra={"aws_region_name": "eu-west-1"})
    assert client._auth_kwargs() == {"aws_region_name": "eu-west-1"}


def test_the_reserved_names_are_exactly_what_the_client_passes() -> None:
    """A drifting list is worse than none: it would refuse a valid key, or miss one.

    If a completion argument is added to this class without being added to the
    reserved set, the silent-override path reopens for that name.
    """
    from nlqueries.llm import litellm_client as module

    tree = ast.parse(textwrap.dedent(inspect.getsource(module.LiteLLMClient)))
    passed: set[str] = set()
    for node in ast.walk(tree):
        # `litellm.completion(model=..., stream=True, ...)`
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"completion", "acompletion"}:
                passed |= {kw.arg for kw in node.keywords if kw.arg is not None}
        # the kwargs dict built in `acomplete`
        elif isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if "model" in keys:
                passed |= {k for k in keys if isinstance(k, str)}
        # `kwargs["temperature"] = temperature`
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    passed.add(target.slice.value)

    # `api_key`/`api_base` are written by `_auth_kwargs` itself and are
    # deliberately NOT reserved: extra may carry them, and the explicit
    # constructor arguments are applied afterwards so they win. That is asserted
    # by test_an_explicit_api_key_wins_over_one_inside_extra.
    passed -= {"api_key", "api_base"}

    assert passed == module._RESERVED_COMPLETION_KWARGS, (
        "the names this class passes to litellm and the names extra is refused "
        f"have diverged: extra={sorted(passed - module._RESERVED_COMPLETION_KWARGS)}, "
        f"missing={sorted(module._RESERVED_COMPLETION_KWARGS - passed)}"
    )
