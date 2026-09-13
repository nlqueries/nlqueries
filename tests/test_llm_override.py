"""Tests for the task-local LLM override (get_llm_client + use_llm_override)."""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import logging
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.llm import (
    LLMOverride,
    current_llm_override,
    get_llm_client,
    output_budget,
    use_llm_override,
)
from nlqueries.llm import override as override_mod
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import LLMClient
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
    """A partial override is a supported shape, and it resolves the provider from config.

    `provider` falls back to `config.LLM_PROVIDER` — `anthropic` on a default
    install — so an override built only from `model` and `extra` would otherwise
    hand a client that cannot use `extra` a request that depends on it.

    The model here is deliberately *not* a `bedrock/` id: that case is now
    resolved rather than refused, because the model names the provider (see
    test_a_bedrock_model_with_no_provider_goes_to_litellm). This covers what is
    left — an `extra` for some other provider, with nothing to say where it goes.
    """
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        use_llm_override(LLMOverride(model="claude-x", extra={"vertex_project": "p"})),
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


# ----------------------------------------------------------------------
# Bedrock is decided by the model, not only by the provider field
# ----------------------------------------------------------------------


def test_a_bedrock_model_with_no_provider_goes_to_litellm() -> None:
    """The documented IAM deployment, and the third way it reached Anthropic.

    On EC2, ECS or EKS the credentials come from the host role, so a host
    application resolving per-tenant settings builds `LLMOverride(model=
    "bedrock/...")` and leaves `provider` unset — no `extra` to trigger the other
    guard. `provider` then fell back to `config.LLM_PROVIDER`, and
    `AnthropicClient` does not refuse a `bedrock/` id: it sends the system
    prompt, the schema and the question to api.anthropic.com and only then fails
    with a model-not-found. For a deployment that adopted Bedrock to keep traffic
    inside its account, the data has already left.
    """
    with (
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        use_llm_override(LLMOverride(model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")),
    ):
        client = get_llm_client()

    assert isinstance(client, LiteLLMClient)


def test_a_bedrock_model_contradicting_a_named_provider_raises() -> None:
    """Two settings naming different providers is a mistake, not a preference."""
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        use_llm_override(LLMOverride(provider="anthropic", model="bedrock/x")),
        pytest.raises(ValueError, match="contradicts model"),
    ):
        get_llm_client()


def test_a_bedrock_provider_name_on_an_override_is_accepted() -> None:
    """`LLM_PROVIDER=bedrock` was accepted; the override channel was not.

    A host that stores a provider name per tenant — the caller `extra` exists for
    — would have that word in its settings store and hit "Unknown LLM provider".
    Both channels now normalise it in the same place.
    """
    with use_llm_override(LLMOverride(provider="bedrock", model="bedrock/x")):
        assert isinstance(get_llm_client(), LiteLLMClient)


def test_naming_bedrock_without_a_bedrock_model_raises_at_the_client() -> None:
    """Deferred from import time, but not dropped: the egress is still refused."""
    with (
        use_llm_override(LLMOverride(provider="bedrock", model="claude-sonnet-4-5")),
        pytest.raises(ValueError, match="not a Bedrock model"),
    ):
        get_llm_client()


def test_the_env_naming_bedrock_is_also_refused_without_a_bedrock_model() -> None:
    """The same rule through the environment channel rather than the override."""
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "bedrock"),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "claude-sonnet-4-5"),
        pytest.raises(ValueError, match="not a Bedrock model"),
    ):
        get_llm_client()


def test_plain_litellm_with_an_anthropic_model_is_still_ordinary_use() -> None:
    """The negative control that makes the flag necessary.

    `config.LLM_PROVIDER` is normalised to "litellm" for Bedrock, which loses the
    distinction — so without the separate flag this check would either miss the
    Bedrock case or reject a perfectly normal LiteLLM deployment.
    """
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "litellm"),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "claude-sonnet-4-5"),
    ):
        assert isinstance(get_llm_client(), LiteLLMClient)


# ----------------------------------------------------------------------
# The two tiers must agree about which cloud they are talking to
# ----------------------------------------------------------------------


def test_a_leftover_fast_model_is_refused_on_a_bedrock_deployment() -> None:
    """The tiers disagreeing is worse than either being wrong on its own.

    A `.env` still carrying `LLM_MODEL_FAST=claude-haiku-...` from a previous
    Anthropic setup left the default tier on Bedrock while every auxiliary call
    went to api.anthropic.com under the leftover key — and the auxiliary calls
    are intent classification and follow-up resolution, which carry the user's
    question and the conversation history. No error, because LiteLLM routes a
    bare `claude-` id perfectly happily.

    Judged from the *default* model, so both ways of selecting Bedrock agree.
    """
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", ""),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"),
        patch.object(config, "LLM_MODEL_FAST", "claude-haiku-4-5-20251001"),
        pytest.raises(ValueError, match="fast model"),
    ):
        get_llm_client(tier="fast")


def test_the_same_refusal_when_bedrock_was_named_rather_than_inferred() -> None:
    """The other channel. Before, one raised and the other silently sent data out.

    `LLM_MODEL` is deliberately *not* a Bedrock id here. With one, this passes
    through the model-prefix branch and says nothing about the branch it is named
    for — which is how the first version of this test survived deleting
    `names_bedrock` from the check it exists to cover.
    """
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "bedrock"),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "claude-sonnet-4-5"),
        patch.object(config, "LLM_MODEL_FAST", "claude-haiku-4-5-20251001"),
        pytest.raises(ValueError, match="fast model"),
    ):
        get_llm_client(tier="fast")


def test_the_refusal_names_the_setting_that_needs_changing() -> None:
    """`LLM_MODEL` and `LLM_MODEL_FAST` fail identically otherwise."""
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "bedrock"),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "bedrock/x"),
        patch.object(config, "LLM_MODEL_FAST", "claude-haiku-4-5-20251001"),
        pytest.raises(ValueError) as excinfo,
    ):
        get_llm_client(tier="fast")
    assert "LLM_MODEL_FAST" in str(excinfo.value)


def test_both_tiers_on_bedrock_are_accepted() -> None:
    """The negative control: the guard must not refuse a correct deployment."""
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", ""),
        patch.object(config, "LLM_PROVIDER", "litellm"),
        patch.object(config, "LLM_MODEL", "bedrock/sonnet"),
        patch.object(config, "LLM_MODEL_FAST", "bedrock/haiku"),
    ):
        assert get_llm_client(tier="fast")._model == "bedrock/haiku"
        assert get_llm_client()._model == "bedrock/sonnet"


def test_a_non_bedrock_deployment_is_untouched_by_the_tier_rule() -> None:
    """The other negative control, and the one that would bite everybody."""
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "anthropic"),
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        patch.object(config, "LLM_MODEL", "claude-sonnet-4-5"),
        patch.object(config, "LLM_MODEL_FAST", "claude-haiku-4-5-20251001"),
    ):
        assert get_llm_client(tier="fast")._model == "claude-haiku-4-5-20251001"


def test_the_environment_contradiction_is_refused_like_the_override_one() -> None:
    """`LLM_PROVIDER=anthropic` with a bedrock/ model is the same mistake.

    It used to be resolved silently to LiteLLM while the identical override was
    refused — and the documentation claimed both were refused.
    """
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", "anthropic"),
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        patch.object(config, "LLM_MODEL", "bedrock/x"),
        patch.object(config, "LLM_MODEL_FAST", "bedrock/x"),
        pytest.raises(ValueError, match="contradicts model"),
    ):
        get_llm_client()


def test_nothing_naming_a_provider_still_lets_the_model_decide() -> None:
    """The distinction the contradiction rule turns on: named versus detected.

    An unset LLM_PROVIDER resolves to `anthropic` by key detection, which must
    not be read as the operator naming Anthropic — that is the IAM deployment,
    and refusing it would break the documented setup.
    """
    with (
        patch.object(config, "LLM_PROVIDER_CONFIGURED", ""),
        patch.object(config, "LLM_PROVIDER", "anthropic"),
        patch.object(config, "LLM_MODEL", "bedrock/x"),
        patch.object(config, "LLM_MODEL_FAST", "bedrock/x"),
    ):
        assert isinstance(get_llm_client(), LiteLLMClient)


# ---------------------------------------------------------------------------
# Output budget
# ---------------------------------------------------------------------------


def test_budget_defaults_are_what_the_call_sites_used_to_hard_code() -> None:
    """The default must be a no-op, or this change moves every deployment's bill.

    1024 / 512 / 200 are the numbers that were written at the call sites before
    the budget was configurable. At the default they are what comes back, so an
    operator who sets nothing sees exactly the behaviour they had.
    """
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 1024):
        assert output_budget("answer") == 1024
        assert output_budget("correction") == 512
        assert output_budget("classification") == 200


def test_raising_the_budget_raises_the_short_calls_too() -> None:
    """The whole point of one number.

    A reasoning model bills its reasoning from the same allowance and spends it
    first, so raising only the answer leaves classification starved at 200 --
    and that is the tier that returns an EMPTY string rather than a short one.
    Measured on `deepseek-v4-pro`: 52 of 56 tokens went to reasoning.
    """
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 8192):
        assert output_budget("answer") == 8192
        assert output_budget("correction") == 4096
        assert output_budget("classification") == 1024


def test_lowering_the_budget_never_goes_under_the_floors() -> None:
    """A budget below the floors would make the short calls useless.

    `classification` has to fit a label; `correction` has to fit a SQL
    statement. Those are what the floors are, and they hold whatever the
    operator sets.
    """
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 100):
        assert output_budget("answer") == 100
        assert output_budget("correction") == 512
        assert output_budget("classification") == 200


def test_a_nonsense_budget_cannot_reach_the_provider() -> None:
    """0 is what an operator writes meaning "no limit". It means the opposite.

    Both SDKs reject a non-positive `max_tokens`, so an unclamped 0 would fail
    every LLM call in the process with an error naming `max_tokens` rather than
    the setting behind it. The literal 1024 this replaced made the value
    unreachable, so this failure mode is one the change introduced.

    Two channels, and both have to hold: the environment, clamped where it is
    read, and an `LLMOverride` from a settings store, which arrives unclamped.
    """
    for bad in (0, -1, -9999):
        with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", bad):
            assert output_budget("answer") >= 1
            assert output_budget("correction") >= 1
            assert output_budget("classification") >= 1
            with use_llm_override(LLMOverride(max_tokens=bad)):
                assert output_budget("answer") >= 1


def test_a_nonpositive_environment_value_is_ignored_not_clamped() -> None:
    """The documented default, not 1. A one-token answer is not a recovery.

    `try`/`finally` because the reload is process-wide state: without it a
    failing assertion skips the restore and leaves `LLM_MAX_OUTPUT_TOKENS`
    pinned at 1 for every test after this one, turning one real failure into a
    cascade of unrelated ones. `test_redshift_guards.py` guards the same shape
    the same way.
    """
    try:
        with patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS": "0"}):
            importlib.reload(config)
            assert config.LLM_MAX_OUTPUT_TOKENS == 1024
    finally:
        importlib.reload(config)


def test_a_clamped_environment_value_names_the_setting(caplog) -> None:  # type: ignore[no-untyped-def]
    """Clamping quietly is worse than not clamping at all.

    Uncorrected, a 0 reached the SDK and failed every call with an error that at
    least said `max_tokens`. Corrected in silence it does something subtler: the
    derived tiers sit on their floors, so classification and SQL repair keep
    working and only the ANSWER collapses to one token. The user sees a
    truncated answer and nothing names the cause. The log is the fix; the clamp
    is just what keeps the process running.
    """
    # `try` OUTSIDE `patch.dict`, so the restoring reload sees the real
    # environment. Nested the other way -- which is how this was written -- the
    # reload still saw `LLM_MAX_OUTPUT_TOKENS=0` and left the module pinned at
    # the default, so on a machine whose `.env` raises the budget every test
    # after this one ran against 1024. Exactly the cascade the sibling test's
    # docstring is about, introduced by the test written to describe it.
    try:
        with (
            patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS": "0"}),
            caplog.at_level(logging.WARNING),
        ):
            importlib.reload(config)
            assert config.LLM_MAX_OUTPUT_TOKENS == 1024
            assert "LLM_MAX_OUTPUT_TOKENS" in caplog.text
            # The message has to describe what actually happens. The first
            # version said "using the tier floor ... answers will be a single
            # token", which was true of a negative and false of the 0 a person
            # actually types -- 0 was falsy, so it fell back to the default and
            # the answer budget was never touched.
            assert "ignored" in caplog.text
    finally:
        importlib.reload(config)


def test_a_non_numeric_budget_does_not_abort_the_import(caplog) -> None:  # type: ignore[no-untyped-def]
    """`LLM_MAX_OUTPUT_TOKENS=` is how people disable a setting, and it is a typo.

    `int("")` raises, and an unhandled ValueError while `nlqueries.config` is
    importing takes down every CLI command with it -- including `doctor`, the one
    an operator would run to find out why nothing works. Tolerating 0 and
    negatives while falling over on a blank is the wrong way round: the blank is
    the likelier mistake.
    """
    for written in ("", "  ", "1024.5", "lots"):
        try:
            with (
                patch.dict(os.environ, {"LLM_MAX_OUTPUT_TOKENS": written}),
                caplog.at_level(logging.WARNING),
            ):
                caplog.clear()
                importlib.reload(config)
                assert config.LLM_MAX_OUTPUT_TOKENS == 1024, written
                assert "LLM_MAX_OUTPUT_TOKENS" in caplog.text, written
        finally:
            importlib.reload(config)


def test_a_clamped_override_names_itself_once(caplog) -> None:  # type: ignore[no-untyped-def]
    """Same for the channel `config` never sees, without flooding the log.

    `output_budget` runs on every LLM call, so a warning per call is a warning
    nobody reads.
    """
    override_mod._WARNED_BUDGETS.clear()
    for bad in (0, -5):
        override_mod._WARNED_BUDGETS.clear()
        caplog.clear()
        with (
            patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 4096),
            caplog.at_level(logging.WARNING),
            use_llm_override(LLMOverride(max_tokens=bad)),
        ):
            # Ignored, so the configured budget stands -- both for 0 and for a
            # negative. They used to behave differently: 0 was falsy and fell
            # through, a negative was clamped to the tier floor, and the warning
            # described only the second.
            assert output_budget("answer") == 4096
            assert output_budget("answer") == 4096
        assert caplog.text.count("LLMOverride.max_tokens") == 1, caplog.text
        assert "ignored" in caplog.text


def test_the_override_wins_over_the_environment() -> None:
    """A per-request budget is the channel the enterprise settings store uses."""
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 1024):
        assert output_budget("answer") == 1024
        with use_llm_override(LLMOverride(max_tokens=4096)):
            assert output_budget("answer") == 4096
            assert output_budget("correction") == 2048
        assert output_budget("answer") == 1024


def test_an_override_without_a_budget_leaves_the_environment_alone() -> None:
    """Every field of an override is optional; an unset one must not read as 0."""
    with (
        patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 2048),
        use_llm_override(LLMOverride(provider="litellm", model="m")),
    ):
        assert output_budget("answer") == 2048


def test_an_unknown_tier_is_refused_rather_than_silently_defaulted() -> None:
    """A typo that resolved to the answer budget would be invisible and expensive."""
    with pytest.raises(ValueError, match="Unknown budget tier"):
        output_budget("clasification")


def test_the_answer_tier_actually_reaches_the_call() -> None:
    """The tier the setting is named for must reach a request, not just exist.

    The first revision of this change defined an `answer` tier and wired nothing
    to it: the clients still carried `max_tokens: int = 1024` as a literal
    default, so `LLM_MAX_OUTPUT_TOKENS` moved the two derived tiers and left the
    answer -- the thing an operator raises it FOR -- pinned at the old value.
    Every other test here passed.

    A literal default cannot work: it is bound at import, and the budget is a
    runtime value. Hence `None`, resolved per call.
    """
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 7000):
        client = LiteLLMClient(model="m", api_key="k")
        with patch("litellm.completion") as completion:
            completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))], usage=None
            )
            client.complete("sys", "user")
        assert completion.call_args.kwargs["max_tokens"] == 7000


def test_an_explicit_budget_still_wins_over_the_tier() -> None:
    """A caller that names a number gets it; the tier is only the default."""
    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 7000):
        client = LiteLLMClient(model="m", api_key="k")
        with patch("litellm.completion") as completion:
            completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi"))], usage=None
            )
            client.complete("sys", "user", max_tokens=42)
        assert completion.call_args.kwargs["max_tokens"] == 42


def test_the_async_bridge_never_hands_a_subclass_none() -> None:
    """Widening the base signature is ours to do; changing the contract is not.

    `LLMClient.acomplete`'s default implementation forwards to `complete` in a
    thread. It used to forward the literal 1024. An out-of-tree client typed the
    way the in-repo doubles are -- `max_tokens: int = 1024` -- would take a
    forwarded `None` straight to its SDK, which rejects it.
    """
    seen: list[int | None] = []

    class OutOfTreeClient(LLMClient):
        def complete(self, system: object, user: str, max_tokens: int = 1024) -> str:  # type: ignore[override]
            seen.append(max_tokens)
            return "ok"

        def stream(self, system: object, user: str):  # type: ignore[override]
            yield "ok"

    with patch.object(config, "LLM_MAX_OUTPUT_TOKENS", 3333):
        asyncio.run(OutOfTreeClient().acomplete("sys", "user"))

    assert seen == [3333], f"the bridge forwarded {seen[0]!r}"


def test_extra_still_refuses_max_tokens() -> None:
    """The field exists precisely because `extra` may not carry it.

    `LiteLLMClient` rejects `max_tokens` in `extra` because the sync and async
    paths disagree about which value wins and one of them loses silently. Adding
    a first-class field must not have quietly reopened that door.
    """
    with pytest.raises(ValueError, match="max_tokens"):
        LiteLLMClient(model="m", extra={"max_tokens": 50})
