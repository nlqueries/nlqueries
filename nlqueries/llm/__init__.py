# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from nlqueries import config
from nlqueries.config import BEDROCK_MODEL_PREFIX, BEDROCK_PROVIDER
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import LLMClient
from nlqueries.llm.litellm_client import LiteLLMClient
from nlqueries.llm.usage import (
    UsageRecord,
    current_usage_sink,
    estimate_tokens,
    record_usage,
    use_usage_sink,
)

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "LLMOverride",
    "LiteLLMClient",
    "UsageRecord",
    "current_llm_override",
    "current_usage_sink",
    "estimate_tokens",
    "get_llm_client",
    "record_usage",
    "use_llm_override",
    "use_usage_sink",
]

_REGISTRY: dict[str, Any] = {
    "anthropic": AnthropicClient,
    "litellm": LiteLLMClient,
}


@dataclass(frozen=True)
class LLMOverride:
    """Per-invocation LLM configuration that supersedes the process env/config.

    Every field is optional; a ``None`` field falls back to the env-derived
    ``nlqueries.config`` default, so a partial override (e.g. only ``api_key``)
    is fine. The host application (e.g. the enterprise layer resolving a
    per-tenant key from its settings store) sets one for the duration of a
    request via :func:`use_llm_override`; :func:`get_llm_client` reads it.

    ``extra`` carries provider-specific keyword arguments that this package has
    no opinion about — AWS region and credentials for a Bedrock deployment, for
    instance. They are passed verbatim to the underlying completion call, which
    keeps cloud-provider vocabulary out of core: the host application knows what
    its provider needs, and core only has to carry it. Note that supplying it
    makes the instance unhashable, as a dict field does to any frozen dataclass;
    nothing hashes an override today.
    """

    provider: str | None = None
    model: str | None = None
    fast_model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    extra: dict[str, Any] | None = None
    #: Tokens an answer may generate, overriding ``config.LLM_MAX_OUTPUT_TOKENS``.
    #:
    #: A first-class field rather than a key in ``extra`` because ``extra`` may
    #: not carry it: ``LiteLLMClient`` rejects ``max_tokens`` there, since the
    #: two code paths disagree about which value wins and one of them loses
    #: silently. That refusal is right, and it left a host with no way at all to
    #: set a budget -- which is what this field is for.
    max_tokens: int | None = None


# Task-local so concurrent requests on one event loop never see each other's
# override: contextvars are copied per asyncio Task, and the value set here
# propagates through every ``await`` in the same task (where get_llm_client is
# called) without leaking across tasks.
_override: ContextVar[LLMOverride | None] = ContextVar("nlqueries_llm_override", default=None)


@contextlib.contextmanager
def use_llm_override(override: LLMOverride | None) -> Iterator[None]:
    """Bind *override* for the duration of the ``with`` block (task-local).

    Passing ``None`` is a no-op binding (the process default keeps applying),
    which lets callers wrap a block unconditionally.
    """
    token = _override.set(override)
    try:
        yield
    finally:
        _override.reset(token)


def current_llm_override() -> LLMOverride | None:
    """Return the override bound in the current context, if any."""
    return _override.get()


#: What each kind of call gets, as a share of the answer budget and a floor.
#:
#: One configurable number, several call sites that need less than an answer.
#: The shares take over as an operator raises the budget; the floors are what
#: hold at and below the default. Without the shares, raising the budget for a
#: reasoning model would fix the answer and leave classification starved at 200
#: -- and that is the tier that returns an EMPTY string rather than a short one,
#: because the reasoning is billed first.
#:
#: Both shares are exact at the default of 1024, so an operator who sets nothing
#: gets 1024 / 512 / 200: precisely the numbers these call sites hard-coded
#: before this was configurable. That is worth arithmetic rather than
#: approximation -- a first attempt used 0.2 for classification, which is 204 at
#: the default, and the test written to assert "the default changes nothing"
#: caught it. A fifth is not a no-op; an eighth is.
_BUDGET_TIERS: dict[str, tuple[float, int]] = {
    # The answer itself: the whole budget.
    "answer": (1.0, 0),
    # A retry that rewrites SQL, or a candidate set. Structured, bounded output.
    "correction": (0.5, 512),
    # A label or a short JSON object: intent, and follow-up resolution.
    "classification": (0.125, 200),
}


def output_budget(tier: str = "answer") -> int:
    """Tokens this kind of call may generate.

    Resolved from the bound :class:`LLMOverride` when it names a budget, else
    from ``config.LLM_MAX_OUTPUT_TOKENS``.

    Callers pass a tier rather than a number so the policy lives here. The
    numbers they used to pass -- 1024, 512, 200 -- were chosen against models
    that emit only the answer; a reasoning model spends the same allowance on
    reasoning first, and the smallest budgets are the ones that then return
    nothing at all.
    """
    if tier not in _BUDGET_TIERS:
        raise ValueError(f"Unknown budget tier: {tier!r}. Available: {sorted(_BUDGET_TIERS)}")
    override = _override.get()
    total = (
        override.max_tokens if override and override.max_tokens else config.LLM_MAX_OUTPUT_TOKENS
    )
    share, floor = _BUDGET_TIERS[tier]
    return max(floor, int(total * share))


def get_llm_client(tier: str = "default") -> LLMClient:
    """Return an LLMClient instance for the configured provider.

    Args:
        tier: ``"fast"`` selects the cheap/fast model (``LLM_MODEL_FAST``) for
              short-output auxiliary calls such as intent classification and
              follow-up resolution.  Any other value uses the default model.

    When an :class:`LLMOverride` is bound via :func:`use_llm_override`, its
    non-``None`` fields take precedence over ``nlqueries.config``; otherwise the
    behavior is exactly the env-derived default.

    Raises ValueError for unknown providers.
    """
    override = _override.get()

    # What, if anything, *named* a provider — the override first, then the raw
    # LLM_PROVIDER. Empty means nothing did and detection chose, which is a
    # different situation from an operator naming one, and the resolved
    # ``config.LLM_PROVIDER`` cannot tell them apart.
    named = override.provider if override and override.provider else config.LLM_PROVIDER_CONFIGURED
    names_bedrock = named.lower() == BEDROCK_PROVIDER
    # "bedrock" is a name people configure, not a client. Normalised in the same
    # place for both channels, so an override built from a settings store that
    # holds the provider name behaves like the env var of the same value.
    provider = (
        "litellm"
        if names_bedrock
        else (override.provider if override and override.provider else config.LLM_PROVIDER)
    )

    model: str | None = None
    if override is not None:
        model = (override.fast_model or override.model) if tier == "fast" else override.model
    if model is None:
        model = config.LLM_MODEL_FAST if tier == "fast" else config.LLM_MODEL

    # Whether this *deployment* is on Bedrock, judged from the default model
    # rather than from whichever model this tier happened to resolve. Judging per
    # tier let the two tiers disagree: a `.env` still carrying
    # LLM_MODEL_FAST=claude-haiku-... from a previous Anthropic setup left the
    # default tier on Bedrock while every auxiliary call — intent classification
    # and follow-up resolution, which carry the question and the conversation
    # history — went to api.anthropic.com under the leftover key, with no error.
    default_model = override.model if override and override.model else config.LLM_MODEL
    on_bedrock = names_bedrock or default_model.startswith(BEDROCK_MODEL_PREFIX)
    model_is_bedrock = bool(model and model.startswith(BEDROCK_MODEL_PREFIX))

    if on_bedrock and not model_is_bedrock:
        # Refused here rather than while ``config`` imports: this used to abort
        # every CLI command, including `connect`, `extract-schema` and the
        # diagnostics someone would reach for to find the misconfiguration. This
        # is the first point at which an LLM is about to be used, so nothing has
        # been sent yet and nothing else breaks.
        setting = "LLM_MODEL_FAST" if tier == "fast" else "LLM_MODEL"
        raise ValueError(
            f"Bedrock is configured, but the {tier} model {model!r} is not a Bedrock "
            f"model. Set {setting} to a Bedrock id, e.g. "
            "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0. A non-Bedrock model "
            "here is routed to its own provider, which sends the request outside "
            "your AWS account."
        )

    if model_is_bedrock and named and not names_bedrock and named.lower() != "litellm":
        # Two settings naming different providers. Refused through either
        # channel: `LLM_PROVIDER=anthropic` alongside a `bedrock/` model is the
        # same mistake as an override that says so, and previously only the
        # override was caught.
        raise ValueError(
            f"provider={named!r} contradicts model={model!r}. A bedrock/ model is "
            "reached through LiteLLM; naming another provider would send the "
            "request there instead."
        )

    # Nothing named a provider, so the model names it — the same rule
    # ``config._detect_provider`` applies to the environment, reached through the
    # override. This is the documented IAM-role deployment: a model id, no
    # credentials, nothing else. Without it an `AnthropicClient` is built, and it
    # does not reject a `bedrock/` id — it transmits the prompt, the schema and
    # the question to api.anthropic.com and only then reports the model missing.
    if model_is_bedrock:
        provider = "litellm"

    if provider not in _REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Available: {list(_REGISTRY)}")

    api_key = override.api_key if override else None
    api_base = override.api_base if override else None
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "api_base": api_base}
    if provider == "litellm":
        # ``extra`` is only meaningful to the client that forwards it to a
        # multi-provider SDK; a client with a narrow constructor would raise a
        # TypeError on it.
        kwargs["extra"] = override.extra if override else None
    elif override is not None and override.extra:
        # Dropping it silently is the dangerous outcome, not the safe one. An
        # override carrying AWS credentials describes a Bedrock call; with no
        # ``provider`` set it resolves to ``config.LLM_PROVIDER`` — ``anthropic``
        # on a default install — and the request would go out to the public
        # Anthropic API under the process-level key, which is the exact egress a
        # deployment chose Bedrock to avoid. Nothing would report it: the answer
        # comes back correct, from the wrong place.
        raise ValueError(
            f"LLMOverride.extra was supplied, but provider {provider!r} cannot accept it. "
            "extra carries provider-specific arguments (AWS region and credentials, "
            'say) that only the LiteLLM client forwards. Set provider="litellm" on '
            "the override, or drop extra — leaving both would send the request to "
            f"{provider!r} without them."
        )
    return cast(LLMClient, _REGISTRY[provider](**kwargs))
