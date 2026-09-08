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

    named = override.provider if override and override.provider else config.LLM_PROVIDER
    # "bedrock" is a name people configure, not a client. Normalised in the same
    # place for both channels, so an override built from a settings store that
    # holds the provider name behaves like the env var of the same value.
    provider = "litellm" if named.lower() == BEDROCK_PROVIDER else named
    named_bedrock = named.lower() == BEDROCK_PROVIDER or (
        config.LLM_PROVIDER_IS_BEDROCK and not (override and override.provider)
    )

    model: str | None = None
    if override is not None:
        model = (override.fast_model or override.model) if tier == "fast" else override.model
    if model is None:
        model = config.LLM_MODEL_FAST if tier == "fast" else config.LLM_MODEL

    # Bedrock is decided by the model, not only by the provider field, and this
    # has to happen before the client is built rather than at the API. An
    # AnthropicClient holding a `bedrock/...` id does not refuse it: it sends the
    # system prompt, the schema and the user's question to api.anthropic.com and
    # only then fails with a model-not-found. For a deployment that adopted
    # Bedrock to keep traffic inside its AWS account, the data has already left.
    model_is_bedrock = bool(model and model.startswith(BEDROCK_MODEL_PREFIX))
    if model_is_bedrock and provider != "litellm":
        if override is not None and override.provider:
            raise ValueError(
                f"provider={override.provider!r} contradicts model={model!r}. A "
                "bedrock/ model is reached through LiteLLM; naming another provider "
                "would send the request there instead."
            )
        # No provider was named, so the model names it. This mirrors the model
        # prefix check in ``config._detect_provider`` — the same rule, reached
        # through the override rather than the environment. It is the documented
        # IAM-role deployment: a model id, no credentials, nothing else.
        provider = "litellm"
    elif named_bedrock and not model_is_bedrock:
        # Deferred to here rather than raised while ``config`` imports. It used to
        # abort every CLI command, including `connect` and `extract-schema`, which
        # never touch an LLM — and the diagnostics someone would reach for to find
        # the misconfiguration. This is the first point where an LLM is actually
        # about to be used, so nothing has been sent yet and nothing else breaks.
        raise ValueError(
            f"Bedrock is configured but model={model!r} is not a Bedrock model. Set "
            "LLM_MODEL to a Bedrock id, e.g. "
            "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0. Naming the provider "
            "does not choose a model, and this one would be routed elsewhere."
        )

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
