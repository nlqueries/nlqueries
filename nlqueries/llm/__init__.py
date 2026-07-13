# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from nlqueries import config
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import LLMClient
from nlqueries.llm.litellm_client import LiteLLMClient

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
    """

    provider: str | None = None
    model: str | None = None
    fast_model: str | None = None
    api_key: str | None = None
    api_base: str | None = None


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

    provider = override.provider if override and override.provider else config.LLM_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Available: {list(_REGISTRY)}")

    model: str | None = None
    if override is not None:
        model = (override.fast_model or override.model) if tier == "fast" else override.model
    if model is None:
        model = config.LLM_MODEL_FAST if tier == "fast" else config.LLM_MODEL

    api_key = override.api_key if override else None
    api_base = override.api_base if override else None
    return cast(LLMClient, _REGISTRY[provider](model=model, api_key=api_key, api_base=api_base))
