# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from nlqueries import config
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import LLMClient
from nlqueries.llm.litellm_client import LiteLLMClient

_REGISTRY: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "litellm": LiteLLMClient,
}


def get_llm_client(tier: str = "default") -> LLMClient:
    """Return an LLMClient instance for the configured provider.

    Args:
        tier: ``"fast"`` selects the cheap/fast model (``LLM_MODEL_FAST``) for
              short-output auxiliary calls such as intent classification and
              follow-up resolution.  Any other value uses the default model.

    Raises ValueError for unknown providers.
    """
    provider = config.LLM_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Available: {list(_REGISTRY)}")
    model = config.LLM_MODEL_FAST if tier == "fast" else config.LLM_MODEL
    return _REGISTRY[provider](model=model)
