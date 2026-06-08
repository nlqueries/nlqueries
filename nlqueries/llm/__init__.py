# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from nlqueries import config
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import LLMClient

_REGISTRY: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
}


def get_llm_client() -> LLMClient:
    """Return an LLMClient instance for the configured provider.

    Reads LLM_PROVIDER from config (default: "anthropic").
    Raises ValueError for unknown providers.
    """
    provider = config.LLM_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[provider]()
