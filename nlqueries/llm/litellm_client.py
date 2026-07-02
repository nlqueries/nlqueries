# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import litellm

from nlqueries import config
from nlqueries.llm.client import LLMClient, SystemParam


def _flatten_system(system: SystemParam) -> str:
    """Convert a list of typed blocks to a plain string for providers that don't
    support Anthropic-style cache_control blocks."""
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if isinstance(b, dict))


class LiteLLMClient(LLMClient):
    """LLM client backed by LiteLLM — supports 100+ providers via a unified interface.

    The model name follows LiteLLM conventions: ``provider/model-id``.
    Examples::

        anthropic/claude-sonnet-4-5
        openai/gpt-4o
        gemini/gemini-1.5-pro
        ollama/llama3

    API keys are read from environment variables automatically by LiteLLM
    (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model if model is not None else config.LLM_MODEL

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, system: SystemParam, user: str, max_tokens: int = 1024) -> str:
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _flatten_system(system)},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, system: SystemParam, user: str) -> Iterator[str]:
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _flatten_system(system)},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Native async API
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        system: SystemParam,
        user: str,
        max_tokens: int = 1024,
        *,
        temperature: float | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _flatten_system(system)},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        response = await litellm.acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": _flatten_system(system)},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
