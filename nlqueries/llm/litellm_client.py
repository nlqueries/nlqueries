# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from collections.abc import Iterator

import litellm

from nlqueries import config
from nlqueries.llm.client import LLMClient


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

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
