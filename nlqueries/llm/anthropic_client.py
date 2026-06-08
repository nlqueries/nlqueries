# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import time
from collections.abc import Iterator

import anthropic

from nlqueries import config
from nlqueries.llm.client import LLMClient

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles on each retry (2**attempt)


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self._model = model if model is not None else config.LLM_MODEL
        # Disable SDK-level retries so our own retry loop has full control.
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=0)

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return next(b.text for b in response.content if b.type == "text")
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_DELAY * (2**attempt))
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream
