# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import anthropic

from nlqueries import config
from nlqueries.llm.client import LLMClient, SystemParam

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles on each retry (2**attempt)


class AnthropicClient(LLMClient):
    supports_prompt_caching = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model if model is not None else config.LLM_MODEL
        # An explicit key (e.g. a per-tenant key from the host app) overrides the
        # env-derived config default; api_base overrides the endpoint when given.
        key = api_key if api_key is not None else config.ANTHROPIC_API_KEY
        base_kwargs: dict[str, Any] = {"base_url": api_base} if api_base is not None else {}
        # Disable SDK-level retries so our own retry loop has full control.
        self._client = anthropic.Anthropic(api_key=key, max_retries=0, **base_kwargs)
        self._aclient = anthropic.AsyncAnthropic(api_key=key, max_retries=0, **base_kwargs)

    # ------------------------------------------------------------------
    # Helper: normalise system param into the list-of-blocks form that
    # the Anthropic API expects (enabling cache_control on structured inputs).
    # ------------------------------------------------------------------

    def _system_param(self, system: SystemParam) -> list[dict[str, Any]]:
        if isinstance(system, str):
            return [{"type": "text", "text": system}]
        return system

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, system: SystemParam, user: str, max_tokens: int = 1024) -> str:
        sys_blocks = self._system_param(system)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=cast(Any, sys_blocks),
                    messages=[{"role": "user", "content": user}],
                )
                return str(next(b.text for b in response.content if b.type == "text"))
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_DELAY * (2**attempt))
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, system: SystemParam, user: str) -> Iterator[str]:
        sys_blocks = self._system_param(system)
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=cast(Any, sys_blocks),
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream

    # ------------------------------------------------------------------
    # Native async API — no thread overhead, no event-loop blocking.
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        system: SystemParam,
        user: str,
        max_tokens: int = 1024,
        *,
        temperature: float | None = None,
    ) -> str:
        sys_blocks = self._system_param(system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": cast(Any, sys_blocks),
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._aclient.messages.create(**kwargs)
                return str(next(b.text for b in response.content if b.type == "text"))
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_BASE_DELAY * (2**attempt))
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        sys_blocks = self._system_param(system)
        async with self._aclient.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=cast(Any, sys_blocks),
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
