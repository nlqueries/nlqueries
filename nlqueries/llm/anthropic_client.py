# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import anthropic

from nlqueries import config
from nlqueries.llm.client import (
    LLMClient,
    OutputBudgetExhausted,
    SystemParam,
    exhausted,
)
from nlqueries.llm.override import output_budget
from nlqueries.llm.usage import UsageRecord, record_usage

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles on each retry (2**attempt)


def _record_anthropic_usage(model: str, usage: Any) -> None:
    """Map an Anthropic ``usage`` object onto a UsageRecord and record it.

    ``input_tokens`` is the non-cached prompt; cache read/creation are the
    separately-priced prompt-cache halves (``None`` when caching is unused).
    Best-effort — a missing/odd usage shape never breaks the completion.
    """
    if usage is None:
        return
    with contextlib.suppress(Exception):
        record_usage(
            UsageRecord(
                model=model,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                estimated=False,
            )
        )


def _text_of(response: Any) -> str:
    """The reply's text, or ``""`` when it produced none.

    `next(b.text for b in response.content if b.type == "text")` was here, and a
    response with no text block made it raise `StopIteration` -- which is what a
    reasoning model returns when the whole allowance went to reasoning. Inside a
    generator that is worse than an error: PEP 479 turns it into a
    `RuntimeError` with nothing in it about budgets.
    """
    return "".join(b.text for b in response.content if b.type == "text")


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

    def complete(self, system: SystemParam, user: str, max_tokens: int | None = None) -> str:
        sys_blocks = self._system_param(system)
        budget = max_tokens or output_budget("answer")
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=budget,
                    system=cast(Any, sys_blocks),
                    messages=[{"role": "user", "content": user}],
                )
                _record_anthropic_usage(self._model, response.usage)
                text = _text_of(response)
                if exhausted(getattr(response, "stop_reason", None), text):
                    raise OutputBudgetExhausted(self._model, budget)
                return text
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_DELAY * (2**attempt))
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, system: SystemParam, user: str) -> Iterator[str]:
        sys_blocks = self._system_param(system)
        budget = output_budget("answer")
        collected: list[str] = []
        final: Any = None
        with self._client.messages.stream(
            model=self._model,
            max_tokens=budget,
            system=cast(Any, sys_blocks),
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                collected.append(text)
                yield text
            # After the caller drains the token stream, the final message carries
            # the authoritative usage (input/output + cache tokens).
            with contextlib.suppress(Exception):
                final = stream.get_final_message()
                _record_anthropic_usage(self._model, final.usage)
        # Outside the suppress, deliberately: raising inside it would be
        # swallowed by the block that exists to make usage recording
        # best-effort, and the caller would get an empty stream and no reason.
        if final is not None and exhausted(getattr(final, "stop_reason", None), "".join(collected)):
            raise OutputBudgetExhausted(self._model, budget)

    # ------------------------------------------------------------------
    # Native async API — no thread overhead, no event-loop blocking.
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        system: SystemParam,
        user: str,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
    ) -> str:
        sys_blocks = self._system_param(system)
        budget = max_tokens or output_budget("answer")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": budget,
            "system": cast(Any, sys_blocks),
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._aclient.messages.create(**kwargs)
                _record_anthropic_usage(self._model, response.usage)
                text = _text_of(response)
                if exhausted(getattr(response, "stop_reason", None), text):
                    raise OutputBudgetExhausted(self._model, budget)
                return text
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_BASE_DELAY * (2**attempt))
                else:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        sys_blocks = self._system_param(system)
        budget = output_budget("answer")
        collected: list[str] = []
        final: Any = None
        async with self._aclient.messages.stream(
            model=self._model,
            max_tokens=budget,
            system=cast(Any, sys_blocks),
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                collected.append(text)
                yield text
            with contextlib.suppress(Exception):
                final = await stream.get_final_message()
                _record_anthropic_usage(self._model, final.usage)
        # See the sync path: outside the suppress, or the raise is eaten.
        if final is not None and exhausted(getattr(final, "stop_reason", None), "".join(collected)):
            raise OutputBudgetExhausted(self._model, budget)
