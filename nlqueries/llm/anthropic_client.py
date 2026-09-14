# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import anthropic
import httpx

from nlqueries import config
from nlqueries.llm.client import (
    TRUNCATED,
    LLMClient,
    LLMTimeout,
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


#: Exception class names every httpx-shaped transport uses for a deadline.
#: Compared by name because the SDK's httpx is not necessarily the one imported
#: here; see ``_deadline``.
_TIMEOUT_NAMES = frozenset(
    {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
)


def _is_timeout(exc: BaseException) -> bool:
    """Whether *exc* is a deadline, whichever httpx the SDK was built against."""
    if isinstance(exc, (anthropic.APITimeoutError, httpx.TimeoutException)):
        return True
    return any(t.__name__ in _TIMEOUT_NAMES for t in type(exc).__mro__)


@contextlib.contextmanager
def _deadline(model: str, seconds: object) -> Iterator[None]:
    """A timed-out call -> :class:`LLMTimeout`.

    The mirror of the one in ``litellm_client``, so a host catches a single
    type whichever client it ended up with. Narrow on purpose: every other
    Anthropic error keeps its own type and message.

    The transport's own exception as well as the SDK's, because the SDK
    converts one into the other only around ``self._client.send(...)`` in
    ``_base_client._request``. With ``stream=True`` that call returns as soon
    as the headers arrive and the body is read lazily afterwards, outside the
    ``try`` -- and neither ``_streaming`` nor ``lib/streaming/_messages``
    handles a timeout at all. So a provider that accepts the request and then
    stalls raises a raw read timeout while ``text_stream`` is being drained,
    which is exactly the hang this exists for and exactly the shape
    ``APITimeoutError`` alone would miss.

    Matched by name as well as by class, and that is not laziness. Which httpx
    the SDK raises from depends on its version -- newer anthropic builds
    against ``httpx2`` -- so ``isinstance(exc, httpx.TimeoutException)`` is
    true here and false in an environment that resolves the other one, with
    nothing in this repository having changed. CI found the matching version
    split on the constructor argument. The name check costs nothing and does
    not depend on which package won.
    """
    try:
        yield
    except Exception as exc:
        if not _is_timeout(exc):
            raise
        raise LLMTimeout(model, seconds) from exc


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
        # On the SDK client rather than per call: the Anthropic SDK applies it
        # to every request made through it, including the ones inside a stream,
        # so there is no path that can be added later and quietly miss it.
        self._timeout = config.LLM_TIMEOUT_SECONDS
        # `anthropic.Timeout`, not `httpx.Timeout`: the SDK re-exports the type
        # it accepts, and which httpx that is depends on the version -- newer
        # anthropic builds against `httpx2`, and CI caught `httpx._config.Timeout`
        # being rejected where `httpx2._config.Timeout` was expected. Locally the
        # two are the same object, which is exactly why the local run did not.
        #
        # Not the bare float either: a float sets EVERY phase,
        # connect included, and the SDK's own default is
        # `Timeout(connect=5.0, read=600, write=600, pool=600)`. Handing it
        # 180.0 would have moved connect from 5s to 180s, so an unreachable
        # endpoint -- blocked egress, a mistyped api_base -- would sit for
        # three minutes instead of five seconds, three times over for a
        # question that classifies, generates and corrects. The read deadline
        # is what this change is for; the connect one was already right.
        self._httpx_timeout = anthropic.Timeout(self._timeout, connect=5.0)
        # Disable SDK-level retries so our own retry loop has full control.
        self._client = anthropic.Anthropic(
            api_key=key, max_retries=0, timeout=self._httpx_timeout, **base_kwargs
        )
        self._aclient = anthropic.AsyncAnthropic(
            api_key=key, max_retries=0, timeout=self._httpx_timeout, **base_kwargs
        )

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
                with _deadline(self._model, self._timeout):
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
        # The deadline wraps the iteration, not just the call that opens the
        # stream: a provider that accepts the request and then stalls is the
        # hang this exists for, and it surfaces while draining `text_stream`.
        with (
            _deadline(self._model, self._timeout),
            self._client.messages.stream(
                model=self._model,
                max_tokens=budget,
                system=cast(Any, sys_blocks),
                messages=[{"role": "user", "content": user}],
            ) as stream,
        ):
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
        if (
            final is not None
            and not collected
            and str(getattr(final, "stop_reason", None)) in TRUNCATED
        ):
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
                with _deadline(self._model, self._timeout):
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
        # See the sync path: the drain is inside the deadline too.
        with _deadline(self._model, self._timeout):
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
        if (
            final is not None
            and not collected
            and str(getattr(final, "stop_reason", None)) in TRUNCATED
        ):
            raise OutputBudgetExhausted(self._model, budget)
