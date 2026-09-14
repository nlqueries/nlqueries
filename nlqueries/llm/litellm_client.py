# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import litellm
import litellm.exceptions

from nlqueries import config
from nlqueries.llm.client import (
    TRUNCATED,
    LLMClient,
    LLMTimeout,
    OutputBudgetExhausted,
    SystemParam,
    exhausted,
    looks_like_timeout,
)
from nlqueries.llm.override import output_budget
from nlqueries.llm.usage import UsageRecord, estimate_tokens, record_usage


@contextlib.contextmanager
def _deadline(model: str, seconds: object) -> Iterator[None]:
    """A timed-out call -> :class:`LLMTimeout`, so a host catches one type.

    Matched three ways: litellm's own ``Timeout``, ``httpx.TimeoutException``,
    and anything else httpx-shaped by class name. The name check is not
    redundant -- a stall part-way through a stream is raised by the transport
    rather than mapped on the way out, and which httpx distribution that is
    depends on what the environment resolved, so an ``isinstance`` against the
    one imported here is true in some environments and false in others.

    Narrow otherwise: every other litellm error keeps its own type and message,
    which are more informative than anything this could substitute.

    ``seconds`` is typed ``object`` because it is whatever ``_call_kwargs``
    resolved, and a host may have put a ``str`` or an ``httpx.Timeout`` in
    ``extra``. :class:`LLMTimeout` renders it accordingly rather than assuming
    a number.

    ``None`` means this process set no deadline -- not that there is none.
    litellm then falls back to ``COMPLETION_HTTP_FALLBACK_SECONDS`` (600.0).
    With no limit of ours to name, the provider's own error goes through
    untouched rather than being relabelled with one nobody set.
    """
    try:
        yield
    except Exception as exc:
        if not (
            isinstance(exc, (litellm.exceptions.Timeout, httpx.TimeoutException))
            or looks_like_timeout(exc)
        ):
            raise
        if seconds is None:
            raise
        raise LLMTimeout(model, seconds) from exc


def _flatten_system(system: SystemParam) -> str:
    """Convert a list of typed blocks to a plain string for providers that don't
    support Anthropic-style cache_control blocks."""
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if isinstance(b, dict))


def _record_litellm_usage(model: str, usage: Any) -> None:
    """Record exact usage from an OpenAI-style ``usage`` object (LiteLLM).

    ``prompt_tokens`` includes any cached tokens, so the cached count (when the
    provider reports it under ``prompt_tokens_details``) is split out into
    ``cache_read_tokens`` and subtracted from the regular input. Best-effort.
    """
    if usage is None:
        return
    with contextlib.suppress(Exception):
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        record_usage(
            UsageRecord(
                model=model,
                input_tokens=max(0, prompt - cached),
                output_tokens=completion,
                cache_read_tokens=cached,
                cache_write_tokens=0,
                estimated=False,
            )
        )


def _record_estimated(model: str, prompt_text: str, output_text: str) -> None:
    """Record a heuristic usage estimate (flagged) when a provider omits usage —
    e.g. the streaming path, where LiteLLM usage is provider-dependent."""
    with contextlib.suppress(Exception):
        record_usage(
            UsageRecord(
                model=model,
                input_tokens=estimate_tokens(prompt_text),
                output_tokens=estimate_tokens(output_text),
                estimated=True,
            )
        )


#: Names this class already passes to ``litellm.(a)completion``, and therefore
#: the names ``extra`` may not carry.
#:
#: Rejected at construction because the collision is otherwise inconsistent and
#: half of it is silent. ``_call_kwargs()`` is spread into the call directly in
#: the sync and streaming paths, where a duplicate keyword is a loud
#: ``TypeError``; in ``acomplete`` it is merged into a dict literal *after* these
#: keys, where it quietly wins. A host that put ``max_tokens`` in ``extra`` would
#: get an exception from one method and a silently capped answer from the other.
#: Core does not otherwise inspect ``extra`` — this is the one constraint it has
#: to enforce, because it is the one it creates.
#:
#: ``timeout`` is deliberately absent from this set. It is not a name this class
#: owns: a host that sets one in ``extra`` is making a per-client choice, and
#: ``_call_kwargs()`` lets it win rather than rejecting it.
_RESERVED_COMPLETION_KWARGS = frozenset(
    {"model", "messages", "max_tokens", "stream", "temperature"}
)


class LiteLLMClient(LLMClient):
    """LLM client backed by LiteLLM — supports 100+ providers via a unified interface.

    The model name follows LiteLLM conventions: ``provider/model-id``.
    Examples::

        anthropic/claude-sonnet-4-5
        openai/gpt-4o
        gemini/gemini-1.5-pro
        ollama/llama3
        bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0

    API keys are read from environment variables automatically by LiteLLM
    (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.). An explicit
    ``api_key``/``api_base`` (e.g. a per-tenant key resolved by the host app)
    overrides the environment for this client's calls; when ``None`` LiteLLM's
    env-based resolution is used unchanged.

    Some providers are not configured by an API key at all. Amazon Bedrock
    authenticates through boto3, which needs a region and either explicit
    credentials or the host's IAM role. Those arrive in ``extra`` and are
    forwarded to LiteLLM untouched, so this class stays free of any one cloud's
    vocabulary while still supporting it.

    ``extra`` may not carry the names this class passes itself —
    ``model``, ``messages``, ``max_tokens``, ``stream``, ``temperature`` — and
    the constructor rejects them rather than letting the collision through. See
    :data:`_RESERVED_COMPLETION_KWARGS`.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if extra:
            clashing = _RESERVED_COMPLETION_KWARGS & extra.keys()
            if clashing:
                raise ValueError(
                    "LiteLLMClient.extra may not contain "
                    f"{sorted(clashing)}: this class passes those to the completion "
                    "call itself. Use the constructor arguments (model=, api_key=, "
                    "api_base=) or the per-call arguments instead."
                )
        self._model = model if model is not None else config.LLM_MODEL
        self._api_key = api_key
        self._api_base = api_base
        self._extra = extra

    def _auth_kwargs(self) -> dict[str, Any]:
        """Per-call kwargs: ``extra`` first, then api_key/api_base when set.

        ``extra`` holds whatever the caller's provider needs and core does not
        model — ``aws_region_name`` and the AWS credential kwargs for Bedrock,
        say. It is forwarded verbatim, including any ``None`` values: LiteLLM
        reads a missing AWS credential as "use the boto3 chain", so it is the
        caller's business whether to send one, not ours to second-guess.

        ``api_key``/``api_base`` are applied last so an explicit key always wins
        over one that happened to arrive in ``extra`` under the same name. The
        copy matters: the dict belongs to the caller, and every call would
        otherwise accumulate into it.
        """
        kwargs: dict[str, Any] = dict(self._extra) if self._extra else {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        return kwargs

    def _call_kwargs(self) -> dict[str, Any]:
        """Everything spread into a completion call: auth, ``extra``, deadline.

        Separate from :meth:`_auth_kwargs` because a deadline is not credentials,
        and that method's contract -- asserted by exact equality in
        ``test_llm_override.py`` -- is the auth-related keys and nothing else.

        A host's own deadline in ``extra`` wins, under either of litellm's two
        spellings: ``timeout``, via ``setdefault``, and ``request_timeout``,
        which ``CompletionTimeout.resolve`` consults after it. Supplying
        ``timeout`` unconditionally would silently override a host that
        configured the other name. Neither is in
        ``_RESERVED_COMPLETION_KWARGS``, because rejecting them would break a
        host already setting one.

        **A bare float, unlike the Anthropic client**, which builds
        ``anthropic.Timeout(seconds, connect=5.0)`` so an unreachable endpoint
        fails in five seconds rather than three minutes. litellm applies a
        float to every phase, so that fast-fail is absent here -- on the path
        carrying OpenAI, Gemini, Bedrock and Ollama -- and ``_deadline``
        reports ``phase="read"`` because there is no per-phase object to read
        one from.

        Deliberate. litellm types this argument ``float | str |
        openai.Timeout | None``: *openai's* re-export, not the ``httpx``
        imported here, and core declares neither package. Passing a rich object
        would tie this line to which distribution the environment resolved, to
        buy a faster failure on a misconfiguration.
        """
        kwargs = self._auth_kwargs()
        if "request_timeout" not in kwargs:
            kwargs.setdefault("timeout", config.LLM_TIMEOUT_SECONDS)
        return kwargs

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, system: SystemParam, user: str, max_tokens: int | None = None) -> str:
        budget = max_tokens or output_budget("answer")
        auth = self._call_kwargs()
        with _deadline(self._model, auth.get("timeout")):
            response = litellm.completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _flatten_system(system)},
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                **auth,
            )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            _record_litellm_usage(self._model, usage)
        else:
            _record_estimated(self._model, f"{_flatten_system(system)}\n{user}", content)
        # Usage is recorded first: the tokens were spent whether or not anything
        # came back, and a deployment tracking cost should see them.
        if exhausted(getattr(response.choices[0], "finish_reason", None), content):
            raise OutputBudgetExhausted(self._model, budget)
        return content

    def stream(self, system: SystemParam, user: str) -> Iterator[str]:
        budget = output_budget("answer")
        auth = self._call_kwargs()
        # The iteration is inside the deadline too, not just the call that opens
        # the stream. litellm applies the timeout per chunk read, so a provider
        # that accepts the request and then stalls raises here -- which is the
        # shape of the hang this exists for, and the one a wrapper around the
        # opening call alone would miss.
        with _deadline(self._model, auth.get("timeout")):
            response = litellm.completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _flatten_system(system)},
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                stream=True,
                **auth,
            )
            collected: list[str] = []
            finish: object = None
            for chunk in response:
                finish = getattr(chunk.choices[0], "finish_reason", None) or finish
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta
        # Streaming usage is provider-dependent in LiteLLM; record an estimate.
        _record_estimated(self._model, f"{_flatten_system(system)}\n{user}", "".join(collected))
        # The finish reason arrives on the last chunk, so this can only be
        # decided once the stream is done -- and only when nothing at all was
        # yielded. `collected` empty, not blank: `exhausted` treats whitespace
        # as nothing, which is right for a reply returned whole and wrong here.
        # A reasoning model can emit a leading newline before it hits the cap,
        # and raising then would land mid-stream on a caller that has already
        # begun writing a response body and cannot unwrite it. Whitespace is
        # delivered as-is; a blank answer beats a broken stream.
        if not collected and str(finish) in TRUNCATED:
            raise OutputBudgetExhausted(self._model, budget)

    # ------------------------------------------------------------------
    # Native async API
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        system: SystemParam,
        user: str,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
    ) -> str:
        budget = max_tokens or output_budget("answer")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _flatten_system(system)},
                {"role": "user", "content": user},
            ],
            "max_tokens": budget,
            **self._call_kwargs(),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        with _deadline(self._model, kwargs.get("timeout")):
            response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            _record_litellm_usage(self._model, usage)
        else:
            _record_estimated(self._model, f"{_flatten_system(system)}\n{user}", content)
        if exhausted(getattr(response.choices[0], "finish_reason", None), content):
            raise OutputBudgetExhausted(self._model, budget)
        return content

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        budget = output_budget("answer")
        auth = self._call_kwargs()
        # See the sync path: the iteration is inside the deadline as well.
        with _deadline(self._model, auth.get("timeout")):
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _flatten_system(system)},
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                stream=True,
                **auth,
            )
            collected: list[str] = []
            finish: object = None
            async for chunk in response:
                finish = getattr(chunk.choices[0], "finish_reason", None) or finish
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta
        _record_estimated(self._model, f"{_flatten_system(system)}\n{user}", "".join(collected))
        # See the sync path: empty, not blank.
        if not collected and str(finish) in TRUNCATED:
            raise OutputBudgetExhausted(self._model, budget)
