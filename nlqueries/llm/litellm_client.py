# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import litellm
import litellm.exceptions

from nlqueries import config
from nlqueries.llm.claude import accepts_temperature, effort_for
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


def _configured_deadline(kwargs: dict[str, Any]) -> object:
    """The deadline litellm will apply, in litellm's own resolution order.

    ``timeout`` then ``request_timeout``, first non-``None`` winning, which is
    what ``CompletionTimeout.resolve`` does with them. Both are read because a
    host may set either in ``extra``, and reading only the first leaves the
    host who chose the other spelling with an untranslated provider error --
    the one configuration where a number was available to name.

    ``None`` therefore means what the docstring below says it means: this
    process set no deadline.
    """
    for key in ("timeout", "request_timeout"):
        value = kwargs.get(key)
        if value is not None:
            return value
    return None


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


def _caches_prompts(model: str) -> bool:
    """Whether to send Anthropic-style ``cache_control`` blocks for *model*.

    Claude on Amazon Bedrock only (owner, 2026-10-05): LiteLLM turns a
    ``cache_control`` on a system content block into Bedrock's ``cachePoint``,
    and Bedrock then serves the stable prefix -- the schema and instructions
    the SQL prompt repeats for every question -- at the cache-read rate. Other
    providers on this path either cache on their own (OpenAI) or are not
    Claude, so they keep the flattened system prompt they always had. The
    model must also be one LiteLLM knows to support prompt caching.
    """
    if not model.startswith("bedrock/"):
        return False
    bare = model[len("bedrock/") :]
    for route in ("converse/", "invoke/"):
        bare = bare.removeprefix(route)
    # A cross-region inference profile puts its geography first: us., eu.,
    # apac., global. -- the model family follows it.
    family = bare.split(".", 1)[1] if bare.split(".", 1)[0] in _PROFILE_PREFIXES else bare
    if not family.startswith("anthropic."):
        return False
    with contextlib.suppress(Exception):
        return bool(litellm.utils.supports_prompt_caching(model=model))
    return False


_PROFILE_PREFIXES = frozenset({"us", "eu", "apac", "global", "us-gov", "ca", "jp", "au"})


def _reaches_anthropic_api(model: str) -> bool:
    """Whether LiteLLM sends *model* to the Anthropic API in its own shape.

    ``anthropic/claude-...``, and a bare ``claude-...``, which LiteLLM routes
    there too. Not ``bedrock/``, ``vertex_ai/`` or ``openrouter/``: each takes
    an effort in its own request shape, and none of those has been checked.
    """
    return model.startswith(("anthropic/", "claude-"))


#: DeepSeek's ``reasoning_effort`` for each ``LLM_EFFORT`` level. DeepSeek takes
#: ``low``, ``high`` and ``max``: ``medium`` has no level of its own and goes
#: down to ``low``, ``xhigh`` up to ``max``.
_DEEPSEEK_EFFORT = {"low": "low", "medium": "low", "high": "high", "xhigh": "max", "max": "max"}


def _deepseek_body(model: str) -> dict[str, Any]:
    """The request-body fields DeepSeek's thinking takes, for a ``deepseek/`` id.

    Sent through ``extra_body``, which the OpenAI SDK merges into the JSON body.
    A top-level ``reasoning_effort`` does not survive LiteLLM's DeepSeek
    mapping: it is turned into ``thinking`` on or off and the level is dropped
    (LiteLLM 1.104.2, ``DeepSeekChatConfig.map_openai_params``). ``thinking``
    would reach the body either way; it travels with the effort for one route.
    See :data:`nlqueries.config.LLM_EFFORT` and :data:`nlqueries.config.LLM_THINKING`.
    """
    if not model.startswith("deepseek/"):
        return {}
    if not config.LLM_THINKING:
        return {"thinking": {"type": "disabled"}}
    effort = _DEEPSEEK_EFFORT.get(config.LLM_EFFORT or "")
    return {"reasoning_effort": effort} if effort else {}


def _streams_usage(model: str) -> bool:
    """Whether to ask *model*'s stream for its usage (``stream_options``).

    On OpenAI-compatible routes only, as LiteLLM lists them, and only where
    LiteLLM says the route accepts ``stream_options``. Without it such a stream
    ends with no usage at all, and the record has to be an estimate.
    """
    with contextlib.suppress(Exception):
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        compatible = getattr(litellm, "openai_compatible_providers", ())
        if provider != "openai" and provider not in compatible:
            return False
        return "stream_options" in (litellm.get_supported_openai_params(model=model) or [])
    return False


def _system_message(system: SystemParam, *, keep_blocks: bool) -> dict[str, Any]:
    """The system message for a LiteLLM call.

    With *keep_blocks* and a list of typed blocks, the blocks go through as
    the message content -- ``cache_control`` and all -- for LiteLLM to turn
    into the provider's cache marker; otherwise the text, flattened.
    """
    if keep_blocks and not isinstance(system, str):
        return {"role": "system", "content": [dict(b) for b in system if isinstance(b, dict)]}
    return {"role": "system", "content": _flatten_system(system)}


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
    ``cache_read_tokens`` and subtracted from the regular input. So are tokens
    written to the cache (``cache_creation_tokens``, which Bedrock reports),
    into ``cache_write_tokens``: they are billed at their own rate, not as
    plain input.

    ``completion_tokens`` already includes any reasoning, as OpenAI-style
    usage counts it, so it is the output as billed. The reasoning part, when
    reported under ``completion_tokens_details``, is also recorded on its own.
    Best-effort.
    """
    if usage is None:
        return
    with contextlib.suppress(Exception):
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        written = int(getattr(details, "cache_creation_tokens", 0) or 0) if details else 0
        produced = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(produced, "reasoning_tokens", 0) or 0) if produced else 0
        record_usage(
            UsageRecord(
                model=model,
                input_tokens=max(0, prompt - cached - written),
                output_tokens=completion,
                cache_read_tokens=cached,
                cache_write_tokens=written,
                estimated=False,
                reasoning_tokens=reasoning,
            )
        )


def _record_stream_usage(
    model: str, usage: Any, system: SystemParam, user: str, collected: list[str]
) -> None:
    """The usage a stream reported in its last chunk, or an estimate without one."""
    if usage is not None:
        _record_litellm_usage(model, usage)
    else:
        _record_estimated(model, f"{_flatten_system(system)}\n{user}", "".join(collected))


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
#: mostly silent. ``extra`` reaches every call through :meth:`_request_kwargs`,
#: which merges it over this model's own arguments, so a ``temperature`` or
#: ``output_config`` in ``extra`` would quietly replace this class's on every
#: path. ``model``, ``messages`` and ``max_tokens`` (and ``stream`` on the
#: streaming calls) are keywords of the call itself: in ``complete``, ``stream``
#: and ``astream`` a duplicate is a loud ``TypeError``, while ``acomplete`` puts
#: them in a dict literal *before* ``extra``, which quietly wins. A host that put
#: ``max_tokens`` in ``extra`` would get an exception from one method and a
#: silently capped answer from another. Core does not otherwise inspect
#: ``extra``: this is the one constraint it has to enforce, because it is the one
#: it creates.
#:
#: ``timeout`` is deliberately absent from this set. It is not a name this class
#: owns: a host that sets one in ``extra`` is making a per-client choice, and
#: ``_call_kwargs()`` lets it win rather than rejecting it.
_RESERVED_COMPLETION_KWARGS = frozenset(
    {"model", "messages", "max_tokens", "stream", "temperature", "output_config"}
)


class LiteLLMClient(LLMClient):
    """LLM client backed by LiteLLM — supports 100+ providers via a unified interface.

    The model name follows LiteLLM conventions: ``provider/model-id``.
    Examples::

        anthropic/claude-sonnet-5-5
        openai/gpt-4o
        gemini/gemini-1.5-pro
        ollama/llama3
        bedrock/us.anthropic.claude-sonnet-4-6

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
    ``model``, ``messages``, ``max_tokens``, ``stream``, ``temperature``,
    ``output_config`` — and the constructor rejects them rather than letting
    the collision through. See :data:`_RESERVED_COMPLETION_KWARGS`.
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
        # Per model, not per class: on this path only Claude on Bedrock takes
        # cache markers (see _caches_prompts). The orchestrator reads it to
        # decide whether to mark the prompt's stable block.
        self.supports_prompt_caching = _caches_prompts(self._model)
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
        openai.Timeout | None``, and whether that re-export is the same class
        as the ``httpx`` imported here depends on what the environment
        resolved -- the SDKs do not all build against one distribution. Passing
        a rich object ties this line to that resolution; a float does not.
        """
        kwargs = self._auth_kwargs()
        if "request_timeout" not in kwargs:
            kwargs.setdefault("timeout", config.LLM_TIMEOUT_SECONDS)
        return kwargs

    def _model_kwargs(self, temperature: float | None = None) -> dict[str, Any]:
        """This model's own arguments: its effort, and a temperature it accepts.

        For Claude, ``output_config`` goes in as a keyword argument, which
        LiteLLM maps into the Anthropic request. ``extra_body`` would not work
        on that route: LiteLLM forwards it as a field literally named
        ``extra_body``, and Anthropic rejects the request with a 400. For
        DeepSeek it is the only route for the effort (see :func:`_deepseek_body`).
        """
        kwargs: dict[str, Any] = {}
        effort = effort_for(self._model) if _reaches_anthropic_api(self._model) else None
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}
        if temperature is not None and accepts_temperature(self._model):
            kwargs["temperature"] = temperature
        body = _deepseek_body(self._model)
        if body:
            kwargs["extra_body"] = body
        return kwargs

    def _request_kwargs(
        self, temperature: float | None = None, *, stream: bool = False
    ) -> dict[str, Any]:
        """The model's arguments and the call's, as one set for a completion.

        Both can carry ``extra_body``: this model's (DeepSeek's thinking fields)
        and a host's own, in ``extra``. They are merged rather than one
        replacing the other, the host's keys winning where both set one; a
        host ``extra_body`` that is not a mapping is left as it is.

        A *stream* on a route that can report its usage asks for it (see
        :func:`_streams_usage`), unless the host set ``stream_options`` itself.
        """
        kwargs = self._model_kwargs(temperature)
        ours = kwargs.pop("extra_body", None)
        kwargs.update(self._call_kwargs())
        if ours:
            theirs = kwargs.get("extra_body")
            if theirs is None:
                kwargs["extra_body"] = ours
            elif isinstance(theirs, dict):
                kwargs["extra_body"] = {**ours, **theirs}
        if stream and _streams_usage(self._model):
            kwargs.setdefault("stream_options", {"include_usage": True})
        return kwargs

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, system: SystemParam, user: str, max_tokens: int | None = None) -> str:
        budget = max_tokens or output_budget("answer")
        kwargs = self._request_kwargs()
        with _deadline(self._model, _configured_deadline(kwargs)):
            response = litellm.completion(
                model=self._model,
                messages=[
                    _system_message(system, keep_blocks=self.supports_prompt_caching),
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                **kwargs,
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
        kwargs = self._request_kwargs(stream=True)
        # The iteration is inside the deadline too, not just the call that opens
        # the stream. litellm applies the timeout per chunk read, so a provider
        # that accepts the request and then stalls raises here -- which is the
        # shape of the hang this exists for, and the one a wrapper around the
        # opening call alone would miss.
        with _deadline(self._model, _configured_deadline(kwargs)):
            response = litellm.completion(
                model=self._model,
                messages=[
                    _system_message(system, keep_blocks=self.supports_prompt_caching),
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                stream=True,
                **kwargs,
            )
            collected: list[str] = []
            finish: object = None
            usage: Any = None
            for chunk in response:
                # Usage, when asked for, comes on the last chunk, which an
                # OpenAI-style stream sends with no choices at all.
                usage = getattr(chunk, "usage", None) or usage
                if not chunk.choices:
                    continue
                finish = getattr(chunk.choices[0], "finish_reason", None) or finish
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta
        # Exact where the stream reported its usage, an estimate where it did not.
        _record_stream_usage(self._model, usage, system, user, collected)
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
                _system_message(system, keep_blocks=self.supports_prompt_caching),
                {"role": "user", "content": user},
            ],
            "max_tokens": budget,
            **self._request_kwargs(temperature),
        }
        with _deadline(self._model, _configured_deadline(kwargs)):
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
        kwargs = self._request_kwargs(stream=True)
        # See the sync path: the iteration is inside the deadline as well.
        with _deadline(self._model, _configured_deadline(kwargs)):
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    _system_message(system, keep_blocks=self.supports_prompt_caching),
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                stream=True,
                **kwargs,
            )
            collected: list[str] = []
            finish: object = None
            usage: Any = None
            async for chunk in response:
                # See the sync path: the usage chunk carries no choices.
                usage = getattr(chunk, "usage", None) or usage
                if not chunk.choices:
                    continue
                finish = getattr(chunk.choices[0], "finish_reason", None) or finish
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta
        _record_stream_usage(self._model, usage, system, user, collected)
        # See the sync path: empty, not blank.
        if not collected and str(finish) in TRUNCATED:
            raise OutputBudgetExhausted(self._model, budget)
