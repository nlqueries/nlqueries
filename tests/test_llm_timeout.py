"""Every LLM call has a deadline, and one exception when it passes.

Until this existed there was none: a slow or stuck model hung the request for
as long as it liked. The server log that prompted this showed a seven-minute
gap between calls on one question, and a later question that made two calls and
then simply stopped -- no error, no completion, no answer.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import litellm.exceptions
import pytest
from nlqueries import config
from nlqueries.llm import LLMTimeout
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.litellm_client import LiteLLMClient

# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


def _timeout(monkeypatch: pytest.MonkeyPatch, written: str | None) -> float:
    """Drive `_llm_timeout` the way the process does: from the environment."""
    if written is None:
        monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", written)
    return config._llm_timeout()


def test_the_default_is_180_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _timeout(monkeypatch, None) == 180.0


def test_a_deployment_can_set_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _timeout(monkeypatch, "45") == 45.0
    assert _timeout(monkeypatch, "12.5") == 12.5


@pytest.mark.parametrize("written", ["inf", "-inf", "Infinity", "nan", "1e400"])
def test_a_non_finite_deadline_is_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, written: str
) -> None:
    """`float()` parses all of these; `int()` -- the sibling setting -- cannot.

    Each one passes a `<= 0` test, reaches the SDK, and produces a deadline
    that never fires, because every comparison against `inf` or `nan` is false.
    That is the unbounded wait this setting exists to end, arrived at through
    the setting itself. `1e400` is the one that needs no ill intent: it
    overflows to `inf` silently.
    """
    with caplog.at_level(logging.WARNING):
        assert _timeout(monkeypatch, written) == 180.0
    assert "LLM_TIMEOUT_SECONDS" in caplog.text


@pytest.mark.parametrize("written", ["0", "-1", "-0.5"])
def test_a_non_positive_deadline_is_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, written: str
) -> None:
    """`0` reads as "no timeout" and means the opposite to the SDKs."""
    with caplog.at_level(logging.WARNING):
        assert _timeout(monkeypatch, written) == 180.0
    assert "LLM_TIMEOUT_SECONDS" in caplog.text


def test_an_unparseable_deadline_is_ignored_rather_than_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`LLM_TIMEOUT_SECONDS=` is how people disable a setting.

    An unhandled ValueError here aborts the import of `nlqueries.config`, which
    takes every CLI command with it -- including `doctor`, the one an operator
    runs to find out what is wrong.
    """
    with caplog.at_level(logging.WARNING):
        assert _timeout(monkeypatch, "") == 180.0
    assert "LLM_TIMEOUT_SECONDS" in caplog.text


def test_the_documented_sdk_defaults_are_still_what_the_sdks_do() -> None:
    """`config.py` and `docs/configuration.md` both name these numbers.

    They say this setting replaces a 600s bound rather than an unbounded wait
    -- an earlier revision claimed the latter and was wrong. That claim is only
    true while the pinned SDKs still default the way they do, and a number in
    prose is the first thing to go stale, so it is asserted rather than
    trusted. If this fails after an SDK bump the fix is to update both
    documents, not to delete the test.
    """
    from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS

    # The public surface, not `anthropic._constants`. That module is private and
    # only importable as an attribute because `_client.py` happens to pull it
    # in, so an SDK reshuffle would land here as an `AttributeError` on
    # whichever unrelated pull request ran next, rather than as the "the
    # documented number has changed" signal this exists to give. A default
    # client reports the same two values; no request is made by constructing
    # one.
    defaults = anthropic.Anthropic(api_key="unused-no-request-is-made").timeout

    assert defaults.read == 600, (
        f"docs say the Anthropic SDK already bounds a read at 600s; it is now {defaults.read}"
    )
    assert defaults.connect == 5.0, (
        "the 5s connect this change preserves is the SDK's own default; it is now "
        f"{defaults.connect}"
    )
    assert COMPLETION_HTTP_FALLBACK_SECONDS == 600.0, (
        "docs say litellm falls back to 600s with no timeout passed; it is now "
        f"{COMPLETION_HTTP_FALLBACK_SECONDS}"
    )


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------


def _reply(content: str) -> MagicMock:
    return MagicMock(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _one_chunk() -> list[SimpleNamespace]:
    """A single stream chunk that finishes normally."""
    return [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason="stop")]
        )
    ]


def _drive(client: LiteLLMClient, path: str) -> None:
    """Run one entry point far enough for its request kwargs to be recorded."""
    if path == "complete":
        client.complete("sys", "user")
    elif path == "stream":
        list(client.stream("sys", "user"))
    elif path == "acomplete":
        asyncio.run(client.acomplete("sys", "user"))
    else:

        async def drain() -> list[str]:
            return [t async for t in client.astream("sys", "user")]

        asyncio.run(drain())


@pytest.mark.parametrize("path", ["complete", "stream", "acomplete", "astream"])
def test_litellm_passes_the_deadline_on_every_call(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """All four, because the name says every call and one of them is the one used.

    This asserted `complete` alone. Swapping `_call_kwargs()` for
    `_auth_kwargs()` in `astream` -- the path the orchestrators actually run --
    left the suite green, which is the same shape of gap this branch has
    already had to close once elsewhere.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k")

    streaming = path in ("stream", "astream")
    target = "litellm.acompletion" if path.startswith("a") else "litellm.completion"

    if path == "acomplete":
        recorded: dict[str, object] = {}

        async def capture(**kwargs: object) -> MagicMock:
            recorded.update(kwargs)
            return _reply("hi")

        with patch(target, side_effect=capture):
            _drive(client, path)
        assert recorded["timeout"] == 42.0
        return

    if path == "astream":
        recorded = {}

        class Empty:
            def __aiter__(self) -> Empty:
                return self

            async def __anext__(self) -> SimpleNamespace:
                raise StopAsyncIteration

        async def capture(**kwargs: object) -> Empty:
            recorded.update(kwargs)
            return Empty()

        with patch(target, side_effect=capture):
            _drive(client, path)
        assert recorded["timeout"] == 42.0
        return

    value = _one_chunk() if streaming else _reply("hi")
    with patch(target, return_value=value) as call:
        _drive(client, path)

    assert call.call_args.kwargs["timeout"] == 42.0


def test_a_host_request_timeout_is_not_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """litellm's other spelling for the same thing.

    `CompletionTimeout.resolve` consults `request_timeout` after `timeout`, so
    supplying `timeout` unconditionally would silently replace a host's
    `extra={"request_timeout": 30}` with this process's default -- nothing
    logged, and the name is not reserved.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 180.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"request_timeout": 30})

    with patch("litellm.completion", return_value=_reply("hi")) as call:
        client.complete("sys", "user")

    assert call.call_args.kwargs["request_timeout"] == 30
    assert "timeout" not in call.call_args.kwargs


def test_a_hosts_own_timeout_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """`extra` is a per-client decision and outranks a process-wide default."""
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"timeout": 5.0})

    with patch("litellm.completion", return_value=_reply("hi")) as call:
        client.complete("sys", "user")

    assert call.call_args.kwargs["timeout"] == 5.0


def test_litellm_timeout_becomes_LLMTimeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k")

    boom = litellm.exceptions.Timeout(message="too slow", model="m", llm_provider="openai")
    with patch("litellm.completion", side_effect=boom), pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == 42.0
    assert caught.value.model == "m"
    assert "LLM_TIMEOUT_SECONDS" in str(caught.value)
    # The SDK error is kept as the cause; the translation adds a name, it does
    # not throw away what the provider said.
    assert isinstance(caught.value.__cause__, litellm.exceptions.Timeout)


def test_a_litellm_stall_from_another_httpx_is_still_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same version split, on the client that had a plain isinstance.

    The Anthropic client matched by name; this one did not, so in an
    environment where litellm resolves a different httpx a mid-stream stall
    would have escaped untranslated -- a raw transport error reaching a host
    that had been asked to catch `LLMTimeout`. The silent half of the split.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 9.0)
    client = LiteLLMClient(model="m", api_key="k")

    # Subclasses nothing this module imported: the same class arriving from a
    # different httpx distribution.
    other_read_timeout = type("ReadTimeout", (Exception,), {})

    def chunks():  # noqa: ANN202
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="half "), finish_reason=None)]
        )
        raise other_read_timeout("stalled")

    with patch("litellm.completion", return_value=chunks()), pytest.raises(LLMTimeout):
        list(client.stream("sys", "user"))


def test_a_request_timeout_deadline_is_still_named_in_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host who chose litellm's other spelling gets the same promise.

    Standing the default down for `request_timeout` left that host with no
    `timeout` key at all, so the deadline lookup found nothing and the
    provider's error went through untranslated -- in the one configuration
    where a number was available to name, and against the docstring's promise
    that a caller catches one type whichever client it got.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 180.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"request_timeout": 30})

    boom = litellm.exceptions.Timeout(message="slow", model="m", llm_provider="openai")
    with patch("litellm.completion", side_effect=boom), pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == 30
    assert "30s" in str(caught.value)


def test_the_deadline_lookup_follows_litellms_own_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`timeout` before `request_timeout`, because that is what litellm does.

    `CompletionTimeout.resolve` takes the first non-None of the call argument,
    `kwargs["timeout"]` and `kwargs["request_timeout"]`. A host that sets both
    gets the first, so naming the second in the error would report a limit that
    did not expire.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 180.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"timeout": 45, "request_timeout": 30})

    boom = litellm.exceptions.Timeout(message="slow", model="m", llm_provider="openai")
    with patch("litellm.completion", side_effect=boom), pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == 45


def test_another_litellm_error_keeps_its_own_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """The translation is narrow. An auth failure is not a deadline."""
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k")

    with (
        patch("litellm.completion", side_effect=RuntimeError("401 unauthorized")),
        pytest.raises(RuntimeError, match="401"),
    ):
        client.complete("sys", "user")


def test_litellm_acomplete_translates_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 7.0)
    client = LiteLLMClient(model="m", api_key="k")

    async def boom(**_kwargs: object) -> MagicMock:
        raise litellm.exceptions.Timeout(message="too slow", model="m", llm_provider="openai")

    with patch("litellm.acompletion", side_effect=boom), pytest.raises(LLMTimeout):
        asyncio.run(client.acomplete("sys", "user"))


def test_a_stream_that_stalls_after_it_opens_still_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline covers the drain, not only the call that opens the stream.

    A provider that accepts the request and then stops sending is the shape of
    the hang this exists for, and a wrapper around the opening call alone would
    not see it.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 9.0)
    client = LiteLLMClient(model="m", api_key="k")

    def chunks():  # noqa: ANN202
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="half "), finish_reason=None)]
        )
        # The transport type, not litellm's: a stall part-way through a stream
        # is raised by httpx, and only some paths are mapped on the way out.
        raise httpx.ReadTimeout("stalled")

    with patch("litellm.completion", return_value=chunks()), pytest.raises(LLMTimeout):
        list(client.stream("sys", "user"))


def test_a_host_that_disables_the_deadline_gets_the_providers_own_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`extra={"timeout": None}` switches OURS off, not the deadline.

    `extra` forwards `None` rather than dropping it, so a host can decline the
    value this process sets. Translating the provider's error anyway would
    claim a limit nobody set -- with `seconds=None` in a message that formats
    it -- so it goes through untouched.

    What such a host gets is litellm's own bound rather than an unbounded call:
    `CompletionTimeout.resolve` takes the first non-`None` of the call
    argument, `kwargs["timeout"]`, `kwargs["request_timeout"]` and finally
    `COMPLETION_HTTP_FALLBACK_SECONDS`, 600.0. An earlier revision of this
    docstring said "no deadline", which was wrong.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"timeout": None})

    assert client._call_kwargs()["timeout"] is None

    boom = litellm.exceptions.Timeout(message="from the provider", model="m", llm_provider="openai")
    with (
        patch("litellm.completion", side_effect=boom),
        pytest.raises(litellm.exceptions.Timeout),
    ):
        client.complete("sys", "user")


def test_a_non_numeric_deadline_still_reports_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm's `timeout` accepts a `str` too, and `extra` forwards it untouched.

    `:g` raises on a string, so formatting it blindly would replace the
    provider's timeout with a `ValueError` -- thrown from the constructor whose
    whole job is to report that timeout clearly.
    """
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k", extra={"timeout": "30s"})

    boom = litellm.exceptions.Timeout(message="slow", model="m", llm_provider="openai")
    with patch("litellm.completion", side_effect=boom), pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == "30s"
    assert "30s" in str(caught.value)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic(monkeypatch: pytest.MonkeyPatch, seconds: float = 30.0) -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", seconds)
    return AnthropicClient(model="claude-test")


def test_anthropic_builds_its_sdk_clients_with_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the SDK client, so no request made through it can miss it."""
    client = _anthropic(monkeypatch, 33.0)
    for sdk in (client._client, client._aclient):
        assert sdk.timeout.read == 33.0
        assert sdk.timeout.write == 33.0
        assert sdk.timeout.pool == 33.0


def test_the_deadline_does_not_swallow_the_connect_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare float sets EVERY phase, connect included.

    The SDK's own default is `Timeout(connect=5.0, read=600, write=600,
    pool=600)`. Handing it 180.0 would move connect from 5s to 180s, so an
    unreachable endpoint -- blocked egress, a mistyped `api_base` -- would sit
    for three minutes rather than five seconds, and three times over for a
    question that classifies, generates and corrects. The read deadline is what
    this change is for; connect was already right.
    """
    client = _anthropic(monkeypatch, 180.0)
    for sdk in (client._client, client._aclient):
        assert sdk.timeout.connect == 5.0
        assert sdk.timeout.read == 180.0


def test_a_timeout_from_another_httpx_is_still_a_timeout() -> None:
    """The version split CI found, in the shape that would not have been caught.

    Which httpx the Anthropic SDK raises from depends on its version -- newer
    builds are against `httpx2` -- so an `isinstance` against the httpx
    imported here is true in one environment and false in another with nothing
    in this repository having changed. CI caught the matching split on the
    constructor argument (`httpx._config.Timeout` rejected where
    `httpx2._config.Timeout` was expected) while the local run passed, because
    locally the two are the same object.
    """
    from nlqueries.llm.anthropic_client import _is_timeout

    # Not a subclass of anything this module imported: a stand-in for the same
    # class arriving from a different httpx distribution.
    other_httpx_read_timeout = type("ReadTimeout", (Exception,), {})

    assert _is_timeout(other_httpx_read_timeout("stalled")) is True
    assert _is_timeout(RuntimeError("401 unauthorized")) is False


def test_anthropic_timeout_becomes_LLMTimeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch, 33.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = anthropic.APITimeoutError(request=MagicMock())

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == 33.0
    assert isinstance(caught.value.__cause__, anthropic.APITimeoutError)


def test_a_connect_failure_reports_the_connect_limit_not_the_read_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holding connect at 5s while read is 180 made the message wrong by 36x.

    Blocked egress or a mistyped `api_base` expires the connect phase after
    five seconds. Reporting "did not respond within 180s. Raise
    LLM_TIMEOUT_SECONDS" would be wrong about the number and useless about the
    fix, since that setting does not touch connect -- and it is the message an
    operator gets precisely when they have the endpoint wrong.
    """
    client = _anthropic(monkeypatch, 180.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = httpx.ConnectTimeout("no route")

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.phase == "connect"
    assert caught.value.seconds == 5.0
    assert "5s" in str(caught.value)
    assert "180" not in str(caught.value)
    assert "will not help" in str(caught.value)


def test_the_sdk_wrapper_does_not_hide_which_phase_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`APITimeoutError` is what the SDK raises; the cause says which phase."""
    client = _anthropic(monkeypatch, 180.0)
    client._client = MagicMock()
    wrapped = anthropic.APITimeoutError(request=MagicMock())
    wrapped.__cause__ = httpx.ConnectTimeout("no route")
    client._client.messages.create.side_effect = wrapped

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.phase == "connect"
    assert caught.value.seconds == 5.0


def test_a_pool_timeout_points_at_the_setting_that_really_set_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pool` is not `connect`, and grouping them made this wrong both ways.

    `anthropic.Timeout(seconds, connect=5.0)` sets read, write AND pool to the
    configured value -- only connect is held short. So a pool timeout expires
    at LLM_TIMEOUT_SECONDS, and telling the operator that raising it "will not
    help" sends them away from the one control that would. It also means every
    pooled connection was busy, which is concurrency, not an unreachable
    endpoint.
    """
    client = _anthropic(monkeypatch, 180.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = httpx.PoolTimeout("all busy")

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.phase == "pool"
    assert caught.value.seconds == 180.0
    message = str(caught.value)
    assert "will not help" not in message
    assert "Raise LLM_TIMEOUT_SECONDS" in message
    assert "connection" in message


def test_a_read_timeout_still_names_the_setting_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _anthropic(monkeypatch, 180.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.phase == "read"
    assert caught.value.seconds == 180.0
    assert "LLM_TIMEOUT_SECONDS if this model is legitimately slow" in str(caught.value)


def test_anthropic_acomplete_translates_too(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch, 33.0)

    async def boom(**_kwargs: object) -> object:
        raise anthropic.APITimeoutError(request=MagicMock())

    client._aclient = MagicMock()
    client._aclient.messages.create = boom

    with pytest.raises(LLMTimeout):
        asyncio.run(client.acomplete("sys", "user"))


def test_anthropic_stream_that_stalls_mid_drain_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _anthropic(monkeypatch, 33.0)

    class Stalling:
        def __enter__(self) -> Stalling:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        @property
        def text_stream(self):  # noqa: ANN202
            def gen():  # noqa: ANN202
                yield "half "
                # `httpx.ReadTimeout`, NOT `anthropic.APITimeoutError`. The SDK
                # converts one into the other only around
                # `self._client.send(...)` in `_base_client._request`; with
                # `stream=True` that returns once the headers arrive and the
                # body is read lazily afterwards, outside the `try`. Neither
                # `_streaming` nor `lib/streaming/_messages` handles a timeout
                # at all. Injecting the SDK type here would be injecting an
                # error the SDK cannot raise at this point -- which is what the
                # first version of this test did, and it passed while
                # exercising nothing.
                raise httpx.ReadTimeout("stalled")

            return gen()

    client._client = MagicMock()
    client._client.messages.stream.return_value = Stalling()

    with pytest.raises(LLMTimeout):
        list(client.stream("sys", "user"))


def test_another_anthropic_error_keeps_its_own_type(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch, 33.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = RuntimeError("401 unauthorized")

    with pytest.raises(RuntimeError, match="401"):
        client.complete("sys", "user")


# ---------------------------------------------------------------------------
# The async drains
#
# `astream` is the only streaming path this repository calls --
# `orchestrator.py:202` and `document_orchestrator.py:94` both use it, and
# nothing outside the base-class shim in `client.py` calls `stream`. The sync
# drain tests above were therefore covering the paths that carry no traffic:
# removing `with _deadline(...)` from either `astream`, leaving it around the
# opening call only, left the suite green.
# ---------------------------------------------------------------------------


def _adrain(client: object) -> list[str]:
    async def go() -> list[str]:
        return [t async for t in client.astream("sys", "user")]  # type: ignore[attr-defined]

    return asyncio.run(go())


def test_litellm_astream_that_stalls_after_it_opens_still_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 9.0)
    client = LiteLLMClient(model="m", api_key="k")

    class Stalling:
        def __aiter__(self) -> Stalling:
            self._sent = False
            return self

        async def __anext__(self) -> SimpleNamespace:
            if self._sent:
                # See the sync path: httpx, not litellm.
                raise httpx.ReadTimeout("stalled")
            self._sent = True
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="half "), finish_reason=None)
                ]
            )

    async def opened(**_kwargs: object) -> Stalling:
        return Stalling()

    with patch("litellm.acompletion", side_effect=opened), pytest.raises(LLMTimeout):
        _adrain(client)


def test_anthropic_astream_that_stalls_mid_drain_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _anthropic(monkeypatch, 33.0)

    class Stalling:
        async def __aenter__(self) -> Stalling:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        @property
        def text_stream(self):  # noqa: ANN202
            async def gen():  # noqa: ANN202
                yield "half "
                # See the sync path: the transport raises, not the SDK.
                raise httpx.ReadTimeout("stalled")

            return gen()

    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = Stalling()

    with pytest.raises(LLMTimeout):
        _adrain(client)
