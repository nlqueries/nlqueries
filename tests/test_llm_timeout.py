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


def test_litellm_passes_the_deadline_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    client = LiteLLMClient(model="m", api_key="k")

    with patch("litellm.completion", return_value=_reply("hi")) as call:
        client.complete("sys", "user")

    assert call.call_args.kwargs["timeout"] == 42.0


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
        raise litellm.exceptions.Timeout(message="stalled", model="m", llm_provider="openai")

    with patch("litellm.completion", return_value=chunks()), pytest.raises(LLMTimeout):
        list(client.stream("sys", "user"))


def test_a_host_that_disables_the_deadline_gets_the_providers_own_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`extra={"timeout": None}` is a supported way to say "no deadline".

    `extra` forwards `None` rather than dropping it, so a host can switch ours
    off. There is then no deadline of ours to name, and translating the error
    anyway would claim a limit nobody set -- with `seconds=None` in a message
    that formats it.
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
    assert client._client.timeout == 33.0
    assert client._aclient.timeout == 33.0


def test_anthropic_timeout_becomes_LLMTimeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch, 33.0)
    client._client = MagicMock()
    client._client.messages.create.side_effect = anthropic.APITimeoutError(request=MagicMock())

    with pytest.raises(LLMTimeout) as caught:
        client.complete("sys", "user")

    assert caught.value.seconds == 33.0
    assert isinstance(caught.value.__cause__, anthropic.APITimeoutError)


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
                raise anthropic.APITimeoutError(request=MagicMock())

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
                raise litellm.exceptions.Timeout(
                    message="stalled", model="m", llm_provider="openai"
                )
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
                raise anthropic.APITimeoutError(request=MagicMock())

            return gen()

    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = Stalling()

    with pytest.raises(LLMTimeout):
        _adrain(client)
