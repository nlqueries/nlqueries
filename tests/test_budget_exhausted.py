"""An exhausted output budget is an error, not an empty answer.

A reasoning model bills its reasoning from the same allowance and spends it
first, so a budget sized for the answer alone can return nothing at all. An
empty string is indistinguishable from a model with nothing to say; these
assert it is distinguishable now.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.llm import OutputBudgetExhausted
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.client import exhausted
from nlqueries.llm.litellm_client import LiteLLMClient


def _reply(content: str | None, finish_reason: str) -> MagicMock:
    """A LiteLLM response object, shaped the way the client reads it."""
    return MagicMock(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _chunks(pieces: list[str | None], finish_reason: str) -> list[SimpleNamespace]:
    """Stream chunks; the finish reason arrives on the last one, as it does live."""
    out = []
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        out.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=piece),
                        finish_reason=finish_reason if last else None,
                    )
                ]
            )
        )
    return out


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("finish", "content", "expected"),
    [
        ("length", "", True),
        ("length", "   \n ", True),  # whitespace is not an answer
        ("max_tokens", "", True),  # the Anthropic spelling
        ("length", "partial ans", False),  # truncated, but an answer
        ("stop", "", False),  # nothing to say, which is not this
        ("stop", "fine", False),
        (None, "", False),
    ],
)
def test_exhausted_distinguishes_no_room_from_nothing_to_say(
    finish: object, content: str, expected: bool
) -> None:
    assert exhausted(finish, content) is expected


# ---------------------------------------------------------------------------
# The clients
# ---------------------------------------------------------------------------


def test_complete_raises_when_the_budget_produced_nothing() -> None:
    client = LiteLLMClient(model="m", api_key="k")
    with (
        patch("litellm.completion", return_value=_reply("", "length")),
        pytest.raises(OutputBudgetExhausted) as caught,
    ):
        client.complete("sys", "user", max_tokens=7)
    assert caught.value.budget == 7
    assert caught.value.model == "m"
    assert "7" in str(caught.value)


def test_complete_returns_a_truncated_answer_rather_than_raising() -> None:
    """A short answer beats no answer. Only nothing at all is an error."""
    client = LiteLLMClient(model="m", api_key="k")
    with patch("litellm.completion", return_value=_reply("half an ans", "length")):
        assert client.complete("sys", "user") == "half an ans"


def test_an_ordinary_empty_reply_is_not_an_exhausted_budget() -> None:
    """`finish_reason=stop` with no content is a model with nothing to say."""
    client = LiteLLMClient(model="m", api_key="k")
    with patch("litellm.completion", return_value=_reply("", "stop")):
        assert client.complete("sys", "user") == ""


def test_a_provider_that_omits_the_finish_reason_still_works() -> None:
    """LiteLLM normalises 100+ providers and does not promise this field.

    Reading it directly turned a missing attribute into an AttributeError on
    every call, which is how the existing usage tests caught this -- their fakes
    do not carry the field, and neither will some provider. A reply whose finish
    reason is unknown is an ordinary reply.
    """
    client = LiteLLMClient(model="m", api_key="k")
    bare = MagicMock(
        choices=[SimpleNamespace(message=SimpleNamespace(content="an answer"))],
        usage=None,
    )
    with patch("litellm.completion", return_value=bare):
        assert client.complete("sys", "user") == "an answer"


def test_acomplete_raises_too() -> None:
    client = LiteLLMClient(model="m", api_key="k")

    async def fake(**_kwargs: object) -> MagicMock:
        return _reply("", "length")

    with patch("litellm.acompletion", side_effect=fake), pytest.raises(OutputBudgetExhausted):
        asyncio.run(client.acomplete("sys", "user", max_tokens=5))


def test_stream_raises_only_after_yielding_nothing() -> None:
    """The finish reason arrives last, so this can only be decided at the end."""
    client = LiteLLMClient(model="m", api_key="k")
    with (
        patch("litellm.completion", return_value=_chunks([None], "length")),
        pytest.raises(OutputBudgetExhausted),
    ):
        list(client.stream("sys", "user"))


def test_stream_keeps_what_it_yielded() -> None:
    """Text already delivered is an answer; taking it away to raise is worse."""
    client = LiteLLMClient(model="m", api_key="k")
    with patch("litellm.completion", return_value=_chunks(["some ", "text"], "length")):
        assert list(client.stream("sys", "user")) == ["some ", "text"]


def test_astream_raises_when_it_yielded_nothing() -> None:
    client = LiteLLMClient(model="m", api_key="k")

    class FakeStream:
        def __aiter__(self) -> FakeStream:
            self._it = iter(_chunks([None], "length"))
            return self

        async def __anext__(self) -> SimpleNamespace:
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

    async def fake(**_kwargs: object) -> FakeStream:
        return FakeStream()

    async def drain() -> list[str]:
        return [t async for t in client.astream("sys", "user")]

    with patch("litellm.acompletion", side_effect=fake), pytest.raises(OutputBudgetExhausted):
        asyncio.run(drain())


# ---------------------------------------------------------------------------
# Anthropic
#
# The least obvious logic in the change lives here: the `max_tokens` spelling,
# the raise placed OUTSIDE `contextlib.suppress`, and `_text_of` standing in for
# a `next(...)` that raised StopIteration. None of it was covered.
# ---------------------------------------------------------------------------


def _block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _message(blocks: list[SimpleNamespace], stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=0),
    )


def _anthropic(monkeypatch: pytest.MonkeyPatch) -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    return AnthropicClient(model="claude-test")


def test_anthropic_complete_raises_on_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """`max_tokens` is the Anthropic spelling of `length`."""
    client = _anthropic(monkeypatch)
    client._client = MagicMock()
    client._client.messages.create.return_value = _message([], "max_tokens")

    with pytest.raises(OutputBudgetExhausted) as caught:
        client.complete("sys", "user", max_tokens=9)
    assert caught.value.budget == 9


def test_anthropic_complete_survives_a_reply_with_no_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The StopIteration this replaced.

    `next(b.text for b in response.content if b.type == "text")` raises
    StopIteration on a reply with no text block -- what a reasoning model
    returns -- and inside a generator PEP 479 turns that into a bare
    RuntimeError. With `stop_reason="end_turn"` there is no budget problem to
    report, so the correct answer is an empty string, not an exception.
    """
    client = _anthropic(monkeypatch)
    client._client = MagicMock()
    client._client.messages.create.return_value = _message([], "end_turn")

    assert client.complete("sys", "user") == ""


def test_anthropic_complete_returns_a_truncated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _anthropic(monkeypatch)
    client._client = MagicMock()
    client._client.messages.create.return_value = _message([_block("half")], "max_tokens")

    assert client.complete("sys", "user") == "half"


class _FakeStream:
    """Mimics the SDK's streaming context manager."""

    def __init__(self, texts: list[str], stop_reason: str) -> None:
        self._texts = texts
        self._stop = stop_reason

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    @property
    def text_stream(self):  # noqa: ANN202
        return iter(self._texts)

    def get_final_message(self) -> SimpleNamespace:
        return _message([], self._stop)


def test_anthropic_stream_raises_after_yielding_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the raise must survive `contextlib.suppress(Exception)`.

    The final-message read is wrapped in a suppress so usage recording is
    best-effort. Raising inside that block would be swallowed by it, and the
    caller would get an empty stream with no reason -- which is the behaviour
    this whole change exists to remove.
    """
    client = _anthropic(monkeypatch)
    client._client = MagicMock()
    client._client.messages.stream.return_value = _FakeStream([], "max_tokens")

    with pytest.raises(OutputBudgetExhausted):
        list(client.stream("sys", "user"))


def test_anthropic_stream_keeps_what_it_yielded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch)
    client._client = MagicMock()
    client._client.messages.stream.return_value = _FakeStream(["some ", "text"], "max_tokens")

    assert list(client.stream("sys", "user")) == ["some ", "text"]


# ---------------------------------------------------------------------------
# The health probes
#
# Both ask for five tokens to keep the check cheap, and both throw the answer
# away -- they are asking whether the provider is reachable, not what it said.
# A reasoning model spends all five reasoning, so on exactly the deployment
# this change exists for, the probe now raises. Reporting that as a fault would
# tell the operator to raise a limit that has no bearing on a hard-coded probe
# budget, in the command someone runs to find out what is wrong.
# ---------------------------------------------------------------------------


def _exhausting_client() -> MagicMock:
    client = MagicMock()
    client.complete.side_effect = OutputBudgetExhausted("reasoner", 5)
    return client


def test_doctor_reports_ok_when_the_probe_budget_is_exhausted() -> None:
    from nlqueries.cli.main import _check_llm

    with (
        patch("nlqueries.config.llm_credentials_available", return_value=True),
        patch("nlqueries.llm.get_llm_client", return_value=_exhausting_client()),
    ):
        result = _check_llm()

    assert result.status == "ok"


def test_doctor_still_reports_a_real_failure() -> None:
    """The suppress is narrow: anything else still fails the check."""
    from nlqueries.cli.main import _check_llm

    broken = MagicMock()
    broken.complete.side_effect = RuntimeError("401 unauthorized")

    with (
        patch("nlqueries.config.llm_credentials_available", return_value=True),
        patch("nlqueries.llm.get_llm_client", return_value=broken),
    ):
        result = _check_llm()

    assert result.status == "fail"
    assert "401 unauthorized" in result.detail


def _health(client: MagicMock) -> str:
    from nlqueries.mcp_server.server import health

    with (
        patch("nlqueries.mcp_server.server.config.llm_credentials_available", return_value=True),
        patch("nlqueries.llm.get_llm_client", return_value=client),
        patch("urllib.request.urlopen", side_effect=OSError("no qdrant here")),
        patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=None),
        patch("nlqueries.mcp_server.server.list_agents", return_value=[]),
    ):
        return health(probe_llm=True)


def test_mcp_health_reports_ok_when_the_probe_budget_is_exhausted() -> None:
    llm_line = next(ln for ln in _health(_exhausting_client()).splitlines() if "**LLM**" in ln)
    assert llm_line.startswith("✅")


def test_mcp_health_still_reports_a_real_failure() -> None:
    broken = MagicMock()
    broken.complete.side_effect = RuntimeError("401 unauthorized")

    llm_line = next(ln for ln in _health(broken).splitlines() if "**LLM**" in ln)
    assert llm_line.startswith("❌")
    assert "401 unauthorized" in llm_line
