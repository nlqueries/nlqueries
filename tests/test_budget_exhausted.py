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
