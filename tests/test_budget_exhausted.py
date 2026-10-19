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
# Both ask for a small budget (PROBE_MAX_TOKENS) to keep the check cheap, and
# both throw the answer away -- they are asking whether the provider is
# reachable, not what it said. A reasoning model can spend the whole budget
# reasoning, so on exactly the deployment this change exists for, the probe
# raises. Reporting that as a fault would tell the operator to raise a limit
# that has no bearing on a hard-coded probe budget, in the command someone runs
# to find out what is wrong.
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
    # And says so. `doctor` is what an operator runs when answers come back
    # empty; a bare "responds" would send them to look somewhere else.
    assert "LLM_MAX_OUTPUT_TOKENS" in result.detail
    assert "probe budget" in result.detail


def test_doctor_says_nothing_extra_when_the_model_answers() -> None:
    """The note is a signal, so it must not appear on every run."""
    from nlqueries.cli.main import _check_llm

    fine = MagicMock()
    fine.complete.return_value = "OK"

    with (
        patch("nlqueries.config.llm_credentials_available", return_value=True),
        patch("nlqueries.llm.get_llm_client", return_value=fine),
    ):
        result = _check_llm()

    assert result.status == "ok"
    assert "LLM_MAX_OUTPUT_TOKENS" not in result.detail


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
    assert "LLM_MAX_OUTPUT_TOKENS" in llm_line


def test_mcp_health_says_nothing_extra_when_the_model_answers() -> None:
    fine = MagicMock()
    fine.complete.return_value = "OK"

    llm_line = next(ln for ln in _health(fine).splitlines() if "**LLM**" in ln)
    assert llm_line.startswith("✅")
    assert "LLM_MAX_OUTPUT_TOKENS" not in llm_line


def test_mcp_health_still_reports_a_real_failure() -> None:
    broken = MagicMock()
    broken.complete.side_effect = RuntimeError("401 unauthorized")

    llm_line = next(ln for ln in _health(broken).splitlines() if "**LLM**" in ln)
    assert llm_line.startswith("❌")
    assert "401 unauthorized" in llm_line


# ---------------------------------------------------------------------------
# Anthropic, async
#
# These two are what a default deployment actually runs: `get_llm_client`
# returns `AnthropicClient` for provider="anthropic", `Orchestrator` and
# `document_orchestrator` consume `astream`, and `aclassify_intent`,
# `aresolve_followup`, `sql_generation` and `candidates` consume `acomplete`.
# The `astream` guard is a hand-copy of the sync one, so a dropped clause or a
# mistyped attribute would disable the exception with nothing failing: the
# MagicMock fakes elsewhere never match TRUNCATED, because `str(MagicMock())`
# is a repr.
# ---------------------------------------------------------------------------


class _FakeAsyncStream:
    """The async SDK streaming context manager, as the client uses it."""

    def __init__(self, texts: list[str], stop_reason: str) -> None:
        self._texts = texts
        self._stop = stop_reason

    async def __aenter__(self) -> _FakeAsyncStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    @property
    def text_stream(self):  # noqa: ANN202
        async def gen():  # noqa: ANN202
            for t in self._texts:
                yield t

        return gen()

    async def get_final_message(self) -> SimpleNamespace:
        return _message([], self._stop)


def test_anthropic_acomplete_raises_on_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch)

    async def create(**_kwargs: object) -> SimpleNamespace:
        return _message([], "max_tokens")

    client._aclient = MagicMock()
    client._aclient.messages.create = create

    with pytest.raises(OutputBudgetExhausted) as caught:
        asyncio.run(client.acomplete("sys", "user", max_tokens=11))
    assert caught.value.budget == 11


def test_anthropic_acomplete_returns_a_truncated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncated is still an answer. Only nothing at all is an error."""
    client = _anthropic(monkeypatch)

    async def create(**_kwargs: object) -> SimpleNamespace:
        return _message([_block("half")], "max_tokens")

    client._aclient = MagicMock()
    client._aclient.messages.create = create

    assert asyncio.run(client.acomplete("sys", "user")) == "half"


def test_anthropic_acomplete_survives_a_reply_with_no_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async half of the StopIteration `_text_of` replaced."""
    client = _anthropic(monkeypatch)

    async def create(**_kwargs: object) -> SimpleNamespace:
        return _message([], "end_turn")

    client._aclient = MagicMock()
    client._aclient.messages.create = create

    assert asyncio.run(client.acomplete("sys", "user")) == ""


def _adrain(client: AnthropicClient) -> list[str]:
    async def go() -> list[str]:
        return [t async for t in client.astream("sys", "user")]

    return asyncio.run(go())


def test_anthropic_astream_raises_after_yielding_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _anthropic(monkeypatch)
    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = _FakeAsyncStream([], "max_tokens")

    with pytest.raises(OutputBudgetExhausted):
        _adrain(client)


def test_anthropic_astream_keeps_what_it_yielded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _anthropic(monkeypatch)
    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = _FakeAsyncStream(["some ", "text"], "max_tokens")

    assert _adrain(client) == ["some ", "text"]


def test_anthropic_astream_does_not_raise_after_yielding_only_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty, not blank -- the distinction the guard turns on.

    `exhausted()` treats whitespace as no answer, which is right for a reply
    returned whole and wrong here: a reasoning model that emits a leading
    newline and then hits the cap has already handed those tokens to the
    caller. Raising at the end of the generator would land an error on a
    response body that has started, which is the mid-stream failure the guard
    is placed to avoid.
    """
    client = _anthropic(monkeypatch)
    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = _FakeAsyncStream(["\n\n"], "max_tokens")

    assert _adrain(client) == ["\n\n"]


def test_anthropic_astream_is_quiet_on_an_ordinary_empty_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`end_turn` with nothing to say is not an exhausted budget."""
    client = _anthropic(monkeypatch)
    client._aclient = MagicMock()
    client._aclient.messages.stream.return_value = _FakeAsyncStream([], "end_turn")

    assert _adrain(client) == []


# ---------------------------------------------------------------------------
# The probe's budget and the reasoning-model 400
#
# The probe asked for 5 tokens. OpenAI's Responses API, which LiteLLM uses for
# models such as o3-pro, gpt-5-pro and the codex models, refuses fewer than 16
# outright, and OpenAI and Azure reasoning models report a budget spent before
# any text as an HTTP 400 rather than an empty reply.
# ---------------------------------------------------------------------------

#: OpenAI's wording, as LiteLLM passes it on.
_BUDGET_SPENT_400 = (
    "litellm.BadRequestError: OpenAIException - Could not finish the message because "
    "max_tokens or model output limit was reached. Please try again with higher max_tokens."
)


def test_the_probe_asks_for_the_least_the_responses_api_accepts() -> None:
    from nlqueries.llm import PROBE_MAX_TOKENS, probe_round_trip

    fine = MagicMock()
    fine.complete.return_value = "OK"

    assert probe_round_trip(fine) is False
    assert PROBE_MAX_TOKENS == 16
    assert fine.complete.call_args.kwargs["max_tokens"] == 16


@pytest.mark.parametrize(
    "error",
    [
        OutputBudgetExhausted("reasoner", 16),
        RuntimeError(_BUDGET_SPENT_400),
    ],
    ids=["exhausted", "openai-400"],
)
def test_a_budget_spent_before_any_text_is_a_round_trip(error: Exception) -> None:
    from nlqueries.llm import probe_round_trip

    client = MagicMock()
    client.complete.side_effect = error
    assert probe_round_trip(client) is True


def test_the_400_is_recognised_behind_a_wrapping_exception() -> None:
    """A host or SDK layer that re-raises keeps the provider's text in the cause."""
    from nlqueries.llm import probe_round_trip

    def _wrapped(*_a: object, **_k: object) -> str:
        try:
            raise RuntimeError(_BUDGET_SPENT_400)
        except RuntimeError as exc:
            raise ValueError("provider call failed") from exc

    client = MagicMock()
    client.complete.side_effect = _wrapped
    assert probe_round_trip(client) is True


@pytest.mark.parametrize(
    "message",
    [
        "401 unauthorized",
        # What the 5-token probe got from the Responses API: a real failure.
        "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16",
        "The model `o3-pro` does not exist",
    ],
)
def test_any_other_error_still_propagates(message: str) -> None:
    from nlqueries.llm import probe_round_trip

    client = MagicMock()
    client.complete.side_effect = RuntimeError(message)
    with pytest.raises(RuntimeError, match="."):
        probe_round_trip(client)


def test_doctor_reports_ok_on_the_reasoning_model_400() -> None:
    from nlqueries.cli.main import _check_llm

    client = MagicMock()
    client.complete.side_effect = RuntimeError(_BUDGET_SPENT_400)
    with (
        patch("nlqueries.config.llm_credentials_available", return_value=True),
        patch("nlqueries.llm.get_llm_client", return_value=client),
    ):
        result = _check_llm()

    assert result.status == "ok"
    assert "16-token probe budget" in result.detail
    assert client.complete.call_args.kwargs["max_tokens"] == 16


def test_mcp_health_reports_ok_on_the_reasoning_model_400() -> None:
    client = MagicMock()
    client.complete.side_effect = RuntimeError(_BUDGET_SPENT_400)

    llm_line = next(ln for ln in _health(client).splitlines() if "**LLM**" in ln)
    assert llm_line.startswith("✅")
    assert "16-token probe budget" in llm_line
    assert client.complete.call_args.kwargs["max_tokens"] == 16
