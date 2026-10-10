"""DeepSeek's thinking controls, and usage read from an OpenAI-style stream.

DeepSeek takes ``reasoning_effort`` (``low``, ``high``, ``max``) and
``thinking`` (``{"type": "enabled"|"disabled"}``) in its request body. LiteLLM
turns a top-level ``reasoning_effort`` for DeepSeek into ``thinking`` and drops
the level, so the client sends both in ``extra_body``. The tests at the end run
the real LiteLLM against a local server standing in for DeepSeek and read the
request body it receives, so they hold for whatever LiteLLM is installed.

Streams on OpenAI-compatible routes report no usage unless asked
(``stream_options={"include_usage": True}``); asked, the last chunk carries it,
with the reasoning tokens under ``completion_tokens_details``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.llm import litellm_client
from nlqueries.llm.litellm_client import LiteLLMClient
from nlqueries.llm.usage import UsageRecord, use_usage_sink

DEEPSEEK = "deepseek/deepseek-v4-pro"


def _acomplete_kwargs(client: LiteLLMClient) -> dict[str, Any]:
    """The kwargs ``acomplete`` hands to ``litellm.acompletion``."""
    captured: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
        )

    with patch.object(litellm_client.litellm, "acompletion", _fake):
        asyncio.run(client.acomplete("system", "user"))
    return captured


# ---------------------------------------------------------------------------
# What the client asks LiteLLM to send
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("effort", "sent"),
    [("low", "low"), ("medium", "low"), ("high", "high"), ("xhigh", "max"), ("max", "max")],
)
def test_llm_effort_is_sent_as_deepseek_s_reasoning_effort(effort: str, sent: str) -> None:
    with patch.object(config, "LLM_EFFORT", effort), patch.object(config, "LLM_THINKING", True):
        kwargs = _acomplete_kwargs(LiteLLMClient(model=DEEPSEEK))

    assert kwargs["extra_body"] == {"reasoning_effort": sent}
    # Not top-level, where LiteLLM's DeepSeek mapping would drop the level.
    assert "reasoning_effort" not in kwargs


def test_a_blank_effort_sends_deepseek_nothing() -> None:
    with patch.object(config, "LLM_EFFORT", None), patch.object(config, "LLM_THINKING", True):
        kwargs = _acomplete_kwargs(LiteLLMClient(model=DEEPSEEK))

    assert "extra_body" not in kwargs


def test_thinking_off_disables_it_and_sends_no_effort() -> None:
    """An effort means nothing without thinking, so none goes with it."""
    with patch.object(config, "LLM_EFFORT", "max"), patch.object(config, "LLM_THINKING", False):
        kwargs = _acomplete_kwargs(LiteLLMClient(model=DEEPSEEK))

    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-4-6", "openai/gpt-4o", "bedrock/us.anthropic.claude-sonnet-4-6"],
)
def test_other_models_get_no_deepseek_fields(model: str) -> None:
    with patch.object(config, "LLM_EFFORT", "max"), patch.object(config, "LLM_THINKING", False):
        kwargs = _acomplete_kwargs(LiteLLMClient(model=model))

    assert "extra_body" not in kwargs


def test_a_host_extra_body_is_merged_with_the_model_s() -> None:
    """Merged, not replaced either way; the host's keys win where both set one,
    and the host's own dict is not written to."""
    host = {"extra_body": {"user_tag": "t1", "reasoning_effort": "high"}}
    with patch.object(config, "LLM_EFFORT", "low"), patch.object(config, "LLM_THINKING", True):
        kwargs = _acomplete_kwargs(LiteLLMClient(model=DEEPSEEK, extra=host))

    assert kwargs["extra_body"] == {"reasoning_effort": "high", "user_tag": "t1"}
    assert host == {"extra_body": {"user_tag": "t1", "reasoning_effort": "high"}}


def test_a_host_extra_body_reaches_other_models_unchanged() -> None:
    host = {"extra_body": {"user_tag": "t1"}}
    kwargs = _acomplete_kwargs(LiteLLMClient(model="openai/gpt-4o", extra=host))

    assert kwargs["extra_body"] == {"user_tag": "t1"}


def test_every_path_sends_the_fields() -> None:
    """complete, stream and astream build their kwargs the same way."""
    seen: list[dict[str, Any]] = []

    def _sync(**kwargs: Any) -> Any:
        seen.append(kwargs)
        if kwargs.get("stream"):
            return iter([_chunk("x", "stop")])
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
        )

    async def _async(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _AsyncChunks([_chunk("x", "stop")])

    client = LiteLLMClient(model=DEEPSEEK)
    with (
        patch.object(config, "LLM_EFFORT", "high"),
        patch.object(config, "LLM_THINKING", True),
        patch.object(litellm_client.litellm, "completion", _sync),
        patch.object(litellm_client.litellm, "acompletion", _async),
    ):
        client.complete("system", "user")
        list(client.stream("system", "user"))

        async def _drain() -> None:
            async for _ in client.astream("system", "user"):
                pass

        asyncio.run(_drain())

    assert [k["extra_body"] for k in seen] == [{"reasoning_effort": "high"}] * 3


# ---------------------------------------------------------------------------
# LLM_THINKING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("written", "on"), [("off", False), (" OFF ", False), ("on", True)])
def test_llm_thinking_is_read(monkeypatch: pytest.MonkeyPatch, written: str, on: bool) -> None:
    monkeypatch.setenv("LLM_THINKING", written)
    assert config._thinking() is on


@pytest.mark.parametrize("written", [None, "", "  "])
def test_unset_or_blank_thinking_is_on(
    monkeypatch: pytest.MonkeyPatch, written: str | None
) -> None:
    if written is None:
        monkeypatch.delenv("LLM_THINKING", raising=False)
    else:
        monkeypatch.setenv("LLM_THINKING", written)
    assert config._thinking() is True


def test_a_misspelt_thinking_stays_on_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LLM_THINKING", "disabled")
    with caplog.at_level(logging.WARNING):
        assert config._thinking() is True
    assert "LLM_THINKING" in caplog.text


# ---------------------------------------------------------------------------
# Usage from a stream
# ---------------------------------------------------------------------------


def _chunk(content: str | None, finish: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _usage_chunk() -> SimpleNamespace:
    """The last chunk of an OpenAI-style stream asked for usage: no choices."""
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=9,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
    )
    return SimpleNamespace(choices=[], usage=usage)


class _AsyncChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._it = iter(chunks)

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


EXACT = UsageRecord(
    model=DEEPSEEK,
    input_tokens=8,
    output_tokens=9,
    cache_read_tokens=4,
    estimated=False,
    reasoning_tokens=6,
)


def test_a_stream_records_the_usage_its_last_chunk_reports() -> None:
    chunks = [_chunk("hel"), _chunk("lo", "stop"), _usage_chunk()]
    completion = MagicMock(return_value=iter(chunks))
    sink: list[UsageRecord] = []
    with patch.object(litellm_client.litellm, "completion", completion), use_usage_sink(sink):
        out = "".join(LiteLLMClient(model=DEEPSEEK).stream("system", "user"))

    assert out == "hello"
    assert sink == [EXACT]
    assert completion.call_args.kwargs["stream_options"] == {"include_usage": True}


def test_an_async_stream_records_the_usage_its_last_chunk_reports() -> None:
    chunks = [_chunk("hel"), _chunk("lo", "stop"), _usage_chunk()]
    sent: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> Any:
        sent.update(kwargs)
        return _AsyncChunks(chunks)

    async def _drain() -> str:
        return "".join([piece async for piece in LiteLLMClient(model=DEEPSEEK).astream("s", "u")])

    sink: list[UsageRecord] = []
    with patch.object(litellm_client.litellm, "acompletion", _fake), use_usage_sink(sink):
        out = asyncio.run(_drain())

    assert out == "hello"
    assert sink == [EXACT]
    assert sent["stream_options"] == {"include_usage": True}


def test_a_stream_without_usage_is_still_estimated() -> None:
    completion = MagicMock(return_value=iter([_chunk("hello", "stop")]))
    sink: list[UsageRecord] = []
    with patch.object(litellm_client.litellm, "completion", completion), use_usage_sink(sink):
        list(LiteLLMClient(model=DEEPSEEK).stream("system", "user"))

    assert len(sink) == 1 and sink[0].estimated is True


def test_a_completion_records_its_reasoning_tokens() -> None:
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=7,
        prompt_tokens_details=None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )

    async def _fake(**kwargs: Any) -> Any:
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
        )

    sink: list[UsageRecord] = []
    with patch.object(litellm_client.litellm, "acompletion", _fake), use_usage_sink(sink):
        asyncio.run(LiteLLMClient(model=DEEPSEEK).acomplete("system", "user"))

    # Reasoning is part of the output as billed, and visible on its own.
    assert (sink[0].output_tokens, sink[0].reasoning_tokens) == (7, 5)


@pytest.mark.parametrize(
    ("model", "asked"),
    [
        (DEEPSEEK, True),
        ("openai/gpt-4o", True),
        # Not OpenAI-compatible routes, though LiteLLM takes stream_options for Bedrock.
        ("bedrock/us.anthropic.claude-sonnet-4-6", False),
        ("anthropic/claude-sonnet-4-6", False),
    ],
)
def test_usage_is_asked_for_on_openai_compatible_routes(model: str, asked: bool) -> None:
    completion = MagicMock(return_value=iter([_chunk("x", "stop")]))
    with patch.object(litellm_client.litellm, "completion", completion):
        list(LiteLLMClient(model=model).stream("system", "user"))

    assert ("stream_options" in completion.call_args.kwargs) is asked


def test_a_host_s_own_stream_options_wins() -> None:
    host = {"stream_options": {"include_usage": False}}
    completion = MagicMock(return_value=iter([_chunk("x", "stop")]))
    with patch.object(litellm_client.litellm, "completion", completion):
        list(LiteLLMClient(model=DEEPSEEK, extra=host).stream("system", "user"))

    assert completion.call_args.kwargs["stream_options"] == {"include_usage": False}


# ---------------------------------------------------------------------------
# The real LiteLLM, against a server standing in for DeepSeek
# ---------------------------------------------------------------------------


def _stream_body() -> bytes:
    base = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "deepseek"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 6},
            },
        },
    ]
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


_COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
}


@pytest.fixture
def deepseek() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """A local server answering as DeepSeek; yields its base URL and the
    request bodies it received."""
    bodies: list[dict[str, Any]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the http.server hook
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            data = _stream_body() if body.get("stream") else json.dumps(_COMPLETION).encode()
            self.send_response(200)
            kind = "text/event-stream" if body.get("stream") else "application/json"
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", bodies
    finally:
        server.shutdown()
        server.server_close()


def test_the_effort_level_reaches_deepseek_s_request_body(
    deepseek: tuple[str, list[dict[str, Any]]],
) -> None:
    base, bodies = deepseek
    client = LiteLLMClient(model=DEEPSEEK, api_base=base, api_key="k")
    with patch.object(config, "LLM_EFFORT", "xhigh"), patch.object(config, "LLM_THINKING", True):
        client.complete("system", "user")

    assert bodies[0]["reasoning_effort"] == "max"
    assert "extra_body" not in bodies[0]


def test_thinking_off_reaches_deepseek_s_request_body(
    deepseek: tuple[str, list[dict[str, Any]]],
) -> None:
    base, bodies = deepseek
    client = LiteLLMClient(model=DEEPSEEK, api_base=base, api_key="k")
    with patch.object(config, "LLM_EFFORT", "high"), patch.object(config, "LLM_THINKING", False):
        client.complete("system", "user")

    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in bodies[0]


def test_a_real_stream_asks_for_usage_and_records_it(
    deepseek: tuple[str, list[dict[str, Any]]],
) -> None:
    base, bodies = deepseek
    sink: list[UsageRecord] = []
    with use_usage_sink(sink):
        out = "".join(LiteLLMClient(model=DEEPSEEK, api_base=base, api_key="k").stream("s", "u"))

    assert out == "hello"
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert sink == [EXACT]
