"""Tests for the LLMClient abstract interface and AnthropicClient implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest
from nlqueries.llm import get_llm_client
from nlqueries.llm.anthropic_client import _BASE_DELAY, _MAX_RETRIES, AnthropicClient
from nlqueries.llm.client import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_error() -> anthropic.RateLimitError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    return anthropic.RateLimitError("rate limited", response=resp, body={})


def _make_mock_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_mock_stream(tokens: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.text_stream = iter(tokens)
    return ctx


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


def test_llm_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract"):
        LLMClient()  # type: ignore[abstract]


@pytest.mark.parametrize("missing_method", ["complete", "stream"])
def test_subclass_missing_abstract_method_cannot_be_instantiated(missing_method):
    all_methods: dict = {
        "complete": lambda self, system, user, max_tokens=1024: "",
        "stream": lambda self, system, user: iter([]),
    }
    all_methods.pop(missing_method)
    Incomplete = type("Incomplete", (LLMClient,), all_methods)
    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_concrete_subclass_can_be_instantiated():
    class Concrete(LLMClient):
        def complete(self, system, user, max_tokens=1024):
            return "ok"

        def stream(self, system, user):
            yield "ok"

    assert isinstance(Concrete(), LLMClient)


# ---------------------------------------------------------------------------
# AnthropicClient — construction
# ---------------------------------------------------------------------------


def test_anthropic_client_uses_default_model_from_config():
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        client = AnthropicClient()
    assert client._model == "claude-sonnet-4-5"


def test_anthropic_client_accepts_custom_model():
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        client = AnthropicClient(model="claude-haiku-4-5")
    assert client._model == "claude-haiku-4-5"


def test_anthropic_client_is_llm_client_subclass():
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        client = AnthropicClient()
    assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# AnthropicClient.complete() — happy path
# ---------------------------------------------------------------------------


def test_complete_returns_string():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.return_value = _make_mock_response("hello world")

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        result = AnthropicClient().complete(system="sys", user="usr")

    assert result == "hello world"
    assert isinstance(result, str)


def test_complete_passes_system_and_user_to_api():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.return_value = _make_mock_response("ok")

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        AnthropicClient().complete(system="my system", user="my question")

    call_kwargs = mock_sdk.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "my system"
    assert call_kwargs["messages"][0]["content"] == "my question"


# ---------------------------------------------------------------------------
# AnthropicClient.complete() — retry on RateLimitError
# ---------------------------------------------------------------------------


def test_complete_retries_correct_number_of_times():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = _make_rate_limit_error()

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
        patch("nlqueries.llm.anthropic_client.time.sleep"),
        pytest.raises(anthropic.RateLimitError),
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        AnthropicClient().complete(system="sys", user="usr")

    assert mock_sdk.messages.create.call_count == _MAX_RETRIES + 1


def test_complete_sleeps_between_retries():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = _make_rate_limit_error()
    mock_sleep = MagicMock()

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
        patch("nlqueries.llm.anthropic_client.time.sleep", mock_sleep),
        pytest.raises(anthropic.RateLimitError),
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        AnthropicClient().complete(system="sys", user="usr")

    assert mock_sleep.call_count == _MAX_RETRIES


def test_complete_uses_exponential_backoff():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = _make_rate_limit_error()
    mock_sleep = MagicMock()

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
        patch("nlqueries.llm.anthropic_client.time.sleep", mock_sleep),
        pytest.raises(anthropic.RateLimitError),
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        AnthropicClient().complete(system="sys", user="usr")

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    expected = [_BASE_DELAY * (2**i) for i in range(_MAX_RETRIES)]
    assert delays == expected


def test_complete_succeeds_after_transient_rate_limit():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = [
        _make_rate_limit_error(),
        _make_mock_response("recovered"),
    ]

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
        patch("nlqueries.llm.anthropic_client.time.sleep"),
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        result = AnthropicClient().complete(system="sys", user="usr")

    assert result == "recovered"
    assert mock_sdk.messages.create.call_count == 2


def test_complete_raises_rate_limit_after_all_retries_exhausted():
    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = _make_rate_limit_error()

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
        patch("nlqueries.llm.anthropic_client.time.sleep"),
        pytest.raises(anthropic.RateLimitError),
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        AnthropicClient().complete(system="sys", user="usr")


# ---------------------------------------------------------------------------
# AnthropicClient.stream() — token yielding
# ---------------------------------------------------------------------------


def test_stream_yields_tokens():
    mock_sdk = MagicMock()
    mock_sdk.messages.stream.return_value = _make_mock_stream(["hello", " ", "world"])

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        tokens = list(AnthropicClient().stream(system="sys", user="usr"))

    assert tokens == ["hello", " ", "world"]


def test_stream_yields_str_tokens():
    mock_sdk = MagicMock()
    mock_sdk.messages.stream.return_value = _make_mock_stream(["tok1", "tok2", "tok3"])

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        for token in AnthropicClient().stream(system="sys", user="usr"):
            assert isinstance(token, str)


def test_stream_empty_response():
    mock_sdk = MagicMock()
    mock_sdk.messages.stream.return_value = _make_mock_stream([])

    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic", return_value=mock_sdk),
        patch("nlqueries.llm.anthropic_client.config") as mock_cfg,
    ):
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.LLM_MODEL = "claude-sonnet-4-5"
        tokens = list(AnthropicClient().stream(system="sys", user="usr"))

    assert tokens == []


# ---------------------------------------------------------------------------
# get_llm_client() factory
# ---------------------------------------------------------------------------


def test_get_llm_client_returns_anthropic_client_by_default():
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.config") as mock_ac_cfg,
        patch("nlqueries.llm.config") as mock_llm_cfg,
    ):
        mock_ac_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_ac_cfg.LLM_MODEL = "claude-sonnet-4-5"
        mock_llm_cfg.LLM_PROVIDER = "anthropic"
        client = get_llm_client()

    assert isinstance(client, AnthropicClient)


def test_get_llm_client_raises_on_unknown_provider():
    with (
        patch("nlqueries.llm.config") as mock_llm_cfg,
        pytest.raises(ValueError, match="Unknown LLM provider"),
    ):
        mock_llm_cfg.LLM_PROVIDER = "unknown_provider"
        get_llm_client()


# ---------------------------------------------------------------------------
# LiteLLMClient — construction
# ---------------------------------------------------------------------------


def test_litellm_client_uses_default_model_from_config():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with patch("nlqueries.llm.litellm_client.config") as mock_cfg:
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        client = LiteLLMClient()
    assert client._model == "anthropic/claude-sonnet-4-5"


def test_litellm_client_accepts_custom_model():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with patch("nlqueries.llm.litellm_client.config") as mock_cfg:
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        client = LiteLLMClient(model="openai/gpt-4o")
    assert client._model == "openai/gpt-4o"


def test_litellm_client_is_llm_client_subclass():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with patch("nlqueries.llm.litellm_client.config") as mock_cfg:
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        client = LiteLLMClient()
    assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# LiteLLMClient.complete()
# ---------------------------------------------------------------------------


def _make_litellm_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_litellm_stream(tokens: list[str]) -> list[MagicMock]:
    chunks = []
    for token in tokens:
        delta = MagicMock()
        delta.content = token
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        chunks.append(chunk)
    return chunks


def test_litellm_complete_returns_string():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with (
        patch(
            "nlqueries.llm.litellm_client.litellm.completion",
            return_value=_make_litellm_response("hello"),
        ),
        patch("nlqueries.llm.litellm_client.config") as mock_cfg,
    ):
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        result = LiteLLMClient().complete(system="sys", user="usr")

    assert result == "hello"
    assert isinstance(result, str)


def test_litellm_complete_passes_system_and_user_as_messages():
    from nlqueries.llm.litellm_client import LiteLLMClient

    mock_completion = MagicMock(return_value=_make_litellm_response("ok"))
    with (
        patch("nlqueries.llm.litellm_client.litellm.completion", mock_completion),
        patch("nlqueries.llm.litellm_client.config") as mock_cfg,
    ):
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        LiteLLMClient().complete(system="my system", user="my question")

    messages = mock_completion.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "my system"}
    assert messages[1] == {"role": "user", "content": "my question"}


def test_litellm_complete_returns_empty_string_on_none_content():
    from nlqueries.llm.litellm_client import LiteLLMClient

    response = _make_litellm_response(None)
    response.choices[0].message.content = None
    with (
        patch("nlqueries.llm.litellm_client.litellm.completion", return_value=response),
        patch("nlqueries.llm.litellm_client.config") as mock_cfg,
    ):
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        result = LiteLLMClient().complete(system="sys", user="usr")

    assert result == ""


# ---------------------------------------------------------------------------
# LiteLLMClient.stream()
# ---------------------------------------------------------------------------


def test_litellm_stream_yields_tokens():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with (
        patch(
            "nlqueries.llm.litellm_client.litellm.completion",
            return_value=_make_litellm_stream(["hello", " ", "world"]),
        ),
        patch("nlqueries.llm.litellm_client.config") as mock_cfg,
    ):
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        tokens = list(LiteLLMClient().stream(system="sys", user="usr"))

    assert tokens == ["hello", " ", "world"]


def test_litellm_stream_skips_none_deltas():
    from nlqueries.llm.litellm_client import LiteLLMClient

    chunks = _make_litellm_stream(["tok1", "tok2"])
    # insert a chunk with None content between the two real tokens
    null_delta = MagicMock()
    null_delta.content = None
    null_choice = MagicMock()
    null_choice.delta = null_delta
    null_chunk = MagicMock()
    null_chunk.choices = [null_choice]
    chunks.insert(1, null_chunk)

    with (
        patch("nlqueries.llm.litellm_client.litellm.completion", return_value=chunks),
        patch("nlqueries.llm.litellm_client.config") as mock_cfg,
    ):
        mock_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        tokens = list(LiteLLMClient().stream(system="sys", user="usr"))

    assert tokens == ["tok1", "tok2"]


# ---------------------------------------------------------------------------
# get_llm_client() — litellm provider
# ---------------------------------------------------------------------------


def test_get_llm_client_returns_litellm_client_when_configured():
    from nlqueries.llm.litellm_client import LiteLLMClient

    with (
        patch("nlqueries.llm.litellm_client.config") as mock_litellm_cfg,
        patch("nlqueries.llm.config") as mock_llm_cfg,
    ):
        mock_litellm_cfg.LLM_MODEL = "anthropic/claude-sonnet-4-5"
        mock_llm_cfg.LLM_PROVIDER = "litellm"
        client = get_llm_client()

    assert isinstance(client, LiteLLMClient)
