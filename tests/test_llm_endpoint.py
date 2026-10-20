"""Tests for the plain-http LLM gateway warning."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from nlqueries.llm import endpoint
from nlqueries.llm.anthropic_client import AnthropicClient
from nlqueries.llm.endpoint import warn_if_plain_http
from nlqueries.llm.litellm_client import LiteLLMClient

_LOGGER = "nlqueries.llm.endpoint"


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts as a new process would: nothing warned yet, no env URL."""
    monkeypatch.setattr(endpoint, "_WARNED", set())
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    yield


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


@pytest.mark.parametrize(
    "url",
    [
        "http://gateway.internal:4000",
        "http://10.0.0.5/v1",
        "HTTP://Gateway.Internal",
        "http://[2001:db8::1]:8080",
        # Names are not resolved: only the literal `localhost` counts.
        "http://localhost.localdomain",
        "http://127.0.0.1.nip.io",
    ],
)
def test_a_non_loopback_http_url_warns(caplog: pytest.LogCaptureFixture, url: str) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        warn_if_plain_http(url, "api_base")
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "api_base" in messages[0]
    assert "plain http" in messages[0]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:4000",
        "http://127.3.2.1",
        "http://localhost:4000/v1",
        "http://LOCALHOST",
        "http://[::1]:4000",
        "https://gateway.internal",
        "https://10.0.0.5:4000/v1",
        "",
        None,
    ],
)
def test_loopback_and_https_do_not_warn(caplog: pytest.LogCaptureFixture, url: str | None) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        warn_if_plain_http(url, "api_base")
    assert _warnings(caplog) == []


def test_warned_once_per_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(3):
            warn_if_plain_http("http://gateway.internal:4000/v1", "api_base")
        warn_if_plain_http("http://other.internal:4000", "api_base")
    messages = _warnings(caplog)
    assert len(messages) == 2
    assert "http://gateway.internal:4000" in messages[0]
    assert "http://other.internal:4000" in messages[1]


def test_credentials_in_the_url_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        warn_if_plain_http("http://user:s3cret@gateway.internal/v1?key=t0ken", "api_base")
    (message,) = _warnings(caplog)
    assert "s3cret" not in message
    assert "t0ken" not in message
    assert "user" not in message
    assert "http://gateway.internal" in message


def _anthropic(**kwargs: str) -> AnthropicClient:
    with (
        patch("nlqueries.llm.anthropic_client.anthropic.Anthropic"),
        patch("nlqueries.llm.anthropic_client.anthropic.AsyncAnthropic"),
    ):
        return AnthropicClient(model="claude-sonnet-5-5", api_key="k", **kwargs)


def test_anthropic_client_warns_on_http_api_base(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _anthropic(api_base="http://gateway.internal:4000")
    (message,) = _warnings(caplog)
    assert message.startswith("api_base ")


def test_anthropic_client_warns_on_http_env_base_url(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gateway.internal:4000")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _anthropic()
    (message,) = _warnings(caplog)
    assert message.startswith("ANTHROPIC_BASE_URL ")


def test_anthropic_client_judges_the_api_base_over_the_env(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK ignores the env var when a base_url is passed, and so does this."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gateway.internal:4000")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _anthropic(api_base="https://gateway.internal")
    assert _warnings(caplog) == []


@pytest.mark.parametrize("url", ["http://127.0.0.1:4000", "https://gateway.internal"])
def test_anthropic_client_quiet_for_loopback_and_https(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _anthropic()
        _anthropic(api_base=url)
    assert _warnings(caplog) == []


def test_litellm_client_warns_on_http_api_base(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        LiteLLMClient(model="openai/gpt-4o", api_base="http://gateway.internal:4000")
    (message,) = _warnings(caplog)
    assert message.startswith("api_base ")


def test_litellm_client_warns_on_http_api_base_in_extra(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        LiteLLMClient(model="openai/gpt-4o", extra={"api_base": "http://gateway.internal"})
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize("url", ["http://127.0.0.1:4000", "https://gateway.internal"])
def test_litellm_client_quiet_for_loopback_and_https(
    caplog: pytest.LogCaptureFixture, url: str
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        LiteLLMClient(model="openai/gpt-4o", api_base=url)
        LiteLLMClient(
            model="openai/gpt-4o", extra={"api_base": "http://gateway.internal"}, api_base=url
        )
    assert _warnings(caplog) == []
