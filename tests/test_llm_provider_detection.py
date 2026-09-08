"""Provider and fast-model detection from the environment.

These functions run once at import time to set ``config.LLM_PROVIDER`` and
``config.LLM_MODEL_FAST``, so a wrong answer here is a deployment that fails on
its first question with an error naming a model nobody configured. The cases
below are about *ordering*: each new branch has to sit ahead of an older one
that would otherwise swallow it.
"""

from __future__ import annotations

import pytest
from nlqueries import config

_BEDROCK = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from no LLM env at all.

    The developer running this very likely has ANTHROPIC_API_KEY exported, which
    is one of the values under test.
    """
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_MODEL_FAST",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_explicit_bedrock_provider_is_routed_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`bedrock` is not a client of its own; without this it raises Unknown LLM provider."""
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("LLM_MODEL", _BEDROCK)
    assert config._detect_provider() == "litellm"


def test_the_bedrock_provider_name_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "Bedrock")
    monkeypatch.setenv("LLM_MODEL", _BEDROCK)
    assert config._detect_provider() == "litellm"


def test_detection_never_raises_while_the_module_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misconfigured provider must not take the CLI down with it.

    `nlqueries.config` is imported by every command, including `connect` and
    `extract-schema`, which never touch an LLM — and by the diagnostics someone
    would run to find the misconfiguration. Refusing the bedrock/model mismatch
    is right, but `get_llm_client` is where it belongs: the first point at which
    an LLM is actually about to be used, so nothing has been sent yet.
    """
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")

    assert config._detect_provider() == "litellm"

    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    assert config._detect_provider() == "litellm"


def test_another_explicit_provider_is_still_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert config._detect_provider() == "anthropic"


def test_a_bedrock_model_wins_over_an_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordering guard.

    A Bedrock deployment commonly still has ANTHROPIC_API_KEY in its environment
    for something else. If the key is consulted first, a `bedrock/...` model id
    goes to the Anthropic SDK and fails at the API, far from its cause.
    """
    monkeypatch.setenv("LLM_MODEL", _BEDROCK)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-set-for-something-else")
    assert config._detect_provider() == "litellm"


def test_an_anthropic_key_alone_still_selects_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control for the case above: nothing else changed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert config._detect_provider() == "anthropic"


def test_an_openai_key_alone_still_selects_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert config._detect_provider() == "litellm"


def test_the_fast_model_stays_on_bedrock_despite_an_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second ordering guard.

    `claude-haiku-...` and `openai/gpt-4o-mini` are both invalid model ids on
    Bedrock. Either one breaks every fast call — the intent classifier and the
    follow-up resolver — while the main model keeps working, which reads as a
    broken product rather than one unset variable.
    """
    monkeypatch.setenv("LLM_MODEL", _BEDROCK)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-set-for-something-else")
    assert config._detect_fast_model("litellm") == _BEDROCK


def test_an_explicit_fast_model_still_wins_on_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reusing the default model is the fallback, not a policy."""
    monkeypatch.setenv("LLM_MODEL", _BEDROCK)
    monkeypatch.setenv("LLM_MODEL_FAST", "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    assert (
        config._detect_fast_model("litellm")
        == "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )


def test_the_openai_fast_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The negative control: no bedrock model, so the old branch still applies."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert config._detect_fast_model("litellm") == "openai/gpt-4o-mini"


def test_the_anthropic_fast_default_is_unchanged() -> None:
    assert config._detect_fast_model("anthropic") == "claude-haiku-4-5-20251001"
