"""
CLI preflights on a Bedrock deployment.

Three CLI paths gated on the *presence of an API key* rather than on whether an
LLM is reachable, so the deployment the Bedrock documentation recommends most --
an IAM role and no key at all -- was refused by commands that would have worked.
``doctor`` was the worst of them: it reported the LLM as misconfigured on a
working host, in the command someone runs precisely to find out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.connectors.base import DatabaseConnector, QueryRecord, QueryResult, SchemaSpec

_BEDROCK = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """An IAM host has no LLM key, and the developer running this probably does.

    Both the environment and the resolved constants, because they are only the
    same in a fresh process: `config.ANTHROPIC_API_KEY` was read at import and
    would otherwise keep a key this fixture believes it has removed.
    """
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER_CONFIGURED", "")
    monkeypatch.setattr(config, "LLM_MODEL", "claude-sonnet-4-5")


def test_a_bedrock_model_counts_as_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """boto3 supplies them: an instance profile, a task role, IRSA, a profile."""
    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)

    assert config.llm_credentials_available() is True


def test_naming_bedrock_counts_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER_CONFIGURED", "bedrock")

    assert config.llm_credentials_available() is True


def test_a_key_still_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant")

    assert config.llm_credentials_available() is True


def test_an_openai_key_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """It did not, in the --describe-columns branch, which checked LLM_API_KEY --
    a name this product does not define anywhere."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    assert config.llm_credentials_available() is True


def test_nothing_configured_is_still_nothing() -> None:
    """The negative control: the check must still be able to say no."""
    assert config.llm_credentials_available() is False


def test_a_non_bedrock_model_alone_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other negative control. Only Bedrock authenticates without a key."""
    monkeypatch.setattr(config, "LLM_MODEL", "openai/gpt-4o")

    assert config.llm_credentials_available() is False


def test_it_reads_the_current_configuration_rather_than_a_cached_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI's checks run long after import; a value computed once would lie."""
    assert config.llm_credentials_available() is False
    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)
    assert config.llm_credentials_available() is True


def test_doctor_does_not_report_a_bedrock_host_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command someone runs to find the problem must not invent one.

    The client is stubbed rather than left to resolve. `_check_llm` goes on to
    make a real completion, and it builds that client from the import-time
    `config`, not from the environment this test patches -- so unstubbed, on any
    machine that had a key exported when `nlqueries.config` was imported, this
    test would send a real billed request. The first version of it did.

    Asserting `status == "ok"` rather than "not this particular failure" for the
    same reason: the weaker form is satisfied by any other failure, including the
    outbound call failing.
    """
    from nlqueries.cli import main as cli_main

    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)

    with patch("nlqueries.llm.get_llm_client", return_value=_StubLLM()):
        result = cli_main._check_llm()

    assert result.status == "ok", f"doctor faulted a working Bedrock host: {result.detail}"


def test_doctor_still_reports_a_host_with_nothing_configured() -> None:
    """The negative control for the above."""
    from nlqueries.cli import main as cli_main

    result = cli_main._check_llm()

    assert result.status == "fail"
    assert "no credentials" in (result.detail or "")


_FAKE_CFG: dict[str, object] = {
    "db_type": "postgres",
    "url": "postgresql://user:pass@localhost/testdb",
}


class _StubConnector(DatabaseConnector):
    """Canned query history; connect and schema are no-ops."""

    def connect(self, credentials: dict[str, object]) -> None:
        pass

    def test_connection(self) -> bool:
        return True

    def extract_schema(self) -> SchemaSpec:
        raise RuntimeError("stub -- no schema")

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        return [
            QueryRecord(
                sql=f"SELECT id FROM users WHERE status = 'active_{i}'",
                execution_count=5,
                avg_duration_ms=None,
                last_executed=None,
            )
            for i in range(20)
        ]

    def _execute_query(self, sql: str) -> QueryResult:
        return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)


class _StubLLM:
    """Enough of an LLM client for the annotator, with nothing behind it."""

    def complete(self, *_a: object, **_kw: object) -> str:
        return "{}"


def _run_process_history(tmp_path: object, *, annotate: bool) -> tuple[int, list[str]]:
    """`process-history` with the connector stubbed, exercising the real preflight."""
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    err_console = MagicMock()
    runner = CliRunner()
    args = [
        "process-history",
        "dvdrental",
        "--days",
        "30",
        "--min-executions",
        "1",
        "--annotate" if annotate else "--no-annotate",
        "--no-embed",
    ]
    with (
        patch("nlqueries.cli.main.console", MagicMock()),
        patch("nlqueries.cli.main.err_console", err_console),
        patch("nlqueries.cli.main._require_connector", return_value=_FAKE_CFG),
        patch("nlqueries.cli.main._resolve_alias", side_effect=lambda x: x),
        patch("nlqueries.cli.main._load_password", return_value=None),
        patch.dict(
            "nlqueries.connectors.CONNECTOR_REGISTRY",
            {"postgres": lambda: _StubConnector()},
            clear=False,
        ),
        patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path),
        # Never reach AWS. This is a test about the preflight, and a unit test
        # that depends on the machine's AWS credentials is a test about the
        # machine -- running it unpatched here really did attempt an SSO refresh.
        patch("nlqueries.llm.get_llm_client", return_value=_StubLLM()),
        # Both names, per the guard in tests/conftest.py: `cli.main` binds
        # CONNECTORS_FILE at import, so redirecting only `config`'s is silent
        # and leaves the operator's own registry in the line of fire.
        patch("nlqueries.config.CONNECTORS_FILE", tmp_path / "connectors.yaml"),
        patch("nlqueries.cli.main.CONNECTORS_FILE", tmp_path / "connectors.yaml"),
    ):
        result = runner.invoke(cli, args)
    return result.exit_code, [str(c) for c in err_console.print.call_args_list]


def test_process_history_annotate_is_not_refused_on_a_bedrock_host(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--annotate` is the default, so this refusal blocked the ordinary command."""
    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)

    exit_code, errors = _run_process_history(tmp_path, annotate=True)

    refusals = [e for e in errors if "requires an LLM credential" in e]
    assert not refusals, f"refused a working Bedrock host: {refusals}"
    assert exit_code == 0, f"did not get past the preflight: {errors}"


def test_process_history_annotate_is_still_refused_with_nothing_configured(
    tmp_path: object,
) -> None:
    """The negative control: the preflight must still be able to refuse."""
    exit_code, errors = _run_process_history(tmp_path, annotate=True)

    assert exit_code == 1
    assert any("requires an LLM credential" in e for e in errors)


def test_doctor_does_not_contradict_itself_on_a_bedrock_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` printed a passing LLM line and a Config line warning of no key.

    Two checks answering the same question two ways is worse than either being
    wrong alone: it leaves the reader with no way to tell which to believe.
    """
    from nlqueries.cli import main as cli_main

    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)

    config_result = cli_main._check_config()

    assert "no LLM credentials set" not in (config_result.detail or "")


def test_doctor_still_warns_when_there_are_no_credentials_at_all() -> None:
    """The negative control: the Config line must still be able to warn."""
    from nlqueries.cli import main as cli_main

    result = cli_main._check_config()

    assert result.status == "warn"
    assert "no LLM credentials set" in (result.detail or "")


def test_the_mcp_health_check_accepts_a_bedrock_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It reported "no ANTHROPIC_API_KEY set", which was already wrong for OpenAI."""
    from nlqueries.config import llm_credentials_available

    monkeypatch.setattr(config, "LLM_MODEL", _BEDROCK)

    assert llm_credentials_available() is True


def test_the_mcp_health_check_accepts_an_openai_only_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing half of that bug: an OpenAI deployment had no Anthropic key."""
    from nlqueries.config import llm_credentials_available

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    assert llm_credentials_available() is True
