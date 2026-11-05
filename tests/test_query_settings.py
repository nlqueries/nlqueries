"""The answer-behaviour settings: where they come from, and what they change.

What these pin: ``query_config.env`` is read from where ``query_config_file``
says and only for query settings, with the environment winning; a request's
bound settings reach every step and no other request; an admin's override is
validated rather than run as the default; and each setting changes what it
says it does -- exact matching leaves literals alone and stops suggesting
TRIM, and switching a check off skips it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from dotenv import dotenv_values
from nlqueries import config
from nlqueries.knowledge.column_profiles import profile_notes
from nlqueries.orchestrator.prompt_assembly import (
    _ANSWER_RULES,
    _EXACT_QUOTED,
    _TRIM_QUOTED,
    answer_rules_for,
)
from nlqueries.query_settings import (
    ENV_VARS,
    QuerySettings,
    current,
    source_of,
    template,
    use_query_settings,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_in_a_fresh_process(
    tmp_path: Path, env: dict[str, str], cwd: Path | None = None
) -> dict[str, Any]:
    """What ``nlqueries.config`` holds when imported with *env*, in *cwd*."""
    code = (
        "import json; from nlqueries import config as c; print(json.dumps({"
        "'file': str(c.QUERY_CONFIG_FILE), 'match': c.MATCH_MODE, 'grain': c.GRAIN_CHECK, "
        "'results': c.RESULT_CHECKS, 'applied': c.QUERY_CONFIG_APPLIED, "
        "'ignored': c.QUERY_CONFIG_IGNORED, "
        "'key': __import__('os').environ.get('OPENAI_API_KEY')}))"
    )
    clean = {k: v for k, v in os.environ.items() if k not in config.QUERY_SETTING_VARS}
    clean.pop("NLQ_QUERY_CONFIG", None)
    clean.pop("OPENAI_API_KEY", None)
    clean.update({"NLQ_STATE_DIR": str(tmp_path / "state"), "PYTHONPATH": str(_ROOT), **env})
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd or tmp_path,
        env=clean,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Where the settings come from
# ---------------------------------------------------------------------------


def test_the_file_sets_query_settings_the_environment_does_not(tmp_path: Path) -> None:
    (tmp_path / "query_config.env").write_text(
        "NLQ_MATCH_MODE=exact\nNLQ_GRAIN_CHECK=false\n", encoding="utf-8"
    )
    seen = _load_in_a_fresh_process(tmp_path, {"NLQ_GRAIN_CHECK": "true"})
    assert seen["file"] == str(tmp_path / "query_config.env")
    assert seen["match"] == "exact"
    # The environment wins, as it does over .env.
    assert seen["grain"] is True
    assert seen["applied"] == ["NLQ_MATCH_MODE"]


def test_the_file_ignores_what_is_not_a_query_setting(tmp_path: Path) -> None:
    (tmp_path / "query_config.env").write_text(
        "OPENAI_API_KEY=not-from-here\nNLQ_RESULT_CHECKS=false\n", encoding="utf-8"
    )
    seen = _load_in_a_fresh_process(tmp_path, {})
    assert seen["key"] is None
    assert seen["ignored"] == ["OPENAI_API_KEY"]
    assert seen["results"] is False


def test_where_the_file_is_read_from(tmp_path: Path) -> None:
    state, named, local = tmp_path / "state", tmp_path / "named.env", tmp_path / "query_config.env"
    # Nothing local: the state directory's.
    assert _load_in_a_fresh_process(tmp_path, {})["file"] == str(state / "query_config.env")
    state.mkdir()
    (state / "query_config.env").write_text("NLQ_MATCH_MODE=exact\n", encoding="utf-8")
    assert _load_in_a_fresh_process(tmp_path, {})["match"] == "exact"
    # One in the working directory comes first.
    local.write_text("NLQ_MATCH_MODE=loose\n", encoding="utf-8")
    assert _load_in_a_fresh_process(tmp_path, {})["file"] == str(local)
    # And one NLQ_QUERY_CONFIG names, before both.
    named.write_text("NLQ_MATCH_MODE=exact\n", encoding="utf-8")
    seen = _load_in_a_fresh_process(tmp_path, {"NLQ_QUERY_CONFIG": str(named)})
    assert (seen["file"], seen["match"]) == (str(named), "exact")


def test_an_unknown_match_mode_is_reported_and_loose_used(tmp_path: Path) -> None:
    assert _load_in_a_fresh_process(tmp_path, {"NLQ_MATCH_MODE": "fuzzy"})["match"] == "loose"


def test_the_template_holds_every_setting_at_its_default() -> None:
    text = template()
    uncommented = "\n".join(
        line[2:] if line.startswith("# NLQ_") else line for line in text.splitlines()
    )
    parsed = {k: v for k, v in dotenv_values(stream=__import__("io").StringIO(uncommented)).items()}
    assert parsed == QuerySettings().as_env()
    assert set(parsed) == set(config.QUERY_SETTING_VARS) == set(ENV_VARS.values())
    # Commented out: a template changes nothing until a line is uncommented.
    assert not any(line.startswith("NLQ_") for line in text.splitlines())


def test_source_of_says_where_a_value_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "QUERY_CONFIG_APPLIED", ["NLQ_GRAIN_CHECK"])
    monkeypatch.setenv("NLQ_PLAN_CHECK", "true")
    monkeypatch.delenv("NLQ_RESULT_CHECKS", raising=False)
    assert source_of("NLQ_GRAIN_CHECK") == "query_config.env"
    assert source_of("NLQ_PLAN_CHECK") == "environment"
    assert source_of("NLQ_RESULT_CHECKS") == "default"


# ---------------------------------------------------------------------------
# The settings a request runs with
# ---------------------------------------------------------------------------


def test_the_defaults_are_the_configuration_as_it_is_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GRAIN_CHECK", False)
    monkeypatch.setattr(config, "MATCH_MODE", "exact")
    settings = current()
    assert (settings.grain_check, settings.match_mode) == (False, "exact")


def test_bound_settings_are_seen_inside_and_not_after() -> None:
    with use_query_settings(QuerySettings(result_checks=False)):
        assert current().result_checks is False
    assert current().result_checks is config.RESULT_CHECKS


def test_two_requests_running_at_once_each_see_their_own() -> None:
    async def request(mode: str) -> list[str]:
        seen = []
        with use_query_settings(QuerySettings(match_mode=mode)):
            for _ in range(3):
                seen.append(current().match_mode)
                await asyncio.sleep(0)
            seen.append(await asyncio.to_thread(lambda: current().match_mode))
        return seen

    async def both() -> list[list[str]]:
        return list(await asyncio.gather(request("exact"), request("loose")))

    assert asyncio.run(both()) == [["exact"] * 4, ["loose"] * 4]


def test_an_override_is_applied_by_name() -> None:
    base = QuerySettings()
    out = base.overridden({"match_mode": " EXACT ", "grain_check": False})
    assert (out.match_mode, out.grain_check, out.plan_check) == ("exact", False, True)
    assert base.match_mode == "loose", "the settings it came from are unchanged"


@pytest.mark.parametrize(
    ("values", "says"),
    [
        ({"match_mod": "exact"}, "not query settings: match_mod"),
        ({"grain_check": "false"}, "grain_check must be true or false"),
        ({"plan_check_min_row_reads": True}, "must be a whole number"),
        ({"plan_check_min_row_reads": -1}, "must not be negative"),
        ({"match_mode": "fuzzy"}, "match_mode must be one of exact, loose"),
        ({"match_mode": 1}, "match_mode must be text"),
    ],
)
def test_a_wrong_override_is_refused(values: dict[str, Any], says: str) -> None:
    with pytest.raises(ValueError, match=says):
        QuerySettings().overridden(values)


def test_exact_matching_grounds_no_literal() -> None:
    assert QuerySettings().grounds_literals is True
    assert QuerySettings(match_mode="exact").grounds_literals is False
    assert QuerySettings(literal_grounding=False).grounds_literals is False


# ---------------------------------------------------------------------------
# What exact matching changes in the prompt
# ---------------------------------------------------------------------------


def test_exact_matching_compares_a_quoted_value_as_written() -> None:
    assert _TRIM_QUOTED in _ANSWER_RULES
    assert _TRIM_QUOTED in answer_rules_for({})
    with use_query_settings(QuerySettings(match_mode="exact")):
        rules = answer_rules_for({})
    assert _TRIM_QUOTED not in rules
    assert _EXACT_QUOTED in rules
    assert "with no TRIM and no change of case" in rules


def test_exact_matching_drops_the_trim_suggestion_but_keeps_the_fact() -> None:
    column = {"name": "code", "profile": {"padded": 12}}
    assert profile_notes(column) == ["12 values have spaces around them: compare TRIM(code)"]
    assert profile_notes(column, trim_hint=False) == ["12 values have spaces around them"]


def test_the_prompt_follows_the_request_match_mode() -> None:
    from nlqueries.orchestrator.prompt_assembly import assemble_prompt

    kb = {
        "schema": {
            "tables": [
                {
                    "name": "flights",
                    "columns": [{"name": "code", "type": "TEXT", "profile": {"padded": 3}}],
                }
            ]
        }
    }
    loose = assemble_prompt("flights from APG", kb).static_system
    with use_query_settings(QuerySettings(match_mode="exact")):
        exact = assemble_prompt("flights from APG", kb).static_system
    assert "compare TRIM(code)" in loose and "compare TRIM(code)" not in exact
    assert "3 values have spaces around them" in exact
    assert _EXACT_QUOTED in exact and _TRIM_QUOTED in loose


# ---------------------------------------------------------------------------
# Switching a check off
# ---------------------------------------------------------------------------


def test_with_result_checks_off_a_result_is_returned_as_it_ran() -> None:
    from unittest.mock import MagicMock

    from nlqueries.connectors.base import QueryResult
    from nlqueries.orchestrator.orchestrator import _second_look
    from nlqueries.orchestrator.sql_generation import SQLGenerationResult

    first = SQLGenerationResult(
        sql="SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id",
        is_valid=True,
        validation_error=None,
        dialect="sqlite",
        attempt_count=1,
    )
    repeated = QueryResult(
        columns=["name"], rows=[["Ann"], ["Ann"]], row_count=2, execution_time_ms=1, error=None
    )
    llm = MagicMock()

    def look() -> tuple[Any, Any, list[str]]:
        return asyncio.run(
            _second_look(
                first,
                repeated,
                connector=MagicMock(),
                question="which customers ordered",
                kb={},
                dialect="sqlite",
                llm=llm,
                system="s",
                timeout_seconds=None,
                deadline=None,
            )
        )

    with use_query_settings(QuerySettings(result_checks=False)):
        result, qr, notes = look()
    assert (result, qr, notes) == (first, repeated, [])
    llm.acomplete.assert_not_called()


def test_with_literal_grounding_on_but_exact_matching_nothing_is_looked_up() -> None:
    from unittest.mock import MagicMock

    from nlqueries.orchestrator.sql_generation import validate_and_repair

    lookups = MagicMock()
    kb = {
        "schema": {"tables": [{"name": "schools", "columns": [{"name": "status", "type": "TEXT"}]}]}
    }
    with use_query_settings(QuerySettings(match_mode="exact")):
        result = asyncio.run(
            validate_and_repair(
                "SELECT status FROM schools WHERE status = 'legal'",
                kb,
                "sqlite",
                MagicMock(),
                lookups=lookups,
            )
        )
    assert "'legal'" in result.sql
    assert not lookups.mock_calls


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_query_config_shows_the_settings_and_writes_a_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries.cli.main import cli

    target = tmp_path / "query_config.env"
    monkeypatch.setattr(config, "QUERY_CONFIG_FILE", target)
    runner = CliRunner()
    shown = runner.invoke(cli, ["query-config"])
    assert shown.exit_code == 0, shown.output
    assert "NLQ_MATCH_MODE" in shown.output and "(not present)" in shown.output
    assert not target.exists()

    wrote = runner.invoke(cli, ["query-config", "--init"])
    assert wrote.exit_code == 0, wrote.output
    assert target.read_text(encoding="utf-8") == template()

    target.write_text("NLQ_MATCH_MODE=exact\n", encoding="utf-8")
    again = runner.invoke(cli, ["query-config", "--init"])
    assert "already exists; not overwritten" in again.output
    assert target.read_text(encoding="utf-8") == "NLQ_MATCH_MODE=exact\n"


# ---------------------------------------------------------------------------
# The semantic cache keeps answers apart by settings
# ---------------------------------------------------------------------------


def test_the_fingerprint_changes_with_any_setting() -> None:
    base = QuerySettings()
    assert base.fingerprint() == QuerySettings().fingerprint()
    others = {
        QuerySettings(match_mode="exact").fingerprint(),
        QuerySettings(result_checks=False).fingerprint(),
        QuerySettings(grain_check=False).fingerprint(),
        QuerySettings(plan_check=False).fingerprint(),
        QuerySettings(plan_check_min_row_reads=5).fingerprint(),
        QuerySettings(literal_grounding=False).fingerprint(),
        QuerySettings(literal_grounding_repair=False).fingerprint(),
    }
    assert len(others) == 7 and base.fingerprint() not in others


def test_an_answer_cached_under_loose_matching_is_a_miss_under_exact() -> None:
    from nlqueries.cache.envelope import binding_for_agent, sign, verify

    key = b"k" * 32
    payload = {"question": "flights from APG", "answer": "a", "sql": "SELECT 1", "kind": "answer"}
    loose = binding_for_agent("agent1", "sqlite")
    with use_query_settings(QuerySettings(match_mode="exact")):
        exact = binding_for_agent("agent1", "sqlite")
    assert loose.settings_fingerprint != exact.settings_fingerprint
    signed = sign(payload, loose, key)
    assert verify(signed, loose, key)
    assert not verify(signed, exact, key)
