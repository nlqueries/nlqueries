"""
The shipped compose stack is configured to start, and to answer, with MCP auth.

The image serves the MCP server over SSE, and a networked transport refuses to
start without a verifier (#160). The compose file is the only place the
quickstart configures one, so it is checked here rather than trusted: a
required token, a grants file the server can load, a grant for the subject the
token maps to, and a start command that is still the image's own.

A mismatch between the token's subject and the grant would not fail at start --
the server would authenticate every caller and then deny every call -- which is
why the grant is generated from the same variable and checked against it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from nlqueries.auth.authorizer import load_grants
from nlqueries.auth.principal import Action

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def service() -> dict:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["nlqueries-core"]


def _default(value: str) -> str:
    """The value compose would use when the variable is not set: `${X:-d}` is d."""
    match = re.fullmatch(r"\$\{(\w+):-(.*)\}", value)
    return match.group(2) if match else value


def _script(service: dict) -> str:
    command = service["command"]
    assert command[:2] == ["sh", "-c"], "the grants file is written by a shell step"
    # Compose turns `$$` into `$` before the shell sees it.
    return command[2].replace("$$", "$")


def _grants_text(script: str, subject: str) -> tuple[str, str]:
    """(path written, file contents), as the shell would produce them."""
    match = re.search(r"cat > (\S+) <<GRANTS\n(.*?)\nGRANTS\n", script, re.S)
    assert match, "the grants heredoc is missing or quoted (a quoted one would not expand)"
    return match.group(1), match.group(2).replace("${NLQ_MCP_STATIC_SUBJECT}", subject)


def test_the_token_is_required(service: dict) -> None:
    token = service["environment"]["NLQ_MCP_STATIC_TOKEN"]
    assert token.startswith("${NLQ_MCP_STATIC_TOKEN:?"), (
        "compose must refuse to start without a token, as it does without QDRANT_API_KEY"
    )


def test_a_resource_url_is_set(service: dict) -> None:
    assert _default(service["environment"]["NLQ_MCP_RESOURCE_URL"])


@pytest.mark.parametrize("subject", [None, "alice@example.com"])
def test_the_generated_grants_load_and_cover_the_tokens_subject(
    service: dict, subject: str | None, tmp_path: Path
) -> None:
    """Default and overridden subject alike: the grant is for whoever the
    token authenticates as, and it grants every action on every agent."""
    env = service["environment"]
    subject = subject or _default(env["NLQ_MCP_STATIC_SUBJECT"])
    written_to, text = _grants_text(_script(service), subject)

    assert written_to == _default(env["NLQ_MCP_GRANTS_FILE"])
    path = tmp_path / "grants.yaml"
    path.write_text(text, encoding="utf-8")
    grants = load_grants(path)

    mine = [g for g in grants if g.subject == subject]
    assert mine, f"no grant for the token's subject {subject!r}"
    for action in Action:
        assert any(g.covers(action, "any-agent") for g in mine), action


def test_the_subject_and_grants_path_can_be_set_from_env_file(service: dict) -> None:
    """`environment:` beats `env_file:`, so a fixed literal there would make a
    value set in .env silently ignored."""
    env = service["environment"]
    for name in ("NLQ_MCP_STATIC_SUBJECT", "NLQ_MCP_GRANTS_FILE"):
        assert env[name].startswith("${" + name + ":-"), name


def test_the_server_starts_as_the_image_does(service: dict) -> None:
    """The override replaces the image's CMD, so the two must say the same."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmd = re.search(r"^CMD (\[.*\])$", dockerfile, re.M)
    assert cmd, "the Dockerfile has no exec-form CMD"
    last = _script(service).strip().splitlines()[-1]
    assert last == "exec " + " ".join(json.loads(cmd.group(1)))
