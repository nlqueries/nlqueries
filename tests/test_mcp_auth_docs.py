"""
That the authentication documentation describes the code that exists.

An operator configuring this has the page and nothing else: they cannot see
`Action`, and a wrong action name in a grants file is refused with no working
example to compare against. A documented vocabulary that has drifted from the
real one is the failure this whole stream keeps turning up, so the page is
checked against the code rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from nlqueries.auth.authorizer import load_grants
from nlqueries.auth.principal import AGENTLESS_ACTIONS, TOOL_ACTIONS, Action

DOC = Path(__file__).resolve().parents[1] / "docs" / "mcp-authentication.md"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def documented_rows(text: str) -> list[tuple[str, str, str]]:
    """The action table: (action, tool, names-an-agent)."""
    rows = re.findall(r"^\| `([a-z:]+)` \| `(\w+)` \| (yes|no) \|$", text, re.M)
    assert rows, "the action table could not be found in the page"
    return rows


def test_every_action_is_documented(documented_rows) -> None:
    documented = {action for action, _, _ in documented_rows}

    assert documented == {str(action) for action in Action}


def test_each_action_is_documented_against_the_right_tool(documented_rows) -> None:
    """A table that named the wrong tool would send an operator to grant the
    wrong permission."""
    for action, tool, _ in documented_rows:
        assert str(TOOL_ACTIONS[tool]) == action


def test_the_agent_scoped_column_matches_the_code(documented_rows) -> None:
    for action, _, names_an_agent in documented_rows:
        agentless = Action(action) in AGENTLESS_ACTIONS

        assert agentless == (names_an_agent == "no"), action


def test_the_example_grants_file_is_valid(text: str, tmp_path: Path) -> None:
    """Loaded through the real parser. An example that would be refused is worse
    than none: it is the first thing anyone copies."""
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    assert blocks, "no YAML example found in the page"

    example = next(b for b in blocks if "grants:" in b)
    path = tmp_path / "grants.yaml"
    path.write_text(example, encoding="utf-8")

    grants = load_grants(path)

    assert {g.subject for g in grants} == {"alice@example.com", "monitoring", "operator"}


def test_the_example_grants_only_actions_that_exist(text: str) -> None:
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    example = next(b for b in blocks if "grants:" in b)
    parsed = yaml.safe_load(example)

    named = {a for grant in parsed["grants"] for a in grant["actions"]} - {"*"}

    assert named <= {str(action) for action in Action}


def test_the_environment_variables_named_are_the_ones_read(text: str) -> None:
    """A page naming a variable the code does not read is a support case that
    looks like a bug in the product."""
    from nlqueries.auth import mcp_verifier
    from nlqueries.mcp_server import server

    real = {
        mcp_verifier.RESOURCE_URL_ENV,
        mcp_verifier.OIDC_DISCOVERY_ENV,
        mcp_verifier.OIDC_CLIENT_ID_ENV,
        mcp_verifier.STATIC_TOKEN_ENV,
        mcp_verifier.STATIC_TOKEN_FILE_ENV,
        mcp_verifier.STATIC_SUBJECT_ENV,
        server._GRANTS_FILE_ENV,
        server._ALLOW_UNAUTHENTICATED_ENV,
    }
    documented = set(re.findall(r"\bNLQ_[A-Z_]+\b", text))

    assert real <= documented, f"undocumented: {real - documented}"
    assert documented <= real | {"NLQ_ALLOW_INSECURE_BIND"}, f"not read: {documented - real}"
