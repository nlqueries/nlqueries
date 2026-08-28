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

#: Names an environment variable in the page. Written without a word-boundary
#: escape on purpose: an earlier version of this file carried two literal
#: backspace bytes where `\\b` was meant, so the pattern matched nothing, the
#: set came out empty, and every terminal rendered the line as though it were
#: fine.
_ENV_NAME = re.compile("NLQ_[A-Z0-9_]+")


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
    looks like a bug in the product; one the code reads and the page omits is a
    control nobody knows to configure.

    The set is derived from the modules rather than listed here. A hand-written
    list is the drift this test exists to catch, and the first version had
    exactly that flaw — adding the admission-control settings did not fail it.
    """
    from nlqueries.auth import admission, mcp_verifier
    from nlqueries.mcp_server import server

    real: set[str] = set()
    for module in (mcp_verifier, admission, server):
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, str) and value.startswith("NLQ_") and name.endswith("_ENV"):
                real.add(value)

    documented = set(_ENV_NAME.findall(text))

    assert real <= documented, f"read by the code, absent from the page: {real - documented}"
    assert documented <= real, f"named on the page, not read: {documented - real}"


def test_the_variable_matcher_finds_something() -> None:
    """The pattern is the load-bearing part of the test above: if it matched
    nothing, `documented` would be empty and the second assertion would pass
    vacuously. That is what a stray backspace did to it once.
    """
    assert _ENV_NAME.findall("set NLQ_MCP_STATIC_TOKEN and NLQ_MCP_RESOURCE_URL") == [
        "NLQ_MCP_STATIC_TOKEN",
        "NLQ_MCP_RESOURCE_URL",
    ]
