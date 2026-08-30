"""
Every published port names the interface it is published on (SEC-06, again).

`test_mcp_bind_guard` records how this failed the first time: the MCP CLI
defaulted `--host` to `0.0.0.0`, and `docs/cli-reference.md` presented that
binding as ordinary usage, so the documented quickstart published a
database-query API to whatever could route to the host. The code was fixed and
guarded.

The same mistake was still sitting in `docs/qdrant-setup.md`, which told anyone
following the recommended local setup to run `-p 6333:6333 -p 6334:6334` with
`--restart unless-stopped` and no API key -- an unauthenticated vector store,
published on every interface, coming back after every reboot.

No test over the code could have caught it. `require_qdrant_auth` reads
`QDRANT_URL`, and `http://localhost:6333` is loopback however the container is
published, so the app's own rule passes while the container is open to the
network. The bind exists only in a document and a compose file, which is why
this check reads those instead.

The rule is the one `test_mcp_bind_guard` already states: naming an interface is
a choice, and `0.0.0.0` spelled out is a decision a reviewer can see and weigh.
A bare `host:container` pair is the default that nobody chose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: Directories that are not ours to police.
_SKIP = {".venv", "venv", "node_modules", ".git", "site-packages", ".mypy_cache"}

#: A `-p` / `--publish` argument. Restricted to values that contain a
#: `digit:digit` pair, so `mkdir -p /some/path` is not mistaken for a port
#: mapping -- an earlier draft flagged exactly that.
_PUBLISH_FLAG = re.compile(r"(?:--publish|-p)[ =](\S+)")
_PORT_PAIR = re.compile(r"\d+:\d+")

#: Fenced code blocks, of any language. Only these are checked: prose has to
#: stay free to name the wrong binding in order to explain it, and the sentence
#: below the Qdrant command does exactly that. The cost is that a hazardous
#: command written inline, in backticks, is not seen -- worth accepting, since
#: what people copy is the block.
_FENCED = re.compile(r"```[\w-]*\n(.*?)```", re.S)


def _repo_files(*patterns: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for path in ROOT.rglob(pattern):
            if any(part in _SKIP for part in path.parts):
                continue
            found.append(path)
    return sorted(found)


def _names_an_interface(entry: object) -> bool:
    """Whether *entry* specifies a host address.

    `5433:5432` is host-port to container-port and publishes on every
    interface. `127.0.0.1:5433:5432` and `${BIND_ADDR:-127.0.0.1}:5433:5432`
    name one. Counting separators rather than parsing an address keeps the
    variable form working, which is how the top-level compose file writes it.

    The long syntax is a mapping, and only `host_ip` names an address there --
    Docker defaults it to `0.0.0.0`. It is handled explicitly because the
    obvious shortcut is wrong in the worst direction: `str({'target': 5432,
    'published': 5433})` contains two colons, so stringifying a long-syntax
    entry passes this check without an address ever having been written. Moving
    a service to the long form is the natural thing to do when adding `mode` or
    `protocol`, and it would have republished on every interface with this guard
    reporting success.
    """
    if isinstance(entry, dict):
        return bool(entry.get("host_ip"))
    return str(entry).count(":") >= 2


DOCS = _repo_files("*.md")
COMPOSE = _repo_files("docker-compose*.yml", "docker-compose*.yaml")


def test_the_search_found_the_files_it_is_meant_to_check() -> None:
    """A guard over an empty file list passes and means nothing.

    This is not hypothetical: the first run of this module matched no compose
    files at all, because the benchmark stack is named `.yaml` and the pattern
    only had `.yml`, and it reported success.
    """
    names = {path.name for path in COMPOSE}

    assert "docker-compose.yml" in names
    assert "docker-compose.yaml" in names
    assert any(path.name == "qdrant-setup.md" for path in DOCS)


@pytest.mark.parametrize("path", COMPOSE, ids=lambda p: str(p.relative_to(ROOT)))
def test_compose_files_publish_only_on_a_named_interface(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    unqualified = [
        f"{service}: {entry}"
        for service, spec in (document.get("services") or {}).items()
        for entry in (spec.get("ports") or [])
        if not _names_an_interface(entry)
    ]

    assert not unqualified, (
        f"{path.relative_to(ROOT)} publishes on every interface: {unqualified}. "
        "Use ${BIND_ADDR:-127.0.0.1}:<host>:<container>, or write the wildcard "
        "out in full if that is what you mean."
    )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_documented_commands_publish_only_on_a_named_interface(path: Path) -> None:
    """A command in a document is run far more often than one in a compose file:
    it is the thing people copy."""
    commands = "\n".join(_FENCED.findall(path.read_text(encoding="utf-8")))

    unqualified = [
        argument
        for argument in _PUBLISH_FLAG.findall(commands)
        if _PORT_PAIR.search(argument) and not _names_an_interface(argument)
    ]

    assert not unqualified, (
        f"{path.relative_to(ROOT)} documents publishing on every interface: "
        f"{unqualified}. Bind to 127.0.0.1, or say plainly why the service "
        "should be reachable from the network and what authenticates it."
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"target": 5432, "published": 5433},
        {"target": 5432, "published": "5433", "protocol": "tcp"},
        {"target": 6333, "published": 6333, "mode": "host"},
    ],
)
def test_a_long_syntax_entry_without_a_host_ip_is_refused(entry: dict) -> None:
    """No compose file here uses the long syntax yet, so this pins the rule
    directly rather than through a file.

    It is the case the first version of this module got wrong: the entry was
    stringified, and a mapping's repr carries enough colons to satisfy a
    separator count, so the guard would have reported success on a service
    published to every interface.
    """
    assert not _names_an_interface(entry)


@pytest.mark.parametrize(
    "entry",
    [
        {"target": 5432, "published": 5433, "host_ip": "127.0.0.1"},
        {"target": 5432, "published": 5433, "host_ip": "0.0.0.0"},
    ],
)
def test_a_long_syntax_entry_naming_a_host_ip_is_allowed(entry: dict) -> None:
    """`0.0.0.0` written out passes here for the same reason it passes in the
    short form: an address someone typed is a decision a reviewer can see."""
    assert _names_an_interface(entry)
