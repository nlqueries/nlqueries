"""
A variable that picks a host port in the compose file does not move the server.

The compose file reads `.env` twice: compose substitutes it into the file, and
`env_file: .env` passes the same lines into the container. A variable used on
the host side of a `ports:` entry therefore also reaches the application, and
if the application reads it too, the two halves disagree. That happened with
MCP_PORT: setting it in `.env` to move the host port made the server listen on
the new port inside the container, behind a mapping to 8080 that nothing
answered, while the container still reported healthy.

The rule checked here is that every such variable is pinned in the service's
own `environment:`, which takes precedence over `env_file`. MCP_HOST reaches
the server the same way without naming a host port, so it is checked by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def services() -> dict:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]


def _host_side_variables(entry: str) -> set[str]:
    """Variables substituted into the host part of a short-syntax ports entry."""
    host = entry.rsplit(":", 1)[0]
    return set(re.findall(r"\$\{(\w+)", host))


def _container_port(entry: str) -> str:
    return entry.rsplit(":", 1)[1].split("/")[0]


def _read_by_the_app(name: str) -> bool:
    quoted = re.compile(rf"""["']{re.escape(name)}["']""")
    return any(
        quoted.search(path.read_text(encoding="utf-8"))
        for path in (ROOT / "nlqueries").rglob("*.py")
    )


def _pinned(service: dict, name: str) -> bool:
    value = (service.get("environment") or {}).get(name)
    return value is not None and "${" + name not in str(value)


def test_host_port_variables_the_app_reads_are_pinned_in_the_service(services: dict) -> None:
    checked: list[str] = []
    unpinned: list[str] = []
    for service_name, service in services.items():
        if not service.get("env_file"):
            continue
        for entry in service.get("ports") or []:
            for name in sorted(_host_side_variables(str(entry))):
                if not _read_by_the_app(name):
                    continue
                checked.append(name)
                if not _pinned(service, name):
                    unpinned.append(f"{service_name}: {name} (from {entry!r})")
    # The rule is not vacuous: MCP_PORT is read by the server and picks a host port.
    assert "MCP_PORT" in checked
    assert not unpinned, (
        "set in .env, these would move the server inside the container as well as "
        f"the host port: {unpinned}"
    )


def test_the_server_listens_on_the_port_the_mapping_targets(services: dict) -> None:
    service = services["nlqueries-core"]
    (entry,) = [str(e) for e in service["ports"] if "MCP_PORT" in str(e)]
    assert str(service["environment"]["MCP_PORT"]) == _container_port(entry)


def test_the_server_binds_every_interface_in_the_container(services: dict) -> None:
    # On the container's own loopback the published port is refused, while the
    # healthcheck, which connects from inside, still passes.
    assert services["nlqueries-core"]["environment"].get("MCP_HOST") == "0.0.0.0"
