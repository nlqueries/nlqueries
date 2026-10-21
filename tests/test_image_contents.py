"""What the image carries, and which image the compose file runs.

The Dockerfile adds nothing over the base image: the healthcheck is Python,
which the base already has, so an HTTP client such as `curl` would be one more
package to patch for nothing. The compose file and the docs name a released
version of the image rather than `:latest`, so `docker compose up` starts the
release they were written for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "nlqueries/core"


def _instructions() -> list[str]:
    """The Dockerfile's instructions, continuation lines joined, comments dropped."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return [i.strip() for i in "\n".join(lines).replace("\\\n", " ").splitlines() if i.strip()]


def _compose_image() -> str:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    image: str = compose["services"]["nlqueries-core"]["image"]
    return image


def test_no_instruction_installs_or_runs_curl() -> None:
    assert not [i for i in _instructions() if re.search(r"\bcurl\b", i)]


def test_no_system_packages_are_installed_over_the_base() -> None:
    assert not [i for i in _instructions() if re.search(r"\b(apt-get|apt|apk)\b", i)]


def test_the_healthcheck_is_python() -> None:
    (check,) = [i for i in _instructions() if i.startswith("HEALTHCHECK ")]
    assert " CMD python " in check


def test_the_compose_service_does_not_use_curl() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "curl" not in yaml.safe_dump(compose["services"]["nlqueries-core"])


def test_compose_pins_a_final_release_of_the_image() -> None:
    """Not `:latest`, not a bare name, and not a pre-release."""
    name, _, tag = _compose_image().partition(":")
    assert name == IMAGE
    assert re.fullmatch(r"\d+\.\d+\.\d+", tag), tag
    assert not Version(tag).is_prerelease


@pytest.mark.parametrize("doc", ["README.md", "docs/getting-started.md"])
def test_the_docs_name_the_image_compose_runs(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    named = set(re.findall(rf"`({re.escape(IMAGE)}:[^`]+)`", text))
    assert named == {_compose_image()}
