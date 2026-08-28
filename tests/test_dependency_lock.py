"""
Whether the committed lock still describes what the project declares (SEC-11).

The image installs from `requirements/core.lock` with `--require-hashes`, so a
build resolves nothing: it gets the exact bytes the lock names. That only holds
while the lock is current, and nothing else in the pull-request pipeline would
notice that it is not -- only the release workflow builds the image, so a
dependency added to `pyproject.toml` without regenerating the lock would install
fine for every developer and fail at the release.

These are structural checks, deliberately. Whether the *versions* are right is
what the hashes and a review of the regenerated lock are for; what is worth
catching automatically is a declared dependency the lock has never heard of.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = REPO_ROOT / "requirements" / "core.lock"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: `Name_Here[extra]>=1.0` -> `name-here`. Comparison is on the normalised
#: distribution name: a lock writes `qdrant-client` where pyproject may say
#: `qdrant_client`, and pip-tools keeps the extras in the name it emits
#: (`httpx[http2]==0.28.1`), so the bracket is dropped from both sides.
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(section: list[str]) -> set[str]:
    names = set()
    for requirement in section:
        match = _REQUIREMENT.match(requirement)
        if match:
            names.add(_normalise(match.group(1)))
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def locked() -> set[str]:
    names = set()
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "#", "-")) or "==" not in line:
            continue
        name = line.split("==")[0].strip()
        names.add(_normalise(name.split("[")[0]))
    return names


def test_the_lock_exists_and_is_hashed() -> None:
    """Without hashes the lock pins versions but not bytes, and a compromised
    re-upload of the same version would install."""
    text = LOCK.read_text(encoding="utf-8")

    assert "--hash=sha256:" in text


def test_every_runtime_dependency_is_locked(pyproject: dict, locked: set[str]) -> None:
    missing = _declared(pyproject["project"]["dependencies"]) - locked

    assert not missing, f"declared but not in the lock — run scripts/lock-deps.sh: {missing}"


def test_every_dev_dependency_is_locked(pyproject: dict, locked: set[str]) -> None:
    """The lock covers the dev extra too, so CI and the image agree."""
    dev = pyproject["project"].get("optional-dependencies", {}).get("dev", [])

    assert not _declared(dev) - locked


def test_setuptools_is_pinned(locked: set[str]) -> None:
    """pip-tools leaves setuptools floating unless asked. A lock installed with
    --require-hashes must pin everything, and the failure is a build that either
    silently uses the base image's copy or refuses outright."""
    assert "setuptools" in locked


def test_the_lock_carries_no_unresolved_warnings() -> None:
    """pip-compile writes a WARNING block into the file when it could not pin
    something. It is a comment, so nothing else would ever fail on it."""
    assert "# WARNING" not in LOCK.read_text(encoding="utf-8")


def test_the_dockerfile_installs_from_the_lock() -> None:
    """A lock nothing installs from is decoration."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--require-hashes -r requirements/core.lock" in dockerfile


def test_the_base_image_is_pinned_by_digest() -> None:
    """SEC-11's other half: a mutable base tag means the same Dockerfile builds
    different images."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    froms = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]

    assert froms
    for line in froms:
        assert "@sha256:" in line, line
