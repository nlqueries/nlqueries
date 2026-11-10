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


# ---------------------------------------------------------------------------
# CI tests what the image ships (CI-11)
# ---------------------------------------------------------------------------


def _locked_version(name: str) -> tuple[int, ...]:
    match = re.search(rf"^{re.escape(name)}==([0-9.]+)", LOCK.read_text("utf-8"), re.M)
    assert match, f"{name} is not pinned in the lock"
    return tuple(int(part) for part in match.group(1).split("."))


def test_the_sqlglot_floor_is_the_measured_one_and_the_lock_satisfies_it(pyproject: dict) -> None:
    """30.13 is the lowest release the SQL policy's tests pass on; the floor was
    23.0, which the policy has not been tested against for many releases."""
    floor = next(d for d in pyproject["project"]["dependencies"] if d.startswith("sqlglot"))

    assert floor == "sqlglot>=30.13"
    assert _locked_version("sqlglot") >= (30, 13)


def test_ci_tests_the_lock_on_the_image_s_python() -> None:
    """Unpinned, every test job resolved a newer sqlglot than the image carried."""
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    step = ci[ci.index("- name: Install from the lock (Python 3.11)") :]
    step = step[: step.index("- name: Install package + dev extras")]

    assert "if: matrix.python-version == '3.11'" in step
    assert "pip install --require-hashes -r requirements/core.lock" in step
    assert "scripts/check_installed_matches_lock.py" in step
    assert "python:3.11" in (REPO_ROOT / "Dockerfile").read_text("utf-8")
    assert "with Python 3.11" in LOCK.read_text("utf-8")


def test_the_guard_reports_a_package_installed_at_another_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_installed_matches_lock", REPO_ROOT / "scripts" / "check_installed_matches_lock.py"
    )
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    lock = tmp_path / "x.lock"
    lock.write_text(
        "Sqlglot==30.17.0 \\\n    --hash=sha256:abc\n    # via nlqueries-core\n"
        "pyyaml==6.0.2 \\\n    --hash=sha256:def\n",
        encoding="utf-8",
    )
    assert guard.locked_pins(lock) == {"sqlglot": "30.17.0", "pyyaml": "6.0.2"}

    class _Dist:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        guard.metadata,
        "distributions",
        lambda: [_Dist("sqlglot", "30.29.0"), _Dist("PyYAML", "6.0.2")],
    )
    assert guard.mismatches(guard.locked_pins(lock)) == [
        "sqlglot: locked 30.17.0, installed 30.29.0"
    ]
    assert guard.main(["x", str(lock)]) == 1
