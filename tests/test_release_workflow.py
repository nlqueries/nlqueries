"""What has to stay true of the release workflow.

`release.yml` publishes to PyPI and Docker Hub on a version tag, and can also be
run by hand as a dry run. On a manual run, only the
`if: github.event_name == 'push'` on each publishing job stands between a run on
any branch and a real publish, so these checks fail when an edit drops one, or
adds a publishing job without one. The rest pin down what a release
publishes: which image tags a version tag pushes, and the SBOMs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / ".github" / "workflows" / "release.yml"

PUSH_GUARD = "github.event_name == 'push'"


def _jobs() -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]
    return jobs


def _publishes(job: dict[str, Any]) -> bool:
    """Whether *job* can publish: it takes an environment, a secret, or any
    write permission (an OIDC token for PyPI, `contents: write` for a release)."""
    permissions = job.get("permissions")
    return (
        "environment" in job
        or "secrets." in yaml.safe_dump(job)
        or (isinstance(permissions, dict) and "write" in permissions.values())
    )


def test_the_release_runs_the_whole_ci_workflow() -> None:
    """Not a copy of some of its jobs: a release waits on all of CI."""
    assert _jobs()["ci"].get("uses") == "./.github/workflows/ci.yml"


def test_the_publishing_jobs_are_the_ones_expected() -> None:
    """A guard on a list that silently missed a job would prove nothing."""
    assert {name for name, job in _jobs().items() if _publishes(job)} >= {
        "publish-pypi",
        "publish-docker",
        "github-release",
        "trigger-reindex",
    }


@pytest.mark.parametrize("name", sorted(n for n, j in _jobs().items() if _publishes(j)))
def test_every_publishing_job_runs_only_on_a_tag_push(name: str) -> None:
    """A manual run is a dry run; it must never reach an environment or a secret.

    Compared exactly, not as a substring: `... == 'push' || inputs.publish`
    contains the guard and would publish from a manual run.
    """
    assert str(_jobs()[name].get("if", "")).strip() == PUSH_GUARD


def test_a_push_runs_the_release_only_for_a_version_tag() -> None:
    """The push guard means "a release" only because the push trigger is tags
    only. A `branches` filter beside it would make an ordinary push to `main`
    satisfy every guard and publish."""
    workflow = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    # YAML 1.1, which PyYAML follows, reads the bare key `on` as True.
    triggers = workflow[True]
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert set(triggers["push"]) == {"tags"}
    assert all(str(tag).startswith("v") for tag in triggers["push"]["tags"])


def test_nothing_widens_the_workflow_wide_permissions() -> None:
    """An OIDC token is granted only to the job that uploads to PyPI."""
    workflow = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    with_token = [
        name
        for name, job in _jobs().items()
        if isinstance(job.get("permissions"), dict) and job["permissions"].get("id-token")
    ]
    assert with_token == ["publish-pypi"]


def test_the_dry_run_job_has_no_environment_or_secret() -> None:
    dry_run = _jobs()["release-dry-run"]
    assert "environment" not in dry_run
    assert "secrets." not in yaml.safe_dump(dry_run)
    assert "workflow_dispatch" in str(dry_run.get("if", ""))


def _steps(job: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _jobs()[job]["steps"]
    return steps


def _step(job: str, name: str) -> dict[str, Any]:
    return next(step for step in _steps(job) if step.get("name") == name)


def _image_builds() -> dict[str, dict[str, Any]]:
    """The docker/build-push-action step of each job that builds the image."""
    return {
        name: step
        for name in _jobs()
        for step in _jobs()[name].get("steps", [])
        if str(step.get("uses", "")).startswith("docker/build-push-action@")
    }


def _image_tags_for(ref: str, tmp_path: Path) -> set[str]:
    """Run the `Choose image tags` step as a tag push of *ref* would."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    step = _step("publish-docker", "Choose image tags")
    assert step.get("shell") == "bash"
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    subprocess.run(
        [bash, "-e", "-c", step["run"]],
        env={"GITHUB_REF": ref, "GITHUB_OUTPUT": str(output), "PATH": os.environ["PATH"]},
        check=True,
    )
    lines = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    return set(lines["tags"].split(","))


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("refs/tags/v1.2.3", {"nlqueries/core:1.2.3", "nlqueries/core:latest"}),
        ("refs/tags/v10.20.30", {"nlqueries/core:10.20.30", "nlqueries/core:latest"}),
        ("refs/tags/v1.0.0-rc1", {"nlqueries/core:1.0.0-rc1"}),
        ("refs/tags/v1.0.0rc1", {"nlqueries/core:1.0.0rc1"}),
        ("refs/tags/v1.0.0-beta.2", {"nlqueries/core:1.0.0-beta.2"}),
        ("refs/tags/v1.0.0.dev1", {"nlqueries/core:1.0.0.dev1"}),
    ],
)
def test_only_a_final_version_moves_latest(ref: str, expected: set[str], tmp_path: Path) -> None:
    """A pre-release pushes its own tag and never `latest`, which is what a
    plain `docker pull nlqueries/core` resolves to."""
    assert _image_tags_for(ref, tmp_path) == expected


def _compare_versions(ref: str, version: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the `versions` job's comparison as a push of *ref* would, in a
    checkout whose pyproject.toml declares *version*."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    step = _step("versions", "Compare the tag with pyproject.toml")
    assert step.get("shell") == "bash"
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "nlqueries-core"\nversion = "{version}"\n', encoding="utf-8"
    )
    return subprocess.run(
        [bash, "-c", step["run"]],
        cwd=tmp_path,
        # The whole environment, not just PATH: the step runs `python3`, which
        # on some machines is a launcher that needs more of it to start.
        env={**os.environ, "GITHUB_REF": ref},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.mark.parametrize(
    ("ref", "version", "passes"),
    [
        ("refs/tags/v1.2.3", "1.2.3", True),
        # v0.4.0 shipped a compose file pinned to 0.3.0; a tag that disagrees
        # with the package version it publishes is refused before anything is.
        ("refs/tags/v1.2.4", "1.2.3", False),
        ("refs/tags/v1.2.3", "1.2.4", False),
        # A pre-release is spelled differently from its PEP 440 version and
        # does not move the compose pin, so it is not compared.
        ("refs/tags/v1.0.0-rc1", "1.0.0rc1", True),
        # A dry run has no tag.
        ("refs/heads/main", "1.2.3", True),
    ],
)
def test_a_final_tag_must_name_the_version_it_publishes(
    ref: str, version: str, passes: bool, tmp_path: Path
) -> None:
    result = _compare_versions(ref, version, tmp_path)

    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    if not passes:
        assert "does not match pyproject.toml version" in result.stdout


@pytest.mark.parametrize("job", ["build", "publish-docker"])
def test_nothing_is_built_to_publish_before_the_tag_is_checked(job: str) -> None:
    """Dropping `versions` from either would let a mismatched tag publish again."""
    assert "versions" in _jobs()[job]["needs"]


def test_the_published_image_takes_its_tags_from_that_step() -> None:
    """The filter above proves nothing if the push names `latest` itself."""
    push = _image_builds()["publish-docker"]
    assert push["with"]["push"] is True
    assert push["with"]["tags"] == "${{ steps.tags.outputs.tags }}"


@pytest.mark.parametrize("job", ["publish-docker", "release-dry-run"])
def test_every_image_build_attaches_an_sbom(job: str) -> None:
    """The dry run builds with the same setting, so it exercises it first."""
    assert set(_image_builds()) == {"publish-docker", "release-dry-run"}
    sbom = _image_builds()[job]["with"].get("sbom")
    assert sbom == _image_builds()["publish-docker"]["with"]["sbom"]
    assert str(sbom).startswith("generator=docker/buildkit-syft-scanner:")


def test_the_sbom_generator_is_pinned_by_digest() -> None:
    """`sbom: true` runs BuildKit's default scanner, resolved by a mutable tag
    at build time: the one build input that would not be pinned while the base
    image, the actions and the lock are. Every `sbom:` in the workflow names
    its generator by digest."""
    lines = [
        line.strip()
        for line in RELEASE.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("sbom:")
    ]
    assert lines
    for line in lines:
        assert line.startswith("sbom: generator="), line
        assert "@sha256:" in line, line


def test_the_wheel_sbom_is_built_outside_dist() -> None:
    """`publish-pypi` uploads every file in dist/, and PyPI refuses anything
    that is not a distribution, so the SBOM has its own directory and artifact."""
    script = _step("wheel-sbom", "Generate the wheel's SBOM (CycloneDX)")["run"]
    assert "cyclonedx-py environment" in script
    assert '-o "sbom/' in script
    upload = _step("wheel-sbom", "Store the wheel's SBOM")
    assert upload["with"] == {"name": "wheel-sbom", "path": "sbom/", "retention-days": 7}


def test_the_wheel_sbom_has_a_job_of_its_own_that_publishes_nothing() -> None:
    """Generating it resolves the wheel's whole closure from PyPI, the step
    most likely to fail, so it runs apart from `build` and with nothing to
    publish with: no environment, no secret, no token, read-only contents."""
    job = _jobs()["wheel-sbom"]
    assert job["needs"] == ["build"]
    assert job["permissions"] == {"contents": "read"}
    assert not _publishes(job)
    assert "if" not in job, "a dry run needs the SBOM too"
    assert not [s for s in _steps("build") if "cyclonedx" in yaml.safe_dump(s)]


def test_publishing_to_pypi_does_not_wait_on_the_sbom() -> None:
    """A failed SBOM must not stop the packages reaching PyPI after the image
    is already pushed; `publish-docker` needs only `ci` and `versions`."""
    assert _jobs()["publish-pypi"]["needs"] == ["build"]


@pytest.mark.parametrize("job", ["github-release", "release-dry-run"])
def test_the_jobs_that_take_the_sbom_wait_for_it(job: str) -> None:
    """Only the GitHub release, which can be re-run, is held back by it."""
    assert "wheel-sbom" in _jobs()[job]["needs"]


def test_the_sbom_environment_holds_only_the_wheel_and_its_dependencies() -> None:
    """A virtualenv seeded with pip would list pip and setuptools as
    components of the release. The runner's pip installs into it instead."""
    script = _step("wheel-sbom", "Generate the wheel's SBOM (CycloneDX)")["run"]
    assert 'python -m venv --without-pip "$RUNNER_TEMP/wheel-env"' in script
    assert 'pip --python "$RUNNER_TEMP/wheel-env/bin/python" install' in script
    assert "wheel-env/bin/pip" not in script


def test_the_dry_run_exports_the_image_so_its_sbom_is_built() -> None:
    """With no exporter BuildKit has nothing to attach an attestation to, so
    the dry run exports the image as an OCI tarball, to /dev/null: both
    platforms' layers would otherwise land on the runner's disk."""
    build = _image_builds()["release-dry-run"]["with"]
    assert build["push"] is False
    assert build["outputs"] == "type=oci,dest=/dev/null"


def test_the_github_release_attaches_the_wheel_sbom() -> None:
    downloads = [
        step["with"]
        for step in _steps("github-release")
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert {"name": "wheel-sbom", "path": "sbom/"} in downloads
    assert "dist/* sbom/*" in _step("github-release", "Create GitHub Release")["run"]
