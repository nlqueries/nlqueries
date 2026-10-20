"""What has to stay true of the release workflow.

`release.yml` publishes to PyPI and Docker Hub on a version tag, and can also be
run by hand as a dry run. On a manual run, only the
`if: github.event_name == 'push'` on each publishing job stands between a run on
any branch and a real publish, so these checks fail when an edit drops one, or
adds a publishing job without one.
"""

from __future__ import annotations

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
    """A manual run is a dry run; it must never reach an environment or a secret."""
    assert PUSH_GUARD in str(_jobs()[name].get("if", ""))


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
