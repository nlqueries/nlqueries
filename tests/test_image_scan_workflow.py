"""What has to stay true of the image scan workflow.

`image-scan.yml` builds the image and scans it with Trivy, installed from its
release archive and checked against a SHA-256 kept in the workflow. These pin
the properties that make it trustworthy and the ones that keep it from getting
in the way, and run the install step's own script to show that an archive
whose checksum does not match is refused.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "image-scan.yml"


def _workflow() -> dict[Any, Any]:
    loaded: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return loaded


def _steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _workflow()["jobs"]["scan"]["steps"]
    return steps


def _step(name_starts: str) -> dict[str, Any]:
    (step,) = [s for s in _steps() if str(s.get("name", "")).startswith(name_starts)]
    return step


def test_it_runs_weekly_on_tags_on_image_changes_and_by_hand() -> None:
    # YAML 1.1, which PyYAML follows, reads the bare key `on` as True.
    triggers = _workflow()[True]
    assert set(triggers) == {"schedule", "push", "pull_request", "workflow_dispatch"}
    (cron,) = [entry["cron"] for entry in triggers["schedule"]]
    assert cron.split()[4] == "1"  # Monday
    assert triggers["push"] == {"tags": ["v*"]}
    assert set(triggers["pull_request"]) == {"paths"}
    assert sorted(triggers["pull_request"]["paths"]) == [
        ".github/workflows/image-scan.yml",
        "Dockerfile",
        "requirements/**",
    ]


def test_it_can_write_only_security_events() -> None:
    assert _workflow()["permissions"] == {"contents": "read"}
    assert _workflow()["jobs"]["scan"]["permissions"] == {
        "contents": "read",
        "security-events": "write",
    }


def test_every_action_is_pinned_by_commit() -> None:
    uses = [s["uses"] for s in _steps() if "uses" in s]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref


def test_nothing_is_piped_into_a_shell() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert not re.search(r"\|\s*(sudo\s+)?(ba)?sh\b", text)


def test_trivy_is_pinned_to_a_version_and_a_checksum() -> None:
    env = _workflow()["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["TRIVY_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["TRIVY_SHA256"])
    install = _step("Install Trivy")["run"]
    assert "sha256sum --check --strict" in install
    # The checksum is the one in this file, not one downloaded beside the archive.
    assert "checksums" not in install


def test_the_scan_reports_high_and_critical_without_failing_the_job() -> None:
    scan = _step("Scan")["run"]
    assert "--severity HIGH,CRITICAL" in scan
    assert "--ignore-unfixed" in scan
    assert "--exit-code 0" in scan
    assert "--format sarif" in scan
    (output,) = re.findall(r"--output (\S+)", scan)
    upload = _step("Upload")
    assert upload["uses"].startswith("github/codeql-action/upload-sarif@")
    assert upload["with"] == {"sarif_file": output, "category": "trivy-image"}


def test_a_fork_pull_request_scans_but_does_not_upload() -> None:
    """A fork's token is read-only, so the upload would fail with a 403."""
    upload_if = _step("Upload")["if"]
    assert "github.event_name != 'pull_request'" in upload_if
    assert "github.event.pull_request.head.repo.full_name == github.repository" in upload_if
    assert "if" not in _step("Scan")


def test_a_fork_pull_request_gets_the_findings_in_its_log() -> None:
    """The SARIF holds them only in its file, which is not uploaded there, so
    the same scan prints a table on exactly the runs the upload skips."""
    show = _step("Show the findings")
    assert show["if"] == (
        "github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name != github.repository"
    )
    scan = _step("Scan")["run"]
    for flag in ("--severity HIGH,CRITICAL", "--ignore-unfixed", "--exit-code 0"):
        assert flag in scan and flag in show["run"]
    assert "--format table" in show["run"]


def test_the_image_is_built_locally_and_not_pushed() -> None:
    build = _step("Build the image")["run"]
    assert build.startswith("docker build ")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "docker push" not in text
    assert "docker/login-action" not in text


_NEEDS = ("bash", "sha256sum", "tar")


def _run_install(
    tmp_path: Path, archive: bytes, expected_sha256: str
) -> subprocess.CompletedProcess[str]:
    """Run the workflow's install step with `curl` serving *archive* locally."""
    served = tmp_path / "served.tar.gz"
    served.write_bytes(archive)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    # A `curl` that writes the served archive to its --output argument.
    stub = (
        'curl() { local out=""; while [ $# -gt 0 ]; do '
        'if [ "$1" = "--output" ]; then out="$2"; shift; fi; shift; done; '
        f'cp "{served.as_posix()}" "$out"; }}\n'
    )
    script = stub + _step("Install Trivy")["run"]
    env = {
        **os.environ,
        # Relative, from *tmp_path*: GNU tar reads a "C:" prefix as a remote
        # host, which only matters to this test on Windows.
        "RUNNER_TEMP": runner_temp.name,
        "TRIVY_VERSION": _workflow()["env"]["TRIVY_VERSION"],
        "TRIVY_SHA256": expected_sha256,
    }
    # The bash `shutil.which` found, by full path: on Windows a bare "bash"
    # resolves through System32 first and can start WSL's, which cannot see
    # these paths.
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(
        [bash, "-e", "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _fake_trivy_archive() -> bytes:
    """A tar.gz holding a `trivy` that prints a version, as the real one does."""
    body = b"#!/bin/sh\necho 'Version: test'\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("trivy")
        info.size = len(body)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


@pytest.mark.skipif(not all(shutil.which(t) for t in _NEEDS), reason="needs bash, sha256sum, tar")
def test_an_archive_with_the_wrong_checksum_is_refused(tmp_path: Path) -> None:
    archive = _fake_trivy_archive()
    wrong = hashlib.sha256(archive + b"x").hexdigest()

    result = _run_install(tmp_path, archive, wrong)

    assert result.returncode != 0
    # Refused by the checksum, not by some earlier step failing.
    assert "FAILED" in result.stdout + result.stderr
    version = _workflow()["env"]["TRIVY_VERSION"]
    assert (tmp_path / "runner" / f"trivy_{version}_Linux-64bit.tar.gz").exists()
    assert not (tmp_path / "runner" / "trivy" / "trivy").exists()


@pytest.mark.skipif(not all(shutil.which(t) for t in _NEEDS), reason="needs bash, sha256sum, tar")
def test_an_archive_with_the_right_checksum_is_installed(tmp_path: Path) -> None:
    archive = _fake_trivy_archive()

    result = _run_install(tmp_path, archive, hashlib.sha256(archive).hexdigest())

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runner" / "trivy" / "trivy").exists()
