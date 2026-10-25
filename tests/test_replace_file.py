"""A rename Windows refuses for a moment is retried, not reported.

On Windows, `os.replace` onto a file another process holds open -- an antivirus
scanner or the indexer reading a file just written -- raises PermissionError
until it lets go. `write_private` renames every state file into place, so a
command that writes a file twice in quick succession failed now and then with
"Access is denied". These tests stand in for the refusal; they hold on any
platform by setting the switch the module reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from nlqueries import state_files


@pytest.fixture
def refusals(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """`os.replace` refuses the first `n` attempts; waits are recorded, not slept."""
    state: dict[str, object] = {"n": 0, "calls": 0, "waits": []}
    real_replace = os.replace

    def replace(src: object, dst: object) -> None:
        state["calls"] = int(state["calls"]) + 1  # type: ignore[call-overload]
        if int(state["calls"]) <= int(state["n"]):  # type: ignore[call-overload]
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(state_files.os, "replace", replace)
    monkeypatch.setattr(state_files.time, "sleep", lambda s: state["waits"].append(s))  # type: ignore[attr-defined]
    monkeypatch.setattr(state_files, "_RETRY_REFUSED_REPLACE", True)
    return state


def test_a_brief_refusal_is_retried_and_the_file_is_written(
    tmp_path: Path, refusals: dict[str, object]
) -> None:
    refusals["n"] = 2
    path = tmp_path / "connectors.yaml"
    path.write_text("old", encoding="utf-8")

    state_files.write_private(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert refusals["calls"] == 3
    assert refusals["waits"] == [0.05, 0.1]
    assert [p.name for p in tmp_path.iterdir()] == ["connectors.yaml"]


def test_a_refusal_that_lasts_is_reported_after_the_last_attempt(
    tmp_path: Path, refusals: dict[str, object]
) -> None:
    refusals["n"] = 99
    path = tmp_path / "connectors.yaml"
    path.write_text("old", encoding="utf-8")

    with pytest.raises(PermissionError):
        state_files.write_private(path, "new")

    assert refusals["calls"] == state_files._REPLACE_ATTEMPTS
    assert path.read_text(encoding="utf-8") == "old"
    # The temporary file is cleaned up as on any other failure.
    assert [p.name for p in tmp_path.iterdir()] == ["connectors.yaml"]


def test_off_windows_a_refusal_is_reported_at_once(
    tmp_path: Path, refusals: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Elsewhere a PermissionError means the permissions, which waiting does not change."""
    monkeypatch.setattr(state_files, "_RETRY_REFUSED_REPLACE", False)
    refusals["n"] = 1

    with pytest.raises(PermissionError):
        state_files.write_private(tmp_path / "f.yaml", "x")

    assert refusals["calls"] == 1
    assert refusals["waits"] == []


def test_other_errors_are_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def replace(src: object, dst: object) -> None:
        calls.append(dst)
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(state_files.os, "replace", replace)
    monkeypatch.setattr(state_files, "_RETRY_REFUSED_REPLACE", True)

    with pytest.raises(FileNotFoundError):
        state_files.replace_file(tmp_path / "a", tmp_path / "b")

    assert len(calls) == 1
