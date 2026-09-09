"""
Tests for scripts/check_links.py.

The script exists so a rotted URL in the README cannot go unnoticed. Its own
failure mode is therefore the interesting one: reporting success over links it
never examined. Both cases below were live defects — a badge whose target was
invisible to the pattern, and a mistyped path that produced a green tick over
nothing.

Nothing here reaches the network; `check` is the only function that does, and it
is not exercised.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_links.py"


def _load() -> Any:
    """Import the script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("check_links", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_links = _load()


# --- what the pattern sees ---------------------------------------------------


def test_a_badge_target_is_found_not_just_its_image(tmp_path: pathlib.Path) -> None:
    """`[![alt](image)](target)` is how every badge in the README links out.

    A pattern anchored on `[text](url)` consumes `[![alt](image)` and matches
    only the image, so the target — the URL a reader actually lands on — was
    never checked, while the run still printed a count and said all links
    resolved.
    """
    doc = tmp_path / "README.md"
    doc.write_text(
        "[![CI](https://img.shields.io/ci.svg)](https://github.com/nlqueries/actions)\n",
        encoding="utf-8",
    )

    found = check_links.find_links([doc])

    assert "https://github.com/nlqueries/actions" in found
    assert "https://img.shields.io/ci.svg" in found


def test_an_ordinary_link_and_an_autolink_are_both_found(tmp_path: pathlib.Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text(
        "See [the site](https://nlqueries.com) and <https://nlqueries.com/docs/>.\n",
        encoding="utf-8",
    )

    found = check_links.find_links([doc])

    assert "https://nlqueries.com" in found
    assert "https://nlqueries.com/docs/" in found


def test_trailing_punctuation_inside_the_link_is_stripped(tmp_path: pathlib.Path) -> None:
    """The punctuation that matters is the punctuation *inside* the parentheses.

    A full stop after `[it](...)` is outside the capture and was never a
    problem -- the first version of this test asserted exactly that and so
    proved nothing, which the mutation run showed by surviving the strip being
    deleted. A dot *inside*, as in a hand-written `(.../page.)`, would be
    requested with the dot attached and 404.
    """
    doc = tmp_path / "d.md"
    doc.write_text("Read [it](https://nlqueries.com/docs/a.html.)\n", encoding="utf-8")

    found = check_links.find_links([doc])

    assert "https://nlqueries.com/docs/a.html" in found
    assert "https://nlqueries.com/docs/a.html." not in found


def test_a_full_stop_after_the_link_is_never_captured(tmp_path: pathlib.Path) -> None:
    """The ordinary sentence-ending case, excluded by the pattern on its own."""
    doc = tmp_path / "d.md"
    doc.write_text("Read [it](https://nlqueries.com/docs/a.html).\n", encoding="utf-8")

    assert "https://nlqueries.com/docs/a.html" in check_links.find_links([doc])


def test_a_relative_link_is_ignored(tmp_path: pathlib.Path) -> None:
    """They are checked by the reader's filesystem and obvious in review."""
    doc = tmp_path / "d.md"
    doc.write_text("See [docs](docs/getting-started.md).\n", encoding="utf-8")

    assert check_links.find_links([doc]) == {}


def test_a_url_is_reported_with_every_file_it_appears_in(tmp_path: pathlib.Path) -> None:
    """So a failure names all the places to fix, not the first one found."""
    (tmp_path / "a.md").write_text("[x](https://nlqueries.com)\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("[y](https://nlqueries.com)\n", encoding="utf-8")

    found = check_links.find_links(sorted(tmp_path.glob("*.md")))

    assert len(found["https://nlqueries.com"]) == 2


# --- refusing to pass over nothing -------------------------------------------


def test_a_path_that_does_not_exist_fails_rather_than_being_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this script exists to prevent, wearing the costume of a pass.

    A renamed directory or a run from the wrong working directory used to print
    "Checking 0 external links across 0 files. All links resolve." and exit 0.
    """
    monkeypatch.setattr(sys, "argv", ["check_links.py", "no-such-file.md"])

    assert check_links.main() == 1
    assert "No such path" in capsys.readouterr().err


def test_finding_no_links_at_all_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real file with nothing in it is the same green-tick-over-nothing."""
    doc = tmp_path / "empty.md"
    doc.write_text("no links here\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_links.py", str(doc)])

    assert check_links.main() == 1
    assert "Found no external links" in capsys.readouterr().err


def test_a_file_with_links_is_actually_checked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: the two refusals above must not reject a real run."""
    doc = tmp_path / "d.md"
    doc.write_text("[x](https://nlqueries.com)\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_links.py", str(doc)])
    monkeypatch.setattr(check_links, "check", lambda url: (url, 200, ""))

    assert check_links.main() == 0


def test_an_unreachable_link_fails_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "d.md"
    doc.write_text("[x](https://nlqueries.com/gone)\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_links.py", str(doc)])
    monkeypatch.setattr(check_links, "check", lambda url: (url, None, "HTTP 404"))

    assert check_links.main() == 1
    assert "did not resolve" in capsys.readouterr().err


def test_a_login_wall_counts_as_reachable() -> None:
    """401 and 403 mean a server answered about a page that exists."""
    assert 401 in check_links._REACHABLE
    assert 403 in check_links._REACHABLE
    assert 404 not in check_links._REACHABLE
