"""
The compose file, and the docs that quote it, run the release they ship with.

`docker-compose.yml` pins `nlqueries/core` by version rather than `:latest`, so
a later `docker compose pull` does not move anyone to another release. The
cost is that every release has to move the pin, and v0.4.0 shipped without it:
its compose file still pinned 0.3.0, so the Docker quickstart from the 0.4.0
tag ran 0.3.0, and README and getting-started told the reader so.

The rule checked here is that the pin, and every image tag README and
getting-started quote, equal the version in pyproject.toml. The release
workflow's `versions` job holds a final tag to that same version, so a tag
cannot publish a compose file that names another release's image.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

IMAGE = "nlqueries/core"
# A concrete version only: `nlqueries/core:<version>` in a command template is
# a placeholder, not a pin.
QUOTED_TAG = re.compile(re.escape(IMAGE) + r":(\d+\.\d+\.\d+[^\s`'\")]*)")


@pytest.fixture(scope="module")
def version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        declared = str(tomllib.load(fh)["project"]["version"])
    # A pre-release does not move the pin: pyproject says `1.0.0rc1` so PyPI
    # gets that version, the image is pushed as `1.0.0-rc1`, and compose stays
    # on the last final release. The release workflow's `versions` job skips
    # pre-release tags for the same reason.
    if not re.fullmatch(r"\d+\.\d+\.\d+", declared):
        pytest.skip(f"{declared} is a pre-release; the compose pin stays on the last final release")
    return declared


def test_the_compose_file_pins_this_version(version: str) -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    pins = [
        service["image"]
        for service in compose["services"].values()
        if str(service.get("image", "")).startswith(IMAGE + ":")
    ]

    assert pins == [f"{IMAGE}:{version}"]


@pytest.mark.parametrize("doc", ["README.md", "docs/getting-started.md"])
def test_the_quickstart_names_this_version(doc: str, version: str) -> None:
    quoted = QUOTED_TAG.findall((ROOT / doc).read_text(encoding="utf-8"))

    # Some tag has to be found, or a rewording that drops it would leave this
    # test passing over nothing.
    assert quoted, f"{doc} no longer quotes the {IMAGE} tag the quickstart pulls"
    assert set(quoted) == {version}, f"{doc} quotes {sorted(set(quoted))}, pyproject says {version}"
