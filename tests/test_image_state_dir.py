"""The image keeps every piece of CLI state on the writable volume.

The compose service runs with a read-only root filesystem and mounts one
volume at ``/data/nlqueries``. The Dockerfile moved the knowledge base,
connectors, capsules and feedback there, but not ``NLQ_STATE_DIR``, so the
state directory stayed at ``~/.nlqueries`` on the read-only root, and
``nlqueries query`` inside the container died writing its session file
(``[Errno 30] Read-only file system``), as did ``embed-server start``.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_VOLUME = PurePosixPath("/data/nlqueries")


def _image_env() -> dict[str, str]:
    """Every ``KEY=value`` the Dockerfile's ``ENV`` instructions set."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    env: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("ENV "):
            env.update(re.findall(r"(\w+)=(\S+)", line[4:]))
    return env


@pytest.fixture(scope="module")
def service() -> dict:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["nlqueries-core"]


def test_the_state_dir_is_on_the_volume() -> None:
    state = _image_env().get("NLQ_STATE_DIR")

    assert state is not None, "the image leaves NLQ_STATE_DIR at ~/.nlqueries"
    assert PurePosixPath(state).is_relative_to(_VOLUME), state


@pytest.mark.parametrize(
    "name", ["NLQ_STATE_DIR", "KB_PATH", "CONNECTORS_FILE", "CAPSULES_DIR", "FEEDBACK_DIR"]
)
def test_each_state_path_is_on_the_volume(name: str) -> None:
    assert PurePosixPath(_image_env()[name]).is_relative_to(_VOLUME), name


def test_the_volume_is_where_compose_mounts_it_on_a_read_only_root(service: dict) -> None:
    """The premise of the two tests above: only the volume is writable."""
    assert service.get("read_only") is True
    mounts = [str(v).split(":", 1)[1] for v in service["volumes"] if ":" in str(v)]
    assert str(_VOLUME) in mounts
