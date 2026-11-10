#!/usr/bin/env python3
# nlqueries-core — OSS (BSL 1.1)
"""Fail if any package the lock pins is installed at another version.

CI's Python 3.11 job installs ``requirements/core.lock`` with hashes and then
the package in editable mode, which adds what the lock does not hold (the
``docs`` extra). pip leaves a requirement that is already satisfied alone, so
the locked versions should survive; this checks that they did, so the job
cannot quietly test something other than what the image ships.

Usage: python scripts/check_installed_matches_lock.py [requirements/core.lock]
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;\\]+)")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_pins(lock: Path) -> dict[str, str]:
    """``{normalised name: version}`` for every ``name==version`` line in *lock*."""
    pins: dict[str, str] = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line.strip())
        if match:
            pins[_normalise(match.group(1))] = match.group(2)
    return pins


def mismatches(pins: dict[str, str]) -> list[str]:
    """One line per locked package installed at another version, or not at all."""
    installed = {
        _normalise(dist.metadata["Name"]): dist.version for dist in metadata.distributions()
    }
    found = []
    for name, version in sorted(pins.items()):
        have = installed.get(name)
        if have != version:
            found.append(f"{name}: locked {version}, installed {have or 'nothing'}")
    return found


def main(argv: list[str]) -> int:
    lock = Path(argv[1] if len(argv) > 1 else "requirements/core.lock")
    pins = locked_pins(lock)
    if not pins:
        print(f"{lock}: no pinned packages found", file=sys.stderr)
        return 1
    found = mismatches(pins)
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(f"{len(found)} of {len(pins)} locked packages differ from {lock}", file=sys.stderr)
        return 1
    print(f"all {len(pins)} locked packages installed at their locked versions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
