#!/usr/bin/env bash
# Run the core suite in a Linux container.
#
# Smart App Control is enforced on the Windows dev machine and blocks
# `grpc/_cython/cygrpc*.pyd` — an unsigned native extension pulled in
# transitively by `qdrant-client`. Every test whose import chain reaches gRPC
# fails there with `ImportError: DLL load failed`, and two modules fail to
# collect outright, which aborts the whole run. Smart App Control has no
# per-file exception list, and turning it off cannot be undone without
# reinstalling Windows. A Linux container has no Windows Code Integrity, so the
# same wheels simply load.
#
#   scripts/test-in-container.sh                    # whole suite
#   scripts/test-in-container.sh tests/security     # a subset
#   scripts/test-in-container.sh -k cache_poisoning # one pattern
#
# Arguments are passed straight to pytest.
#
# The testcontainers reaper is left ON, so an abnormal exit cleans up after
# itself. To check anyway, or after a crash of the reaper too:
#
#   docker ps --filter label=org.testcontainers=true
#
# That filter rather than an image name, because the fixtures start
# `postgres:16-alpine` (the Postgres connector and security suites) and
# `qdrant/qdrant:v1.18.2` (the two cache integration modules), and a filter
# keeps finding them if that list changes.
#
# For the record, since a previous revision of this header said otherwise: a
# single Ctrl-C strands nothing even with the reaper off. Docker proxies
# SIGINT to pytest as PID 1, pytest catches `KeyboardInterrupt` and still runs
# `pytest_sessionfinish`, so the fixtures' `finally` blocks stop their own
# containers. What the reaper is for is the case where no teardown runs at
# all -- a `docker kill`, an OOM kill, or a second Ctrl-C cutting into a
# teardown already in progress. Each of those was measured here.

set -euo pipefail

IMAGE="nlqueries-core-test"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker on Windows needs a Windows path. Git Bash reports a POSIX one
# (/c/code/...), which the daemon rejects with "path not found" -- and that
# error names the path, not the cause, so it reads like a missing directory.
# `cygpath` exists only under Git Bash/MSYS; elsewhere $HERE is already right.
if command -v cygpath >/dev/null 2>&1; then
  MOUNT="$(cygpath -w "$HERE")"
else
  MOUNT="$HERE"
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop and try again." >&2
  exit 1
fi

# Only rebuilds when pyproject.toml changes — the dependency layer is keyed on
# the manifest, not on source.
MSYS_NO_PATHCONV=1 docker build --quiet -f "$MOUNT/Dockerfile.test" -t "$IMAGE" "$MOUNT" >/dev/null

# Reaper preflight. See the header: without this image the container fixtures
# raise, the suite turns that into `pytest.skip`, and the run is green with the
# security corpus absent — a worse outcome than failing.
#
# The tag is read out of the image rather than written here. `Dockerfile.test`
# installs the dev extras from `pyproject.toml`, where the constraint is
# `testcontainers[postgres]>=4.0` — NOT the pinned `requirements/core.lock` — so
# a rebuild can carry a newer testcontainers whose `ryuk_image` default has
# moved, and a tag hardcoded in this script would send you after the wrong one.
RYUK_PROBE='from testcontainers.core.config import testcontainers_config as c; print(c.ryuk_image)'
RYUK="$(MSYS_NO_PATHCONV=1 docker run --rm "$IMAGE" python -c "$RYUK_PROBE" 2>/dev/null || true)"
if [ -z "$RYUK" ]; then
  # Could not ask the image. Say so rather than pretending the check ran: the
  # whole point is that a missing reaper is invisible in the results.
  echo "Warning: could not read the reaper image from $IMAGE; preflight skipped." >&2
elif ! docker image inspect "$RYUK" >/dev/null 2>&1; then
  echo "Fetching the testcontainers reaper ($RYUK)..." >&2
  if ! docker pull "$RYUK" >/dev/null 2>&1; then
    echo "Could not obtain the reaper image $RYUK." >&2
    echo "Running anyway would SKIP the security corpus and report green," >&2
    echo "so this stops here. Pull it when you next have registry access," >&2
    echo "or set TESTCONTAINERS_RYUK_DISABLED=true to run without cleanup." >&2
    exit 1
  fi
fi

# The whole repo is mounted read-only rather than a list of subdirectories:
# tests reach for `scripts/`, `docs/` and example files, and discovering each
# missing one through a fresh collection error is a slow way to learn the list.
#
# `.venv` is masked by an anonymous volume *only when the host actually has one*.
# The host copy holds win_amd64 wheels, and anything putting it on `sys.path`
# would load Windows binaries inside Linux and fail confusingly — but the mask is
# nested inside the read-only bind at /app, and mounts are applied outermost
# first, so on a tree with no `.venv` the runtime has to create the mountpoint
# through a read-only filesystem and the run dies before pytest starts:
#
#   make mountpoint "/app/.venv": mkdirat ...: read-only file system
#
# `.venv` is gitignored, so a fresh clone hits that on the very first invocation
# — precisely the opaque failure this script exists to remove.
MASK=()
if [ -d "$HERE/.venv" ]; then
  MASK=(-v "/app/.venv")
fi

# The socket alone is not enough, and this is the important part: with only the
# socket, testcontainers starts its sibling on the *host* and these tests cannot
# reach it, so the fixtures take their `pytest.skip` path. The run then reports
# zero failures while silently omitting the security corpus --
# `test_payload_corpus.py`, `test_cache_poisoning.py`, `test_postgres_connector.py`
# and `tests/integration/` -- which is the same class of misleading green this
# script exists to remove.
#
# The reaper is deliberately NOT disabled here. It was, on the stated grounds
# that it 'collides on its fixed container name (409 Conflict)' -- which
# cannot happen: `Reaper._create_instance` names it
# `testcontainers-ryuk-{SESSION_ID}` with a per-process `uuid4()`. With the
# reaper enabled the whole suite passes and leaves nothing behind, so the
# reason for switching off the only cleanup that survives an abnormal exit
# did not survive being checked.
#
# `ci.yml` and `release.yml` both still set the flag on their own Pytest steps.
# Why either does is not recorded anywhere, and on an ephemeral runner the reaper
# makes no difference either way, so both are left alone rather than changed on a
# guess. Each carries a note saying so, and the two move together or not at all.
#
# THE REAPER IS A PREREQUISITE OF A LOCAL RUN, and it would fail quietly, so
# there is a preflight for it below. It is created inside
# `DockerContainer.start()`, which `tests/security/conftest.py` and
# `tests/test_postgres_connector.py` both wrap in
# `except Exception: pytest.skip(...)` -- so on a fresh, offline or rate-limited
# machine a reaper that cannot start does not fail the run; it removes the
# security corpus from it and reports green. That is the exact class of
# misleading result described twenty lines above, and enabling the reaper is
# what gives it this new way to fire, so the check fails fast instead of the
# comment merely warning about it. The `pytest.skip` swallowing itself predates
# this and is left alone.
#
# The host override plus `host-gateway` is what lets this container reach the
# published port.
#
# Measured: `test_payload_corpus.py` and `test_postgres_connector.py` went from
# skipped to 37 passed, 3 xfailed, 0 skipped.
#
# The Docker socket is passed through because parts of the suite use
# testcontainers to stand up a real Postgres. Without it those tests do not
# skip, they ERROR on `DockerException` — 22 of them — which at a glance is
# indistinguishable from a genuine failure.
#
# Read-only, so a test that writes into the source tree fails here rather than
# silently leaving state on the host — see the note about tests escaping into
# ~/.nlqueries.
exec env MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$MOUNT:/app:ro" \
  ${MASK[@]+"${MASK[@]}"} \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  --add-host host.docker.internal:host-gateway \
  -e TESTCONTAINERS_HOST_OVERRIDE=host.docker.internal \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -w /app \
  "$IMAGE" \
  pytest -p no:cacheprovider "$@"
