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
# AFTER AN INTERRUPTED RUN, CHECK `docker ps`. The testcontainers reaper is
# disabled below (it collides on its fixed container name), and the fixtures
# stop their own containers in a `finally` that does not run if you Ctrl-C or
# the container is killed. Each interrupted invocation can therefore leave a
# `postgres:16-alpine` sibling running on the host indefinitely. CI never
# notices because the runner is discarded; a dev machine accumulates them.

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
# `TESTCONTAINERS_RYUK_DISABLED` matches what ci.yml sets; without it the reaper
# collides on its fixed container name (`409 Conflict`). The host override plus
# `host-gateway` is what lets this container reach the published port.
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
  -e TESTCONTAINERS_RYUK_DISABLED=true \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -w /app \
  "$IMAGE" \
  pytest -p no:cacheprovider "$@"
