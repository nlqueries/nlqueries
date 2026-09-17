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
# `.venv` is masked by an anonymous volume. The host copy holds win_amd64
# wheels; left visible it shadows nothing by default, but anything that adds it
# to `sys.path` would load Windows binaries inside Linux and fail confusingly.
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
  -v "/app/.venv" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -w /app \
  "$IMAGE" \
  pytest -p no:cacheprovider "$@"
