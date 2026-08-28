#!/usr/bin/env bash
# Regenerate the hashed dependency lock (SEC-11).
#
# pip-tools locks are specific to the resolving platform and Python version, so
# the lock MUST be compiled for the environment the image is built for — linux,
# Python 3.11, the runtime base. This does that inside a throwaway container, so
# the result does not depend on the developer's OS or Python.
#
# Usage:   ./scripts/lock-deps.sh
# Output:  requirements/core.lock  (committed)
#
# The connector extras (snowflake, bigquery, mssql, ...) are deliberately not in
# this closure. They are optional installs whose wheels are large and partly
# platform-specific, and the deployment that needs one pins it in its own lock —
# enterprise does exactly that. Locking them here would make every core build
# carry drivers almost nobody uses.
#
# --allow-unsafe is required and is not what it sounds like: it tells pip-tools
# to pin setuptools, which it otherwise leaves floating. A lock installed with
# --require-hashes must pin everything, so without it pip either falls back to
# whatever setuptools the base image happens to carry, or refuses to install.
set -euo pipefail

# Git Bash rewrites container-side paths that look like Windows ones, turning
# `-w /w` into `W:/`. A no-op on Linux and macOS.
export MSYS_NO_PATHCONV=1

PIP_TOOLS_VERSION="7.4.1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm -v "${REPO_ROOT}:/w" -w /w python:3.11-slim bash -lc "
  pip install --quiet 'pip-tools==${PIP_TOOLS_VERSION}' &&
  pip-compile --quiet --generate-hashes --allow-unsafe \
    --extra dev \
    --output-file requirements/core.lock \
    pyproject.toml
"
echo "Wrote requirements/core.lock"
