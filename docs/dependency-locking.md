# Dependency locking (SEC-11)

The image installs its Python dependencies from a **committed, hashed lock file**
rather than re-resolving the `>=` floors in `pyproject.toml` at build time. Two
builds of the same commit install the same bytes, and a yanked or compromised
upstream release cannot be adopted silently.

The base image is pinned by digest for the same reason: a tag is mutable, so
without a digest "the same Dockerfile" does not mean the same image.

## Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Source of truth for direct dependencies, with `>=` floors. |
| `requirements/core.lock` | Compiled, fully pinned, hashed closure. Committed. |
| `scripts/lock-deps.sh` | Regenerates the lock in a container matching the runtime. |

## Regenerating

pip-tools locks are specific to the resolving platform and Python version, so the
lock must be compiled for the runtime target — linux, Python 3.11. Do not run
`pip-compile` from a development machine; the script compiles inside a throwaway
`python:3.11-slim` container so the result does not depend on your OS.

```bash
./scripts/lock-deps.sh
git add requirements/core.lock
```

Regenerate whenever `pyproject.toml` changes. `tests/test_dependency_lock.py`
fails if a declared dependency is missing from the lock, which is the case the
rest of the pipeline would not catch: only the release workflow builds the image,
so a stale lock would otherwise install fine for every developer and fail at the
release.

## Two flags that are not optional

`--allow-unsafe` is required despite the name. It tells pip-tools to pin
setuptools, which it otherwise leaves floating. A lock installed with
`--require-hashes` must pin everything, so without it pip either falls back to
whatever setuptools the base image happens to carry, or refuses to install
outright. The symptom is a `# WARNING` block written into the lock naming the
unpinned package, which is a comment and therefore fails nothing on its own —
`test_the_lock_carries_no_unresolved_warnings` exists for that reason.

`--generate-hashes` is what makes this a supply-chain control rather than a
version pin. Without it the lock fixes the version but not the bytes, and a
re-upload of the same version number would install.

## What is in the closure

`scripts/lock-deps.sh` compiles `pyproject.toml` with `--extra dev` and no other
extra, so the lock holds the base dependencies plus the `dev` extra. Besides the
test and lint tooling (pytest, ruff, mypy, testcontainers), `dev` lists the
Snowflake, BigQuery and DuckDB drivers (`snowflake-connector-python`,
`google-cloud-bigquery`, `google-auth`, `duckdb`) so their tests run, which means
those drivers are in the lock and in the image.

Not in the closure:

- the `mssql`, `redshift`, `mysql` and `onnx` extras. They are optional installs
  whose wheels are large and partly platform-specific, and a deployment that needs
  one installs and pins it itself.
- the `docs` extra (`pdfplumber`, `python-docx`, `openpyxl`) and the `wiki` extra
  (`notion-client`, `atlassian-python-api`, `beautifulsoup4`). An image built from
  the lock cannot ingest PDF, Word or Excel files or sync Notion or Confluence
  sources until those are added.

The lock is used only by the image build. CI does not install from it: the test
jobs in `.github/workflows/ci.yml` and `release.yml` run
`pip install -e ".[dev,docs]"`, which resolves the `>=` floors afresh, so CI can
test newer versions than the image ships.

## pip-tools version

`scripts/lock-deps.sh` pins pip-tools itself. Version 8 changes the default for
`--strip-extras`, which changes how names are written in the lock
(`httpx[http2]==` versus `httpx==`). Pinning keeps the format stable; changing it
is a deliberate edit, not something that happens because a developer's machine
had a newer pip-tools.

## Updating the base image

```
docker buildx imagetools inspect python:3.11-slim-bookworm
```

Take the `Digest` line and update the `FROM` in the `Dockerfile`. Do this
deliberately and on purpose — the value of the pin is that it does not move on
its own.
