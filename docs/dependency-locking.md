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

The lock is installed only by the image build. CI does not install from it: the
test jobs in `.github/workflows/ci.yml`, which `release.yml` calls before it
publishes anything, run `pip install -e ".[dev,docs]"`, which resolves the `>=`
floors afresh, so CI can test newer versions than the image ships.

CI does audit the lock. The `lock-audit` job in `ci.yml` runs
`pip-audit -r requirements/core.lock --require-hashes` on every CI run, and a
release waits on it like every other CI job. A pinned version with a published
advisory fails the job, and a release with it, until the lock is regenerated
(see [Regenerating](#regenerating)) at a fixed version. The separate `security`
job audits the resolved `.[dev,docs]` environment, which includes the `docs`
extra the lock leaves out.

## When an audit fails during a release

`release.yml` runs every `ci.yml` job before it builds anything, so a failed
audit stops a release before PyPI or Docker Hub receive a file. What to do
depends on which job failed and where the flagged package ends up.

1. **Read the job log.** It names the package, the installed version and the
   advisory. If it shows a network error instead of a finding, re-run the job
   once: `lock-audit` downloads the lock's whole closure to check its hashes,
   and a dropped connection fails it.
2. **Find out whether the flagged version ships.** An advisory names a range
   of affected versions, and the two jobs see different ones: `lock-audit`
   the lock's pins, `security` whatever `pip install -e ".[dev,docs]"`
   resolves that day.
   - **In `requirements/core.lock` at an affected version:** it is in the
     image. That includes everything in the `dev` extra, the test and lint
     tools among it, and setuptools, because the lock is compiled with
     `--extra dev` and `--allow-unsafe` (see
     [What is in the closure](#what-is-in-the-closure)). `lock-audit` fails
     on it.
   - **In the lock, but its pin is outside the affected range:** the image is
     not affected. Only the `security` job's fresh resolution reached a newer,
     affected release, and that is what pip users installing today would get.
   - **Not in the lock, but required by `nlqueries-core`:** through the `docs`
     extra, or as a transitive dependency that a newer release of a runtime
     dependency pulls in when the floors are resolved afresh. In the job's
     environment, `pip show <package>` lists what requires it on its
     `Required-by` line; follow that up to `nlqueries-core`. It reaches
     everyone who installs the package with pip.
   - **Required by nothing in `nlqueries-core`'s tree:** a tool the job
     installs for itself, such as pip-audit and its own dependencies. Only
     this does not ship, and it is shown positively, by that `Required-by`
     chain, not assumed because the package is missing from the lock.
3. **If a fixed version exists and can be reached, move to it.** If the
   lock's pin is affected, regenerate the lock (see
   [Regenerating](#regenerating)). If only a fresh resolution is affected,
   raise the floor in `pyproject.toml` past the fix, or, when the advisory
   affects only releases newer than the lock's pin, cap it below the affected
   range. For a tool the job installs for itself, upgrade it in the job. The
   `security` job already upgrades setuptools in its own environment before
   auditing; setuptools is in the lock too, so a setuptools advisory also
   needs the lock regenerated. A fix that only exists beyond a cap in
   `pyproject.toml` (`mcp>=1.29,<2`, say, when moving to 2.x is a port of its
   own) cannot be reached this way: treat it as step 4 until that port is
   done. Then release from the commit that carries the fix.
4. **If no reachable fix exists:**
   - A tool the job installs for itself may be skipped in the `security` job
     with `pip-audit --ignore-vuln <ID>`. A comment beside it names the
     package, says why it does not ship, and gives the date; the ignore comes
     out when a fix is released.
   - A package that ships is not ignored as a matter of course. Replace or
     remove the dependency, or hold the release. The one exception is the
     maintainer's: when the advisory is judged not exploitable in how
     nlqueries uses the package, a time-limited ignore, in whichever job
     flags it, added in a pull request that says why and until when, with the
     release notes saying what is affected and what users can do. A test or
     lint tool is no exception to the first rule: the image carries it.
     Outside that exception, nothing is ignored in `lock-audit`.

Either way, an ignore is the maintainer's decision, made in the pull request
that adds it.

## SBOMs

Each release publishes a software bill of materials for both of its artefacts.

- **The image.** `release.yml` builds it with a BuildKit SBOM attestation, so
  every platform in the pushed manifest carries an SPDX SBOM listing the
  packages in that image, the lock's closure and the base image's own included.
  Read it with:

  ```
  docker buildx imagetools inspect nlqueries/core:<version> --format '{{ json .SBOM }}'
  ```

  The scanner BuildKit runs to produce it, `docker/buildkit-syft-scanner`, is
  pinned by digest in `release.yml` (`sbom: generator=...@sha256:...`), like the
  base image, so the same commit produces the same SBOM. Update it the same
  deliberate way, with `docker buildx imagetools inspect
  docker/buildkit-syft-scanner:stable-1`.

- **The wheel.** A CycloneDX SBOM. The `wheel-sbom` job installs the wheel into
  a virtualenv created without pip, so the environment holds only the wheel and
  its dependencies, and describes it with `cyclonedx-py`. The result,
  `nlqueries_core-<version>-py3-none-any.whl.cdx.json`, is attached to the
  GitHub release beside the wheel and sdist. It is not uploaded to PyPI.

  The wheel declares `>=` floors rather than the lock, so this SBOM records one
  resolution: the one on linux/amd64 with CPython 3.11, the release runner's
  platform and interpreter, at the time of the release. Other platforms and
  Python versions resolve differently (`torch`, for one, brings CUDA packages
  on Linux and none on macOS), and a later install can resolve newer versions.

  The SBOM job runs after `build` and apart from it. PyPI publishing does not
  wait on it; the GitHub release does, so if generating the SBOM fails, the
  packages still reach PyPI and only the GitHub release is held back until the
  job is re-run.

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
