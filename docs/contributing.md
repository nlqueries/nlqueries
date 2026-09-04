# Contributing

NLQueries Core is open source under [BSL 1.1](../LICENSE) and maintained by Theorence Labs. Contributions of code, tests, documentation and bug reports are welcome. This page is the short version: what the project is looking for and how to get a change merged. The full policy lives in [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository.

---

## What to contribute

The areas below are where an outside change is most likely to land quickly, because the code is already built to take it.

**Database connectors.** Every engine is a self-contained module under `nlqueries/connectors/` that implements the connector base class, declares its capabilities, and ships with its own tests. Eight are dedicated today (PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, SQL Server / Azure SQL, DuckDB, SQLite) plus a generic SQLAlchemy connector. A new connector needs schema introspection, an optional query-history source, TLS handling, a statement timeout, and read-only execution. Start from the closest existing engine and open an issue first describing the use case, so the query-history source can be agreed before the code is written.

**Document connectors.** PDF, Word, Excel, Notion and Confluence live under `nlqueries/document_connectors/` and implement `ingest()` and `supports()` on the document connector base class. A new format or source follows the same shape and reuses the built-in chunker.

**Golden question sets and eval cases.** `tests/golden/` holds the questions, intents and follow-ups that `nlqueries eval` and the test suite check against. Cases for dialects and schemas that are thinly covered are valuable on their own, without any code change.

**Test coverage.** The roadmap's near-term item is coverage on the newer connectors (Redshift, SQL Server, DuckDB) and the multi-agent routing path. Tests that exercise a real engine behaviour the suite does not yet touch are welcome.

**Documentation.** The pages on this site are generated from `docs/*.md` in the repository. A correction, a missing caveat, or a worked example is a pull request against that folder.

**Bug reports.** Use [GitHub Issues](https://github.com/nlqueries/nlqueries/issues) and attach the output of `nlqueries health` and `nlqueries --version`. For anything security-related, do not open a public issue: follow [SECURITY.md](../SECURITY.md) and email security@nlqueries.com.

Two things are not open for contribution here. The enterprise layer (web UI, Nexus, SQL Console, SSO and the rest of the [Enterprise column](https://nlqueries.com/#enterprise)) is a separate proprietary codebase; issues about it go to sales@nlqueries.com. And the [roadmap](../ROADMAP.md) has no timeline for new connectors beyond the current set, so an issue with your use case is the way to get one prioritised before writing it.

---

## How to contribute

### 1. Sign the Contributor License Agreement

Every contributor must sign the CLA before any pull request can be merged. We use [CLA Assistant](https://cla-assistant.io/) for this. When you open a PR, the CLA Assistant bot will automatically prompt you to sign the agreement if you have not already done so. The PR will be blocked from merging until the CLA is on file.

If you are contributing on behalf of a company or organization, your authorized signatory must sign the Corporate CLA. Contact [hello@nlqueries.com](mailto:hello@nlqueries.com) to arrange this. The agreement text is in [CONTRIBUTOR_LICENSE_AGREEMENT.md](../CONTRIBUTOR_LICENSE_AGREEMENT.md).

### 2. Set up a development environment

```bash
git clone https://github.com/nlqueries/nlqueries.git
cd nlqueries
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 to 3.14 are supported. See [Getting started](getting-started.md) for the runtime prerequisites (an LLM API key, and Qdrant if you are working on embeddings, the semantic cache or document connectors).

### 3. Branch from `develop`

| Branch | Purpose |
|--------|---------|
| `main` | Stable, released code |
| `develop` | Integration branch for features |
| `feature/<name>` | Feature work — branch off `develop` |
| `fix/<name>` | Bug fixes |

Open PRs against `develop`, not `main`.

### 4. Match the code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check .          # lint
ruff format .         # format
mypy nlqueries/       # type-check
```

CI will fail if any of these report errors.

### 5. Add tests

```bash
pytest
pytest --cov=nlqueries --cov-report=term-missing   # with coverage
```

All new features need tests. Aim for >90% coverage on new code. Connector tests that need a live engine live under `tests/integration/` and skip themselves when the driver or Docker is not available.

### 6. Write a conventional commit message

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(connectors): add Snowflake connector
fix(embeddings): handle empty query list
docs(cli): update --help text
```

### 7. Open the pull request

Describe what changed, why, and how it was tested; the repository's pull request template asks for these. A maintainer reviews and merges once the CLA is on file and CI is green.

---

## Code of conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) (Contributor Covenant).

For the complete policy, including anything this page abbreviates, read [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository.
