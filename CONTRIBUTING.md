# Contributing to NLQueries Core

Thank you for your interest in contributing! NLQueries Core is open-source under [BSL 1.1](LICENSE).

---

## Contributor License Agreement (CLA)

**Every contributor must sign the CLA before any pull request can be merged.**

We use [CLA Assistant](https://cla-assistant.io/) for this. When you open a PR, the CLA Assistant bot will automatically prompt you to sign the agreement if you have not already done so. The PR will be blocked from merging until the CLA is on file.

If you are contributing on behalf of a company or organization, your authorized signatory must sign the Corporate CLA. Contact [hello@nlqueries.com](mailto:hello@nlqueries.com) to arrange this.

---

## Development Setup

```bash
git clone https://github.com/nlqueries/nlqueries.git
cd nlqueries
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Branching

| Branch | Purpose |
|--------|---------|
| `main` | Stable, released code |
| `develop` | Integration branch for features |
| `feature/<name>` | Feature work — branch off `develop` |
| `fix/<name>` | Bug fixes |

Open PRs against `develop`, not `main`.

---

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check .          # lint
ruff format .         # format
mypy nlqueries/       # type-check
```

CI will fail if any of these report errors.

---

## Tests

```bash
pytest
pytest --cov=nlqueries --cov-report=term-missing   # with coverage
```

All new features need tests. Aim for >90% coverage on new code.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(connectors): add Snowflake connector
fix(embeddings): handle empty query list
docs(cli): update --help text
```

---

## Reporting Issues

Use [GitHub Issues](https://github.com/nlqueries/nlqueries/issues). For security vulnerabilities, see [SECURITY.md](SECURITY.md) — do not file a public issue.

---

## Code of Conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant).
