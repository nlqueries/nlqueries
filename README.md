# nlqueries-core

[![CI](https://github.com/nlqueries/nlqueries/actions/workflows/ci.yml/badge.svg)](https://github.com/nlqueries/nlqueries/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nlqueries-core)](https://pypi.org/project/nlqueries-core/)
[![Python](https://img.shields.io/pypi/pyversions/nlqueries-core)](https://pypi.org/project/nlqueries-core/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue)](LICENSE)

**NLQueries Core** turns plain-English questions into validated SQL, builds a self-updating YAML knowledge base from your schema and query history, and exposes everything as an MCP server your AI assistant can call directly. It also answers questions from your documents (PDF, Word, Excel, Notion, Confluence) and can blend both in a single hybrid answer.

> `nlqueries-core` is the standalone OSS library and CLI. For the full web UI, team auth, and admin panel, see [nlqueries-enterprise](https://nlqueries.dev).

---

## Features

| Capability | Description |
|---|---|
| **Database connectors** | PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, SQL Server / Azure SQL, DuckDB |
| **Document connectors** | PDF, Word, Excel, Notion, Confluence — ask questions over ingested documents with citations |
| **Query pipeline** | Filter, cluster, and parameterize query history into reusable `QueryCapsule` templates |
| **Knowledge base** | Auto-generated YAML schema + capsule file, with coverage reporting via `kb-stats` |
| **Multi-agent orchestration** | Routes each question to a SQL agent, document agent, or both in parallel (hybrid) |
| **Semantic cache** | Returns previously-answered similar questions in under 50 ms, no LLM or DB round-trip |
| **Embedding daemon** | Keeps the embedding model resident in memory — ~10 ms per call instead of ~9 s |
| **LLM client** | Anthropic, OpenAI, or any LiteLLM-supported provider |
| **MCP server** | Query execution and schema/knowledge lookup exposed as MCP tools for Claude, Cursor, etc. |
| **CLI** | `nlqueries` (or the shorter `nlq` alias) — connect, build, query, and inspect from your terminal |

See [docs/architecture.md](docs/architecture.md) for how these pieces fit together.

---

## Quickstart

**Prerequisite:** Python 3.11 or 3.12. **Python 3.14+ is not yet supported** — see [docs/troubleshooting.md](docs/troubleshooting.md#w6--pydantic-v1-incompatibility-python-314) before installing on a newer interpreter.

### Option A — Docker (recommended)

```bash
git clone https://github.com/nlqueries/nlqueries.git
cd nlqueries/core
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY or OPENAI_API_KEY at minimum
docker compose up
```

This starts Qdrant (`:6333`) and the NLQueries core service with its MCP server (`:8080`). Run CLI commands against the running stack from a second terminal:

```bash
docker exec -it nlqueries-core nlqueries health
```

### Option B — pip install

```bash
pip install nlqueries-core
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
nlqueries health
```

Optional extras for specific connectors:

```bash
pip install "nlqueries-core[mysql]"     # MySQL
pip install "nlqueries-core[redshift]"  # Amazon Redshift
pip install "nlqueries-core[mssql]"     # SQL Server / Azure SQL
pip install "nlqueries-core[duckdb]"    # DuckDB
pip install "nlqueries-core[docs]"      # PDF / Word / Excel ingestion
pip install "nlqueries-core[wiki]"      # Notion / Confluence sync
```

### First query

```bash
nlqueries connect postgres --host localhost --database mydb --user alice --password secret --alias dev
nlqueries process-history dev --days 30 --annotate
nlqueries export-kb dev
nlqueries query dev "How many orders shipped last month?"
```

Full walkthrough: [docs/getting-started.md](docs/getting-started.md).

---

## Documentation

| Doc | Covers |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Step-by-step setup and your first query |
| [docs/cli-reference.md](docs/cli-reference.md) | Every command and flag |
| [docs/connectors.md](docs/connectors.md) | Database and document connector setup, per-connector notes and caveats |
| [docs/configuration.md](docs/configuration.md) | Environment variables |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common warnings and errors explained |
| [docs/qdrant-setup.md](docs/qdrant-setup.md) | Setting up Qdrant (required for embeddings, semantic cache, document search) |
| [docs/architecture.md](docs/architecture.md) | Module layout and request flow |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributors must sign the CLA before a PR can be merged — see [CONTRIBUTOR_LICENSE_AGREEMENT.md](CONTRIBUTOR_LICENSE_AGREEMENT.md).

---

## License

[Business Source License 1.1](LICENSE) — each release converts to Apache 2.0 four years after its release date.
