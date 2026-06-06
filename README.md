# nlqueries-core

[![CI](https://github.com/nlqueries/nlqueries/actions/workflows/ci.yml/badge.svg)](https://github.com/nlqueries/nlqueries/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nlqueries-core)](https://pypi.org/project/nlqueries-core/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue)](LICENSE)

**NLQueries Core** is the open-source engine that translates natural-language questions into SQL, builds a self-updating YAML knowledge base from your schema, and exposes everything as an MCP server your AI assistant can call directly.

## Features

- **DB Connectors** — PostgreSQL, MySQL, BigQuery, Snowflake (extensible via `BaseConnector`)
- **Query Pipeline** — filter, cluster, and parameterize raw query logs into canonical forms
- **Knowledge Base** — auto-generate and refresh a YAML schema + sample-data file your LLM reads as context
- **Embeddings** — sentence-transformer vectors stored in Qdrant for semantic query matching
- **LLM Client** — thin abstraction over OpenAI / Anthropic; swap backends via config
- **MCP Server** — expose query execution and knowledge lookup as MCP tools
- **CLI** — `nlq` command for all of the above from your terminal

## Quickstart

```bash
pip install nlqueries-core
nlq --help
```

## Architecture

```
nlqueries/
├── connectors/   DB connector implementations
├── processing/   Query filter, clusterer, parameterizer
├── knowledge/    YAML knowledge base generator
├── embeddings/   Sentence-transformer + Qdrant store
├── llm/          LLMClient abstraction
├── mcp_server/   MCP server entry point
└── cli/          CLI commands (click)
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributors must sign the CLA before a PR can be merged.

## License

[Business Source License 1.1](LICENSE) — converts to Apache 2.0 four years after each release date.
