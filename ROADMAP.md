# Roadmap

A short list of what's actively planned or being researched. No committed dates — this is a young OSS project and priorities will shift based on what people actually use. Open a [discussion or issue](https://github.com/nlqueries/nlqueries/issues) if something here matters to you, or if something's missing.

## Near-term

- **Remove the remaining `langchain_text_splitters` dependency** from the document connectors (PDF, Word, Notion, Confluence), replacing it with a small custom chunker. This closes out the last piece of the Python 3.14 compatibility gap — see [docs/troubleshooting.md#w6](docs/troubleshooting.md#w6--pydantic-v1-incompatibility-python-314).
- Continue expanding test coverage on the newer connectors (Redshift, SQL Server, DuckDB) and the multi-agent routing path.

## Under research

- **Nexus layer — relationship and join-path intelligence.** NLQueries currently knows what's in a database (schema, columns, sample rows) but not how tables relate to each other at a semantic level, which is a silent failure mode: wrong joins produce SQL that runs but returns incorrect results. This is an early design exploration, not yet scheduled for implementation.
- **Seeding the knowledge base from public NL→SQL benchmarks** (e.g. Spider) so a fresh, lightly-used database gets useful query capsule examples before it has real query history of its own, rather than starting from schema alone.

## Not currently planned

- We don't have a timeline for expanding beyond the current seven database connectors or five document connectors. If you need one that's missing, an issue with your use case helps prioritize it.
