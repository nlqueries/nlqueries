# Security Policy

## Supported Versions

`nlqueries-core` is pre-1.0 (currently `0.3.x`). Security fixes are released against the latest published version on [PyPI](https://pypi.org/project/nlqueries-core/) — there is no separate long-term-support branch at this stage.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **[security@nlqueries.com](mailto:security@nlqueries.com)** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal reproduction is very helpful)
- The version of `nlqueries-core` affected (`nlqueries --version` or `nlq --version`)

We aim to acknowledge reports within 5 business days. Once a fix is available, we'll credit the reporter (unless you'd prefer to stay anonymous) in the release notes and coordinate disclosure timing with you.

## The container image

The `nlqueries/core` image is built from this repository and scanned for known vulnerabilities in its OS packages and Python dependencies every week (from `main`), whenever a change to `main` alters what goes into the image, and for each release tag. High and critical findings that have a fix available are reported in this repository's Security tab. The scan covers the `linux/amd64` build.

## Hardening the database you point it at

NLQueries generates SQL with a language model and runs it against your database.
Every connector runs the query in the most restrictive execution its engine
offers -- a read-only transaction where one exists, an uncommitted transaction
that is always rolled back where one does not -- but that is one layer, it is
weaker on some engines than others, and it does not stop a role that can read
files off the database host or tables you never meant to expose.

The database's own permission system is the boundary that does not depend on
this project being free of bugs. See
[docs/database-hardening.md](docs/database-hardening.md) for the role to create,
per engine.

## Scope

This policy covers the `nlqueries-core` OSS package (this repository). For the enterprise edition (web UI, team auth, admin panel), report through your enterprise support channel or the same [security@nlqueries.com](mailto:security@nlqueries.com) address.

Note that `nlqueries-core` stores database passwords in your OS keychain (via the `keyring` package) rather than in plain text. Connection metadata (host, port, database, user) is kept in `~/.nlqueries/connectors.yaml` by default (`NLQ_STATE_DIR` or `CONNECTORS_FILE` moves it; mode `0600`) without the password. If keyring is unavailable on the machine (e.g. headless CI with no secret service running), the password falls back to being embedded in that file instead, and `connect` prints a warning when this happens. LLM API keys are read from environment variables, which the `nlqueries` command also loads from a `.env` file in the working directory, and not from one in a parent directory (variables already set in the environment take precedence). Nothing is transmitted anywhere except:

- the target database;
- the configured LLM provider;
- the Qdrant server at `QDRANT_URL` (default `http://localhost:6333`, but it may be on another host), which holds, depending on the commands you use, Query Capsules mined from query history, semantic-cache entries (questions, generated SQL and answers), promoted feedback, the text of ingested document chunks, and schema descriptions when a knowledge base is generated with embeddings through the Python API;
- Hugging Face Hub, from which the embedding model (`all-MiniLM-L6-v2` unless `NLQ_EMBED_MODEL` names another) is downloaded once on first use and then loaded from the local cache;
- when you use them, the Notion or Confluence API (`doc-sync-notion`, `doc-sync-confluence`) and your identity provider's discovery and JWKS endpoints (OIDC token verification).

If you find a case where that's not true, that's exactly the kind of issue this policy is for.
