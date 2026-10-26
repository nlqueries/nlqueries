# Setting Up Qdrant

Qdrant is a vector database. NLQueries creates these collections in it:

| Collection | What's stored | Written by | Read by |
|---|---|---|---|
| `cache_<agent>` | Cached answers indexed by question embedding | the semantic cache, as questions are answered | the semantic cache; `cache stats`, `cache list`, `cache clear` |
| `doc_<source_id>_chunks` | Document chunk embeddings | `doc-ingest`, `doc-sync-notion`, `doc-sync-confluence` | `doc-ask` |
| `agent_<id>_verified` | Question/SQL pairs from positively rated feedback | `promote-feedback`, and `export-kb` at the end of its run | the question path, as example queries |
| `agent_<id>_schema` | Table and column description embeddings, and the agent's query capsules | tables and columns: `generate_knowledge_base(..., embed=True)` in the Python API (no `nlqueries` command passes `embed=True`); capsules: `process-history --embed`, which replaces the agent's capsules on each run | the question path: relevant tables, and capsules as example queries |

Each collection is created on first write with 384-dimension vectors (`nlqueries.config.EMBED_DIMENSIONS`, a constant rather than a setting), cosine distance and INT8 scalar quantization.

Every name above uses the agent's (or source's) storage name: letters, digits, `_`, `-` and `.` are kept and every other character becomes `_`, so `postgres:localhost:mydb` becomes `cache_postgres_localhost_mydb`. The knowledge-base, feedback and session files use the same name. A connector ID and its sanitised form are therefore one agent on purpose: the CLI uses `postgres:localhost:mydb`, the MCP server knows the agent by its knowledge-base file name, `postgres_localhost_mydb`, and both reach the same files and collections. Verified examples and capsules are matched to the agent by storage name too, so a pair promoted or a capsule embedded from the CLI is used for the MCP server's questions, and the reverse.

When upgrading: a document collection written under the raw ID by an earlier release (`doc_postgres:localhost:mydb_chunks`) is still read, by a caller that uses that raw ID, until a collection under the new name holds the source's chunks. The next successful `doc-ingest` for the source writes the new name and deletes the raw-ID collection; `doc-ingest --allow-empty` removes both. The MCP server knows such a source only by its sanitised ID, so it reads only the new name: a document source whose ID contains a replaced character (`:`, `/`, a space) is visible to it once it has been ingested again, as it was not before either. A cache collection for an ID containing `.` gets a new name and starts empty; the old one is not read, because under the old rule it was also the cache of the ID with `_` in place of each `.`. Nothing reads the old one after the upgrade; delete it in Qdrant to reclaim the space. A schema collection written under the raw ID is read the same way, separately for its two kinds of point. That applies to an ID with a character Qdrant accepts in a collection name but the storage name replaces, such as `@` or a space. Qdrant refuses `:` in a collection name, so for a connector ID such as `postgres:localhost:mydb` no raw-ID schema collection could be created, and its table hints and capsules could not be stored until now. Tables and columns are read from it until `generate_knowledge_base(..., embed=True)` writes them under the new name and deletes it; capsules are read from the new name once `process-history --embed` has written them there.

**Example queries.** For each question the question path searches the agent's verified examples and its capsules in `agent_<id>_schema`, nearest first. Capsules are signed with the cache's signing key (`NLQ_CACHE_SIGNING_KEY`) when `process-history --embed` writes them, and a capsule whose signature does not verify is never shown to the model: one written into Qdrant by anything else, or one embedded before capsules were signed. A signed capsule is also left out when its text, embedded again, does not score against the question as its stored vector did, as for verified examples: the signature does not cover the vector. Every process that answers questions must have the key that signed them, as for verified examples. When the searches give no example (Qdrant out of reach, nothing embedded for the agent, or nothing near the question), the first capsules in the knowledge base file that `export-kb` wrote are used, most frequent first, without ranking them against the question.

Releases before capsules were signed wrote them to a collection named by `QDRANT_COLLECTION` (default `nlqueries`), which nothing read. That setting is gone; run `process-history --embed` once per agent, and delete the old collection in Qdrant.

Qdrant is **optional** — NLQueries works without it, but semantic caching, document retrieval and the vector searches above won't be available.

## Check if it's already running

```bash
curl http://localhost:6333/
# Windows PowerShell: Invoke-WebRequest http://localhost:6333/
```

A `200 OK` means it's up, and the body carries the version:

```json
{"title":"qdrant - vector search engine","version":"1.18.2","commit":"..."}
```

> **Qdrant v1.10 or newer is required.** NLQueries searches through the
> Universal Query API (`query_points`), which Qdrant added in v1.10. Against an
> older server every vector search returns `404` — the semantic cache falls back
> to exact-match hits only, dynamic context injection finds nothing, and document
> retrieval returns nothing. Several of those paths treat a failed search as an
> empty result, so the symptom is a system that answers, slowly and without
> context, rather than one that reports an error. Check the `version` above
> before assuming a quiet system is a working one.

> **Keep the client within one minor of the server.** `qdrant-client` compares
> the two and warns on every connection when the minor versions differ by more
> than one — "Qdrant client version X is incompatible with server version Y". It
> is a warning rather than a failure, and searches keep working, but it is easy
> to mistake for the fault above. `requirements/core.lock` pins client 1.19.0,
> which suits the v1.18.2 the compose file ships. If you run your own Qdrant at
> an older version, install a client to match it — the `>=1.10` floor in
> `pyproject.toml` is about the query API existing, not about which server you
> point at.

(`/healthz` also answers, but only with the plain text `healthz check passed` —
it tells you the server is alive, not whether it is new enough.)

### Upgrading an existing Qdrant

**If your `qdrant-data` volume was written by v1.9.x, it must be removed.** There
is no data-preserving path forward. The failure is loud — the container exits on
startup with:

```
Failed to deserialize segment.json: unknown variant `on_disk`,
expected `mmap` or `in_ram_mmap`
```

Measured, by writing a collection with v1.9.3 and reopening the same volume:

| upgraded to | result |
|---|---|
| v1.10.1 | starts, data intact |
| v1.12.4 | starts, data intact |
| v1.18.2 | **panics on startup and exits** |
| v1.9.3 → v1.12.4 → v1.18.2 | **panics at the last hop, identically** |

The last row is the one that matters: stepping one minor at a time does **not**
carry the storage forward. Running an intermediate version postpones the reset
rather than avoiding it, because nothing rewrites the old segment format on the
way through.

Nor is staying on an intermediate version a migration. `qdrant-client` 1.19.0,
which this project locks, treats a server more than one minor behind as
incompatible and warns on every client construction — so pinning v1.12.4 buys a
warning rather than a working upgrade.

```bash
docker compose down
docker volume rm "$(basename "$PWD")_qdrant-data"   # this volume only
docker compose up -d
```

**Not `docker compose down -v`.** That removes *every* named volume in the
stack, and this one also defines `nlqueries-data`, which holds knowledge bases,
`connectors.yaml`, capsules and feedback — none of which the Qdrant version has
anything to do with, and not all of which is regenerable. Name the volume.

Compose prefixes volume names with the project name, which defaults to the
directory the file sits in; `docker volume ls` will show the exact name if the
command above does not match.

The benchmarks stack in `benchmarks/docker-compose.yaml` keeps its own Qdrant and
its own volume, and was pinned to v1.9.2 — so it fails the same way, and the
command above will not match it. Started with `-f benchmarks/docker-compose.yaml`,
Compose takes the project name from that file's directory, so the volume is
created as `benchmarks_qdrant_bench_data` rather than the `qdrant_bench_data`
written in the file:

```bash
docker compose -f benchmarks/docker-compose.yaml down
docker volume rm benchmarks_qdrant_bench_data
```

Nothing in it is worth preserving — a benchmark run rebuilds its own fixtures.

**What that costs.** Nothing in Qdrant here is a system of record, but the parts
are not equally cheap to rebuild. The semantic cache regenerates on its own as
questions are asked. The capsules in `agent_<id>_schema` come back by running
`nlqueries process-history --embed` again, and the verified examples from
`nlqueries promote-feedback` (or the next `export-kb`), which reads the feedback
records kept outside Qdrant. Its table and column points come back only by calling
`generate_knowledge_base(..., embed=True)` again from Python. Document chunks need
their sources re-ingested, which is the only part that costs real time — if you
have ingested a large corpus, plan for it rather than discovering it afterwards.


## Option A — Docker (recommended for local development)

```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

The ports are bound to `127.0.0.1` so the container is reachable from this machine only. Qdrant starts with no authentication unless you give it a key, and an unauthenticated vector store is a write path into the semantic cache — cached SQL is executed against your database. Publishing it on every interface, which is what `-p 6333:6333` does, offers that write path to anything that can route to the host.

NLQueries will not stop you here, because its own check reads `QDRANT_URL`: `http://localhost:6333` is loopback, so no key is required no matter what the container publishes. The two are independent, and the bind is the half that check cannot see.

To reach Qdrant from another machine, set a key on both sides rather than widening the bind alone — `QDRANT__SERVICE__API_KEY` on the container and `QDRANT_API_KEY` for NLQueries (`openssl rand -hex 32`). NLQueries requires the key for any non-loopback `QDRANT_URL` in any case.

Data persists in the `qdrant_storage` named volume across restarts. Manage it with `docker stop/start qdrant`, `docker rm -f qdrant` (data preserved), or `docker volume rm qdrant_storage` (data deleted).

Port 6333 (HTTP REST) is what NLQueries uses; port 6334 (gRPC) is optional.

## Option B — Docker Compose (bundled with the NLQueries stack)

If you're already running the NLQueries Docker Compose stack, Qdrant is included — no separate setup:

```bash
cp .env.example .env
# edit .env: set your LLM API key, QDRANT_API_KEY and NLQ_MCP_STATIC_TOKEN
# (generate each with: openssl rand -hex 32)
docker compose up
```

`QDRANT_API_KEY` and `NLQ_MCP_STATIC_TOKEN` ship empty in `.env.example`, and `docker compose up` refuses to start until both are set. The compose file starts Qdrant with `QDRANT_API_KEY` as its API key and hands the same key to the NLQueries container.

Available at `http://qdrant:6333` inside the Compose network, and at `http://localhost:6333` on your host (published on loopback only, unless you set `BIND_ADDR`). Requests from tools on the host need the key as well, sent in Qdrant's `api-key` header.

## Option C — Qdrant Cloud (managed, no local install)

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) and create a cluster (free tier available)
2. Copy the Cluster URL and API Key from the dashboard
3. Set both:
   ```bash
   export QDRANT_URL="https://<your-cluster>.qdrant.io"
   export QDRANT_API_KEY="<your-api-key>"
   ```

## Option D — Native binary (no Docker)

```bash
# Linux
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz | tar -xz
./qdrant

# macOS
brew install qdrant && qdrant

# Windows — download qdrant-x86_64-pc-windows-msvc.zip from
# https://github.com/qdrant/qdrant/releases, extract, then run:
.\qdrant.exe
```

Starts on port 6333 by default; data stored in `./storage` relative to the binary.

## Configure NLQueries to use Qdrant

```bash
export QDRANT_URL=http://localhost:6333
```

NLQueries creates each collection listed at the top of this page automatically on its first write.

## Running Qdrant in production

### How NLQueries connects

Every Qdrant client in NLQueries is built in one place (`nlqueries.embeddings.qdrant_client.build_qdrant_client`), from two settings:

| Setting | Default | Notes |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | The REST endpoint. A URL with no scheme, such as `qdrant:6333`, is plain http |
| `QDRANT_API_KEY` | empty | Sent with every request when set |

Nothing else is configurable: no timeout, and no gRPC. NLQueries talks REST to the port in `QDRANT_URL`, so port 6334 can stay unpublished.

**A Qdrant that is not on this machine needs a key.** `localhost`, `127.0.0.0/8` and `::1` count as this machine; every other host, a private Docker network included, does not. For any other `QDRANT_URL`, an empty `QDRANT_API_KEY` stops the first Qdrant call with `QdrantAuthenticationRequired`. There is no setting to turn this off. The reason is the semantic cache: it stores SQL that is executed against your database, so a Qdrant that anyone can write to is a way to run SQL. Set the same key on the server, as `QDRANT__SERVICE__API_KEY`.

**Use https when the key crosses a network.** A key sent to a host other than this machine over plain `http://` is logged as a warning, once per URL, and not refused. Anyone on that path can read the key. Use an `https://` URL (Qdrant Cloud serves one), or keep Qdrant on a network nothing else can reach.

### Several NLQueries processes, one Qdrant

Point every process at the same `QDRANT_URL`, and give them all the same two keys. Otherwise each one generates its own, under its own state directory:

- **`NLQ_CACHE_SIGNING_KEY`** signs cached answers, capsules and verified examples. An entry another process signed with a different key does not verify. It is then treated as a cache miss, or left out as an example ([configuration](configuration.md)).
- **`NLQ_CACHE_ENCRYPTION_KEY`**, if [encryption](#encrypting-what-qdrant-stores) is on. Text another process encrypted under a different key cannot be read.

### What to back up

Qdrant holds nothing that cannot be rebuilt, but the parts differ in what rebuilding costs: see [What that costs](#upgrading-an-existing-qdrant). Document chunks are the expensive part, because every source has to be ingested again. If you have ingested a large corpus, back Qdrant up.

Qdrant's own tools do this: its snapshot API, per collection or for the whole storage, or a copy of its storage directory (`/qdrant/storage` in the container) taken while Qdrant is stopped. Qdrant Cloud has backups of its own. Qdrant's documentation covers all three.

Restoring a backup restores signed and encrypted entries, which are only useful with the keys that wrote them. Keep `NLQ_CACHE_SIGNING_KEY` with the deployment's configuration, so a restored Qdrant still verifies. Keep `NLQ_CACHE_ENCRYPTION_KEY` too, but apart from the backups, as [Encrypting what Qdrant stores](#encrypting-what-qdrant-stores) says.

### Sizing

There are no measured figures for NLQueries' footprint in Qdrant, so plan from what is stored:

- **Points:** one per cached answer, per document chunk, per verified example, and per table, column and capsule an agent has embedded.
- **Vectors:** 384 dimensions, cosine distance.
- **Quantization:** INT8 scalar quantization on every collection NLQueries creates, with the quantized vectors kept in RAM (`always_ram`).

Document chunks are usually the largest share, and they grow with what you ingest. Watch Qdrant's own memory and disk use as a corpus grows.

### The Docker Compose service

The compose file's `qdrant` service is a reasonable model for a single-host deployment:

- **A pinned version:** `qdrant/qdrant:v1.18.2`. Move it deliberately, after reading [Upgrading an existing Qdrant](#upgrading-an-existing-qdrant).
- **A required key:** `QDRANT__SERVICE__API_KEY` is taken from `QDRANT_API_KEY`, and compose refuses to start without it.
- **Ports published on loopback only** (`BIND_ADDR`, default `127.0.0.1`). The NLQueries container reaches Qdrant over the compose network as `http://qdrant:6333`.
- **Storage in the named volume `qdrant-data`.**
- **Privileges:** all Linux capabilities dropped, and `no-new-privileges` set.
- **A health check** that opens the REST port, which `nlqueries-core` waits on before it starts.

## Encrypting what Qdrant stores

Qdrant keeps the text of cached answers and of every ingested document chunk, readable by anyone with access to its storage, its snapshots or its API. With `NLQ_CACHE_ENCRYPTION=true`, both are encrypted with AES-256-GCM before they are stored and decrypted when they are read; searching is unchanged, since it runs on vectors computed from the text before it is encrypted.

- **The key** is `NLQ_CACHE_ENCRYPTION_KEY`, or the file `NLQ_CACHE_ENCRYPTION_KEY_FILE` names, or one generated on the first encrypted write and kept as `cache_encryption_key` in the state directory. Set your own as a random value (`openssl rand -hex 32`), never a password or phrase: anyone holding a copy of Qdrant's data can try guesses against it offline. When more than one process or container writes to the same Qdrant, give them all the same key: a generated one belongs to one state directory. Keep a copy, and keep it apart from Qdrant's backups, which it is there to protect.
- **Each chunk's text is bound to the document identifier it was ingested under:** given another identifier, it does not decrypt. The collection is not part of it, so a point copied whole into another collection still decrypts there, and so does a snapshot restored under another name.
- **What stays readable:** the vectors, which are derived from the text and can say something about it, and the questions, SQL, document names, chunk metadata and identifiers the store searches, filters and verifies on.
- **Turning it on** encrypts what is written from then on. What is already stored stays readable as it is: ingest the documents again and run `nlqueries cache clear <agent>` to have them stored encrypted.
- **Turning it off** stores new text unencrypted. Encrypted text is still read while the key is available.
- **Without the key**, encrypted entries cannot be read: a cached answer is a miss and is answered afresh, and a document chunk is left out of retrieval, with one warning per process for each of the two. Ingesting the documents again restores them.

## Verify

```bash
nlqueries health
```

The first check listed is Qdrant: `[OK]` with `reachable at <host> (version …, … ms)` when a request to `QDRANT_URL` succeeds, or `[FAIL]` with `connection refused — is Qdrant running at <url>?` when it does not. The check sends no `QDRANT_API_KEY`, and it reports an error response from the server with the same message as a refused connection; `nlqueries health --verbose` shows the underlying error. A refused connection means the port doesn't match `QDRANT_URL` or the container/binary isn't running — see [troubleshooting.md#w3](troubleshooting.md#w3--qdrant-connection-refused).

(`nlqueries cache stats` is not a connectivity check: it reports `Total entries: 0` when Qdrant cannot be reached.)
