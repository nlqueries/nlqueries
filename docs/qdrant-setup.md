# Setting Up Qdrant

Qdrant is a vector database NLQueries uses for three things:

| Feature | What's stored |
|---|---|
| `process-history --embed` | Query capsule embeddings for semantic search |
| Semantic cache | Cached answers indexed by question embedding |
| Document connectors | Document chunk embeddings for retrieval |

Qdrant is **optional** — NLQueries works without it, but `--embed`, semantic caching, and document retrieval won't be available.

## Check if it's already running

```bash
curl http://localhost:6333/healthz
# Windows PowerShell: Invoke-WebRequest http://localhost:6333/healthz
```

> **Qdrant v1.10 or newer is required.** NLQueries searches through the
> Universal Query API (`query_points`), which Qdrant added in v1.10. Against an
> older server every vector search returns `404` — the semantic cache falls back
> to exact-match hits only, dynamic context injection finds nothing, and document
> retrieval returns nothing. Several of those paths treat a failed search as an
> empty result, so the symptom is a system that answers, slowly and without
> context, rather than one that reports an error. Check the `version` field in
> the response below.

A `200 OK` with `{"title":"qdrant","version":"..."}` means it's up — skip to [Configure NLQueries](#configure-nlqueries-to-use-qdrant).

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
cp .env.example .env   # set your LLM API key
docker compose up
```

Available at `http://qdrant:6333` inside the Compose network, `http://localhost:6333` on your host.

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

NLQueries creates its collection (`nlqueries` by default) automatically on first use.

## Verify

```bash
curl -s http://localhost:6333/healthz
nlqueries cache stats <connector-or-alias>
```

If Qdrant is reachable, `cache stats` prints collection statistics. A connection error means the port doesn't match `QDRANT_URL` or the container/binary isn't running — see [troubleshooting.md#w3](troubleshooting.md#w3--qdrant-connection-refused).
