# Connecting MCP clients

The NLQueries MCP server lets an AI assistant ask your databases questions as a
tool call. This page covers connecting three kinds of client: Claude Desktop
and Cursor on the same machine, over stdio, and any client over the network,
with a bearer token.

Before any client can use an agent, the agent has to exist: register the
database with `nlqueries connect` and build its knowledge base with
`nlqueries export-kb` ([Getting started](getting-started.md), steps 2 to 5). An
agent is a knowledge-base file under `KB_PATH` (`~/.nlqueries/knowledge_base/`
by default), and the server lists only those. Check the agent with
`nlqueries query` before connecting a client: the server runs the same
pipeline, and a problem is easier to see at the command line.

## Which transport

| Client | Transport | Authentication |
|---|---|---|
| Claude Desktop, Cursor and other local clients that start the server themselves | stdio | none: the client starts the process and already has what it has |
| A client on another machine, or several people sharing one server | SSE (`--sse`), or SSE / streamable HTTP in the Docker image | required: a bearer token or an identity provider, plus a grants file |

Over stdio, calls are still authorised and audited, against a principal that is
the process owner ([MCP authentication](mcp-authentication.md#stdio-needs-no-configuration)).

## Claude Desktop

Claude Desktop starts the server itself and talks to it over the process's
pipes. Add an entry to its `claude_desktop_config.json` (Claude Desktop opens
the file from its settings, under Developer):

```json
{
  "mcpServers": {
    "nlqueries": {
      "command": "/path/to/venv/bin/nlqueries",
      "args": ["mcp-server", "start"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

`nlqueries mcp-server start` with no flag is the stdio transport. Two details
decide whether it works the first time.

- **`command` must find the executable.** Claude Desktop does not start from
  your shell, so a `nlqueries` that is only on a virtual environment's `PATH`
  is not found. Give the full path: `.../venv/bin/nlqueries` on macOS and Linux,
  `...\venv\Scripts\nlqueries.exe` on Windows. `nlqueries-mcp`, the package's
  second entry point, starts the same stdio server with no arguments.
- **`env` must carry your settings.** NLQueries reads `.env` from the directory
  it is started in, and only from there ([configuration](configuration.md)).
  Claude Desktop chooses that directory, so a key kept in a project's `.env`
  is not seen. Put the LLM key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in
  `env`, along with any other variable your command-line runs depend on, such
  as `QDRANT_URL` and `QDRANT_API_KEY` for a Qdrant that is not on
  `localhost:6333`, or `NLQ_STATE_DIR` if you moved the state directory.

The connectors file, the knowledge base and the stored passwords are found
the same way the CLI finds them, so nothing else needs passing. Restart Claude
Desktop after editing the file.

## Cursor

Cursor reads the same shape of entry from its `mcp.json`: by Cursor's
convention, `.cursor/mcp.json` in a project or `~/.cursor/mcp.json` for every
project.

```json
{
  "mcpServers": {
    "nlqueries": {
      "command": "/path/to/venv/bin/nlqueries",
      "args": ["mcp-server", "start"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

The same two details apply: a full path in `command`, and every setting the
server needs in `env`.

## Over the network

A networked transport **will not start without authentication**. Configure one
of:

- a pre-shared token: `NLQ_MCP_STATIC_TOKEN`, at least 32 characters
  (`openssl rand -hex 32`), or `NLQ_MCP_STATIC_TOKEN_FILE` to read it from a
  mounted file
  ([option 2](mcp-authentication.md#option-2--a-pre-shared-token));
- an identity provider: `NLQ_MCP_OIDC_DISCOVERY_URL` and `NLQ_MCP_OIDC_CLIENT_ID`
  ([option 1](mcp-authentication.md#option-1--an-identity-provider)).

With either, also set `NLQ_MCP_RESOURCE_URL` to the URL clients use, and
`NLQ_MCP_GRANTS_FILE` to a grants file that says which subjects may do what on
which agents ([grants](mcp-authentication.md#grants)). A subject with no grant
is denied. Then start the server:

```bash
nlqueries mcp-server start --sse --port 8000
```

It binds `127.0.0.1` and refuses a wildcard address. To serve other machines,
put a proxy in front of the loopback port
([cli-reference](cli-reference.md#mcp-server)). Each principal is also limited
in calls per minute and calls in flight ([limits](mcp-authentication.md#limits)).

### The Docker Compose stack

The shipped `docker-compose.yml` serves the MCP server over SSE on port 8080,
published on `127.0.0.1` by default. It requires `NLQ_MCP_STATIC_TOKEN` in
`.env` and will not start without it. It writes a grants file that gives that
token's subject (`NLQ_MCP_STATIC_SUBJECT`, `operator` by default) every action
on every agent, and sets `NLQ_MCP_RESOURCE_URL` to `http://localhost:8080`. See
[Getting started](getting-started.md#option-a-docker-includes-qdrant-no-clone-needed).

### Pointing a client at it

A client connects to the SSE endpoint, `http://localhost:8080/sse` for the
Compose stack, and sends the token on every request:

```
Authorization: Bearer <your NLQ_MCP_STATIC_TOKEN>
```

How a client takes a URL and a header is the client's own configuration. In
Cursor's `mcp.json`, for example, an entry with `url` and `headers` in place of
`command`:

```json
{
  "mcpServers": {
    "nlqueries": {
      "url": "http://localhost:8080/sse",
      "headers": { "Authorization": "Bearer <your NLQ_MCP_STATIC_TOKEN>" }
    }
  }
}
```

With an identity provider, the bearer token is an access token from that
provider for the MCP server's client id. The pre-shared token gives whoever
holds it exactly what the grants file gives its subject; in the Compose stack,
that is everything.

## The tools

| Tool | What it does |
|---|---|
| `list_agents` | The agents the server can use: one per knowledge-base file under `KB_PATH` |
| `get_agent_schema` | An agent's tables, columns, types and foreign keys, from its knowledge base |
| `query` | Answers a question: generates SQL, runs it read-only and returns the rows |
| `submit_feedback` | Records a rating, and optionally corrected SQL, for an answer |
| `health` | Checks the LLM configuration, Qdrant, the embedding daemon and the config; `probe_llm` sends a real completion |
| `invalidate_cache` | Drops an agent's semantic cache, for after a schema change or data refresh |
| `list_connectors` | The registered connectors and their types, with passwords and full URLs redacted |
| `get_query_history` | An agent's recent questions and their ratings, newest first |
| `get_cache_stats` | The number of cached entries for an agent, and its cache collection |

### What `query` returns

- the answer;
- the generated SQL;
- the rows, as a table, saying how many were shown when the result was larger;
- the agent type and the latency.

With `paraphrase`, it adds a plain-English reading of the SQL, produced without
an LLM. With `explain`, it adds the answer's provenance: the route, cache hit or
miss, how much knowledge was injected and the checks that ran. A `dialect`
argument is accepted and ignored: the SQL is always written for the database
the agent's connector points at.

For an agent that lists document sources (`nlqueries agent docs add`, see
[cli-reference](cli-reference.md#agent-document-sources)), `query` may answer
from those documents, or from the database and the documents together, as the
question needs, and lists the sources it cited. An agent that lists none
answers from its database only.

The tool waits 45 seconds for an answer and gives the database 40 of them for
the statement, so a slow query is stopped rather than left running after the
client has been told it timed out.

## Troubleshooting

**The first question times out.** The reply begins:

```
⏱ Query timed out after 45 s for agent '<agent>'.
```

The usual cause on a first call is the embedding model loading inside the
server process. Start the daemon once with `nlqueries embed-server start`
([cli-reference](cli-reference.md#embed-server)) and retry. If it keeps
happening, the query itself is slow: ask for less, such as a narrower filter or
an aggregate instead of raw rows.

**`Agent '<agent>' not found. Use list_agents() to see the agents you can use.`**
There is no knowledge base for that ID. Run `nlqueries export-kb` for the
connector ([step 5](getting-started.md#step-5--export-the-knowledge-base)).
`list_agents` returns knowledge-base file names, in which a connector ID's `:`
becomes `_` (`postgres:localhost:mydb` is `postgres_localhost_mydb`); either
form reaches the same agent.

**The server will not start over SSE.** It names what is missing. With no
authentication configured:

```
Refusing to start the MCP sse transport with no authentication: ...
```

With authentication but no grants file:

```
MCP authentication is configured but NLQ_MCP_GRANTS_FILE is not set, ...
```

**A call is refused.** A subject the grants file does not allow gets
`Not authorized to perform <action> on '<agent>'.` (or without the agent, for
the tools that name none). Grant that action to the subject; the table of
actions is in [MCP authentication](mcp-authentication.md#actions). A request
with no token, or a wrong one, is refused before any tool runs.

**The client cannot find the server, or it starts without your key.** Check the
full path in `command` and the `env` block, and look in the client's own MCP
log for the server's output.
