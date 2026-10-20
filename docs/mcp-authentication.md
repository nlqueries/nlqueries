# Authenticating the MCP server

The MCP server exposes nine tools. One of them, `query`, generates SQL and runs
it against a configured database; others return your schema, your connector
hosts, and the questions people have asked. Until a caller is identified, there
is nothing to authorise: the SQL policy constrains *what* may be run and never
*who* may run it.

This page covers turning that on. If you use the server over stdio — Claude
Desktop, or `nlqueries mcp-server start` with no `--sse` — none of it applies
and nothing has changed.

## stdio needs no configuration

The client launches the process, talks to it over its own pipes, and the caller
already has whatever the process has: the knowledge base, the connector file,
the stored credentials. Requiring a token there would be theatre. The calls are
still authorised and audited, against a principal that is the process owner.

That holds only for calls that arrive over stdio. If a server set up for stdio
is served over HTTP anyway (for example by mounting the module's `mcp` object in
an ASGI app), a call that arrived as an HTTP request with no token is anonymous,
not the process owner, and the local profile refuses it.

## A networked transport will not start without a verifier

A networked transport refuses to start unless one of the two below is
configured: `nlqueries mcp-server start --sse`, or `MCP_TRANSPORT=sse` or
`MCP_TRANSPORT=streamable-http` for the Docker image and
`python -m nlqueries.mcp_server`. The refusal names what to set.

If you are upgrading and need the previous behaviour immediately, set
`NLQ_ALLOW_UNAUTHENTICATED_MCP=1`. It logs a warning on every start and grants
every caller everything, which is what the server did before. It is deliberately
*not* the same switch as `NLQ_ALLOW_INSECURE_BIND`: that one says you know the
port is reachable, this one says you know anyone who reaches it may run SQL
against your database.

Authentication does not lift the bind check. The Docker image and
`python -m nlqueries.mcp_server` take the bind address from `MCP_HOST` (default
`127.0.0.1`) and the port from `MCP_PORT` (default `8000`); `mcp-server start
--sse` takes `--host` and `--port` with the same defaults. A wildcard host
(`0.0.0.0`, `::` or empty) is refused on a networked transport even with a
verifier configured, unless `NLQ_ALLOW_INSECURE_BIND=1` is set. Every other
spelling of an unspecified address counts as a wildcard too: `0`, `0.0`, `0x0`,
`::0`, `0:0:0:0:0:0:0:0`, `[::]`, `::ffff:0.0.0.0` and so on, as the socket layer
reads them. The image sets
`MCP_HOST=0.0.0.0` and `MCP_PORT=8080` but not `NLQ_ALLOW_INSECURE_BIND`, so a
plain `docker run` of it refuses to start until you set that variable; the
boundary is then the address you publish the port on (for example
`-p 127.0.0.1:8080:8080`). The shipped `docker-compose.yml` sets it and publishes the port on `127.0.0.1`
by default (`BIND_ADDR`).

### Host and Origin checks

On a loopback bind (`127.0.0.1`, `localhost`, `::1`) the MCP SDK already accepts
only loopback names in the `Host` header. On any other bind the server checks
them itself, so a page in a browser cannot reach it through a DNS name that
happens to resolve to it. It accepts:

- as `Host`: `localhost`, `127.0.0.1` and `[::1]`, with any port or none;
- as `Host`: the host of `NLQ_MCP_RESOURCE_URL`, with no port or with its
  port, which is the one the URL names or else the scheme's default (443 for
  `https`, 80 for `http`). If the URL names a port other than the default, the
  host is accepted on any port. So `https://mcp.example.com` accepts
  `mcp.example.com` and `mcp.example.com:443` but not `mcp.example.com:8443`;
- as `Origin`, when a client sends one: the origin of `NLQ_MCP_RESOURCE_URL`
  in the same spellings as its host above: with no port, with its port, and
  with any port if the URL names a non-default one. So `https://mcp.example.com`
  and `https://mcp.example.com:443` both accept `Origin: https://mcp.example.com`
  (what a browser sends) and `https://mcp.example.com:443`. Also
  `http://localhost`, `http://127.0.0.1` and `http://[::1]`, with any port or
  none. The loopback origins match the loopback hosts above, so a client let in
  at `127.0.0.1:8080` is not refused for sending `Origin: http://127.0.0.1:8080`.
  With no resource URL, only the loopback origins are accepted.

Any other `Host` gets `421 Misdirected Request`; any other `Origin` gets `403`.
Set `NLQ_MCP_RESOURCE_URL` to the URL clients actually use. The shipped compose
file sets it to `http://localhost:8080`, and clients can reach it there or at
`127.0.0.1:8080`. A server running with `NLQ_ALLOW_UNAUTHENTICATED_MCP` usually
has no resource URL, so it answers only to the loopback names; put a proxy in
front that forwards one of those, or set `NLQ_MCP_RESOURCE_URL` to the address
clients use. A resource URL whose port is not a number is ignored, with the same
result.

### Request size

A request body over 4 MiB gets `413` on both networked transports. The SDK
applies that limit to streamable HTTP; the server applies the same one to SSE.

### Option 1 — an identity provider

```bash
export NLQ_MCP_OIDC_DISCOVERY_URL=https://idp.example.com/.well-known/openid-configuration
export NLQ_MCP_OIDC_CLIENT_ID=nlqueries-mcp
export NLQ_MCP_RESOURCE_URL=https://mcp.internal.example.com
```

Tokens are verified against the provider's JWKS: signature, expiry, audience
(`aud` must equal the client id) and issuer. A provider whose discovery document
omits `issuer` is refused rather than accepted with the issuer check disabled,
and a token carrying no `sub` is refused rather than becoming an identity of
empty string.

The shipped `docker-compose.yml` requires `NLQ_MCP_STATIC_TOKEN` for its
quickstart. To use an identity provider with it instead:

- remove the `NLQ_MCP_STATIC_TOKEN` line from the compose file's
  `environment:` block, **and** remove or empty `NLQ_MCP_STATIC_TOKEN` in
  `.env` — the compose file also loads `.env` into the container, so a token
  left there still counts, and the server refuses both options at once;
- mount your own grants file naming your users' subjects and set
  `NLQ_MCP_GRANTS_FILE` to its path in `.env`. The file the compose stack
  writes grants only `NLQ_MCP_STATIC_SUBJECT` (`operator` by default), so
  without your own every authenticated call is denied.

### Option 2 — a pre-shared token

For an install with no identity provider:

```bash
export NLQ_MCP_STATIC_TOKEN="$(openssl rand -hex 32)"
export NLQ_MCP_STATIC_SUBJECT=operator            # optional; defaults to "static-token"
export NLQ_MCP_RESOURCE_URL=https://mcp.internal.example.com
```

Or `NLQ_MCP_STATIC_TOKEN_FILE=/run/secrets/mcp_token` to read it from a mounted
file rather than the environment.

The token is compared in constant time and must be at least 32 characters. It
authenticates and authorises nothing on its own: the subject it maps to gets
exactly what the grants file gives that subject.

Configure one option or the other. Setting both is refused — two ways in is two
things to get wrong.

## Grants

Once authentication is on, `NLQ_MCP_GRANTS_FILE` is required. Without it every
authenticated call would be denied, so the server refuses to start rather than
serving something that looks working and answers nothing.

```yaml
# Which subjects may do what, on which agents.
grants:
  # An analyst who may ask questions of one agent and read its schema.
  - subject: "alice@example.com"
    agents: ["sales"]
    actions: ["query:execute", "schema:read", "agents:list"]

  # A service account that reads cache statistics across everything.
  - subject: "monitoring"
    agents: ["*"]
    actions: ["cache:stats", "health:read"]

  # An operator with everything, on everything.
  - subject: "operator"
    agents: ["*"]
    actions: ["*"]
```

`subject` is what the identity provider put in `sub`, or `NLQ_MCP_STATIC_SUBJECT`
for a pre-shared token. Several grants may name the same subject; they combine.

A subject with no grant is denied. There is no deny list: a policy with both
grants and denials needs a precedence rule, and precedence rules are where
authorisation bugs live.

`agents` and `actions` must be **lists**. A bare string is refused, because YAML
scalars are iterable and `agents: "prod-*"` read as a sequence yields its
characters — one of which is `*`, which would grant every agent. An action that
is not one of the names below is refused too: a misspelt action grants nothing,
and you would have no way to tell that from a grant that is working.

### Actions

| Action | Tool | Names an agent |
|---|---|---|
| `query:execute` | `query` | yes |
| `schema:read` | `get_agent_schema` | yes |
| `history:read` | `get_query_history` | yes |
| `feedback:submit` | `submit_feedback` | yes |
| `cache:invalidate` | `invalidate_cache` | yes |
| `cache:stats` | `get_cache_stats` | yes |
| `agents:list` | `list_agents` | no |
| `connectors:list` | `list_connectors` | no |
| `health:read` | `health` | no |

The three that name no agent are authorised without one; the rest are authorised
against the agent in the call, and a call whose agent cannot be determined is
refused.

## Limits

Authorisation says whether a caller may run `query` on an agent. It says nothing
about how often, and each call is an LLM charge, a query against your database,
and a slot in a process that will otherwise start as many as it is asked to.

Two limits apply per principal on the networked transports, with defaults:

| Variable | Default | What it bounds |
| --- | --- | --- |
| `NLQ_MCP_RATE_LIMIT_PER_MINUTE` | 60 | Calls a principal may make in a minute |
| `NLQ_MCP_MAX_CONCURRENT` | 8 | Calls a principal may have in flight at once |

Set either to `0` to disable it. A value that is not a number, or is negative,
falls back to the default with a warning rather than to no limit — a typo in a
number should not switch a control off.

The concurrency limit is the one that matters for a slow tool: `query` is
allowed forty-five seconds, so a caller who ignores the rate limit's refusals
could otherwise hold open as many slow calls as they can start.

The rate limit is a fixed window, so a burst of up to twice the limit can cross
a boundary. Both counters live in the process, so a deployment running several
servers behind a load balancer gets the limit multiplied by that number.

These are per principal, so what they ration depends on how many principals
there are.

With an identity provider each caller has their own `sub`, and the numbers above
are per person.

**With a pre-shared token they are not.** Every accepted token resolves to the
same subject, so one allowance covers the whole install: a small team sharing a
token shares 60 calls a minute and 8 in flight between them, and one client in a
loop will exhaust it and see the rest refused. Size both numbers for the
deployment rather than for one person, or move to an identity provider if you
need to tell callers apart.

stdio has no rate or concurrency limits — the caller owns the process, and rationing them against
themselves would achieve nothing — and neither does a transport running with
`NLQ_ALLOW_UNAUTHENTICATED_MCP`, where every request is the same anonymous
caller. There, a limit would not ration anybody: it would put one budget across
a deployment that had none before, which is not what a switch asking for the
previous behaviour should hand you.

### Input size

Every tool refuses an over-long argument before it does anything, on every
transport, stdio included:

| Variable | Default | What it bounds |
| --- | --- | --- |
| `NLQ_MCP_MAX_QUESTION_CHARS` | 4000 | `question`, in `query` and `submit_feedback` |
| `NLQ_MCP_MAX_SQL_CHARS` | 8000 | `generated_sql` and `corrected_sql`, in `submit_feedback` |

An over-limit call gets a short refusal naming the argument and the limit; no
model is called, no feedback is stored, and nothing reaches the database (the
authorisation decision for the call is still audited). The
values count characters, not bytes. A value that is not a positive whole number
falls back to the default with a warning; there is no setting that removes the
limit, so raise it instead.

## What gets recorded

Every decision, allowed and denied, goes to the `nlqueries.audit` logger. The
message is `authorization denied` or `authorization allowed`; the record itself
is attached to the log record as `extra={"audit": payload}`, a dict of this
shape:

```json
{"at": "2026-08-28T09:00:00+00:00", "principal": "alice@example.com",
 "source": "oidc", "action": "query:execute", "agent_id": "sales",
 "decision": "deny", "reason": "no grant covers this action on this agent"}
```

Denials log at `WARNING` and allows at `INFO`, so the two can be routed
differently without parsing messages. A default formatter prints only the
message; to write the payload out, attach a handler to `nlqueries.audit` whose
formatter reads the record's `audit` attribute. Neither carries the token nor
the token's claims — these records travel into support bundles.

The refusal returned to the caller names the action and the agent and never the
reason. A caller learning which agents exist from the wording of a denial is one
of the disclosures this exists to prevent.

For the same reason, the other replies that could name agents follow
`agents:list`:

- `health` lists agent IDs only to a caller who is also granted `agents:list`.
  Anyone else, such as a monitor granted only `health:read`, gets the number of
  agents. Over stdio the caller has every grant and sees the IDs.
- A call naming an agent that does not exist (`query`, `get_agent_schema`) says
  so and names no other agent. `list_agents` is the way to find them.

## Enterprise

nlqueries-enterprise supplies its own authorizer over its tenant tables and
existing RBAC, so a deployment there does not write a grants file. The interface
is the same; only the source of the answer differs.
