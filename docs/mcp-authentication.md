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

## A networked transport will not start without a verifier

`--transport sse` and `--transport streamable-http` refuse to start unless one
of the two below is configured. The refusal names what to set.

If you are upgrading and need the previous behaviour immediately, set
`NLQ_ALLOW_UNAUTHENTICATED_MCP=1`. It logs a warning on every start and grants
every caller everything, which is what the server did before. It is deliberately
*not* the same switch as `NLQ_ALLOW_INSECURE_BIND`: that one says you know the
port is reachable, this one says you know anyone who reaches it may run SQL
against your database.

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

These apply only where callers are told apart, which means an authenticated
transport. stdio has no limits — the caller owns the process, and rationing them
against themselves would achieve nothing — and neither does a transport running
with `NLQ_ALLOW_UNAUTHENTICATED_MCP`, because every request on it is the same
anonymous caller. Limiting that one subject would not ration anybody; it would
put a single budget across the whole deployment that one client could exhaust
and thereby starve the rest. Rationing arrives with authentication, along with
everything else that depends on knowing who is calling.

## What gets recorded

Every decision, allowed and denied, goes to the `nlqueries.audit` logger:

```json
{"at": "2026-08-28T09:00:00+00:00", "principal": "alice@example.com",
 "source": "oidc", "action": "query:execute", "agent_id": "sales",
 "decision": "deny", "reason": "no grant covers this action on this agent"}
```

Denials log at `WARNING` and allows at `INFO`, so the two can be routed
differently without parsing messages. Neither carries the token nor the token's
claims — these records travel into support bundles.

The refusal returned to the caller names the action and the agent and never the
reason. A caller learning which agents exist from the wording of a denial is one
of the disclosures this exists to prevent.

## Enterprise

nlqueries-enterprise supplies its own authorizer over its tenant tables and
existing RBAC, so a deployment there does not write a grants file. The interface
is the same; only the source of the answer differs.
