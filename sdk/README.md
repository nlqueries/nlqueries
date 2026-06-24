# nlqueries-sdk

Official Python SDK for [NLQueries](https://github.com/nlqueries/nlqueries) — query your NLQueries agents programmatically in 5 lines of code.

```bash
pip install nlqueries-sdk
```

## Requirements

- Python 3.11+
- A running NLQueries deployment (self-hosted or cloud)
- An agent API key (`nlq_…`) generated from the NLQueries admin panel

---

## Quickstart

### 1. Basic query

```python
from nlqueries_sdk import NLQueriesClient

client = NLQueriesClient(
    base_url="https://my-nlqueries.company.com",
    api_key="nlq_abc1234567890abcdef1234567890ab",
)

result = client.query(agent_id="sales-agent", question="How many deals closed last month?")

print(result.answer)   # "There were 127 deals closed last month."
print(result.sql)      # "SELECT COUNT(*) FROM deals WHERE closed_at >= ..."
print(result.latency_ms)  # 1840
```

### 2. Session-based multi-turn chat

```python
from nlqueries_sdk import NLQueriesClient

client = NLQueriesClient(base_url="https://my-nlqueries.company.com", api_key="nlq_...")

# First turn — server creates a new session
r1 = client.query("sales-agent", "Show me deals closed last month")
session_id = r1.session_id

# Follow-up — server resolves "those" using conversation history
r2 = client.query("sales-agent", "Filter those to the EMEA region", session_id=session_id)
print(r2.answer)
```

### 3. Streaming tokens via WebSocket

```python
from nlqueries_sdk import NLQueriesClient

client = NLQueriesClient(base_url="https://my-nlqueries.company.com", api_key="nlq_...")

for token in client.query_iter(agent_id="sales-agent", question="Monthly revenue by region?"):
    print(token, end="", flush=True)

print()  # newline after stream ends
print(client.last_result.sql)  # structured result available after iteration
```

### 4. Async usage

```python
import asyncio
from nlqueries_sdk import NLQueriesClient

client = NLQueriesClient(base_url="https://my-nlqueries.company.com", api_key="nlq_...")

async def main() -> None:
    result = await client.query_async(
        agent_id="sales-agent",
        question="How many active users this week?",
    )
    print(result.answer)

asyncio.run(main())
```

### 5. Error handling

```python
from nlqueries_sdk import NLQueriesClient, AuthenticationError, PlanLimitError
import httpx

client = NLQueriesClient(base_url="https://my-nlqueries.company.com", api_key="nlq_...")

try:
    result = client.query("sales-agent", "Total revenue?")
except AuthenticationError:
    print("Invalid or expired API key — generate a new one in the admin panel.")
except PlanLimitError:
    print("Monthly query quota exceeded — upgrade your NLQueries plan.")
except httpx.HTTPStatusError as e:
    print(f"Server error: {e.response.status_code}")
```

---

## API Reference

### `NLQueriesClient`

```python
NLQueriesClient(base_url: str, api_key: str, timeout: int = 60)
```

| Method | Description |
|--------|-------------|
| `query(agent_id, question, ...)` | Synchronous query; returns `AgentQueryResult` |
| `query_async(agent_id, question, ...)` | Async version; same return type |
| `query_iter(agent_id, question, ...)` | Streaming via WebSocket; yields `str` tokens |
| `list_agents()` | Lists all agents; returns `list[dict]` |

### `AgentQueryResult`

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | Original question |
| `answer` | `str` | Natural-language answer |
| `agent_type` | `str` | `"sql"`, `"document"`, or `"hybrid"` |
| `sql` | `str \| None` | Generated SQL (if applicable) |
| `sql_result` | `dict \| None` | Query result rows (if `execute_sql=True`) |
| `citations` | `list[dict]` | Document citations (if document/hybrid) |
| `session_id` | `str \| None` | Session UUID for multi-turn chat |
| `latency_ms` | `int` | End-to-end wall-clock time in ms |
| `from_cache` | `bool` | `True` if the answer was served from the semantic cache |

### Exceptions

| Exception | HTTP status | Description |
|-----------|-------------|-------------|
| `AuthenticationError` | 401 | Invalid or inactive API key |
| `PlanLimitError` | 402 | Monthly query quota or plan feature limit exceeded |
| `httpx.HTTPStatusError` | other 4xx/5xx | Propagated from httpx |

---

## Publishing

```bash
# From the repo root:
cd core/sdk
python -m build
twine upload dist/*
```

See `enterprise/docs/internal/sprint24-launch-readiness.md` for the full release checklist.
