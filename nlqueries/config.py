"""
nlqueries.config
~~~~~~~~~~~~~~~~
Loads runtime configuration from environment variables (and an optional
.env file in the working directory).

All other modules should import settings from here rather than reading
os.environ directly, so the source of truth is a single place.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the current working directory (or any parent).
# This is a no-op when the variables are already set in the environment,
# so it is safe to call unconditionally.
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
"""URL of the Qdrant vector store. Defaults to a local instance."""

QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nlqueries")
"""Name of the Qdrant collection used for query embeddings."""

QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
"""API key for a Qdrant instance running with authentication enabled.

Empty (the default) means an unauthenticated instance, which is the usual local
setup. When a deployment turns Qdrant's ``service.api_key`` on, every client here
must send it or the request is rejected with 401 — and because the vector store
is used behind graceful degradation (the semantic cache silently skips, retrieval
falls back to full-YAML injection), that failure is easy to miss.
"""

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
"""Anthropic API key for Claude-based query generation and summarisation."""


def _detect_provider() -> str:
    """Resolve the LLM provider from the environment.

    Priority: explicit LLM_PROVIDER env var > key-based auto-detection.
    If OPENAI_API_KEY is set (and LLM_PROVIDER is not), routes through
    LiteLLM so the existing litellm client handles OpenAI calls.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip()
    if explicit:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "litellm"
    return "anthropic"


def _detect_model(provider: str) -> str:
    """Resolve the LLM model from the environment.

    Priority: explicit LLM_MODEL env var > provider default.
    When auto-detected as litellm (OpenAI key present), defaults to
    'openai/gpt-4o' so LiteLLM routes to OpenAI without extra config.
    """
    explicit = os.getenv("LLM_MODEL", "").strip()
    if explicit:
        return explicit
    if provider == "litellm" and os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    return "claude-sonnet-4-5"


LLM_PROVIDER: str = _detect_provider()
"""LLM provider to use. Auto-detected from available API keys if not set explicitly."""

LLM_MODEL: str = _detect_model(LLM_PROVIDER)
"""LLM model identifier. Defaults based on detected provider if not set explicitly."""


def _detect_fast_model(provider: str) -> str:
    """Resolve the fast/cheap LLM model from the environment.

    Used for short-output tasks (intent classification, follow-up resolution)
    where a smaller model is sufficient and cheaper.
    Priority: explicit LLM_MODEL_FAST env var > provider default.
    """
    explicit = os.getenv("LLM_MODEL_FAST", "").strip()
    if explicit:
        return explicit
    if provider == "litellm" and os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    return "claude-haiku-4-5-20251001"


LLM_MODEL_FAST: str = _detect_fast_model(LLM_PROVIDER)
"""Fast/cheap LLM model for short-output auxiliary calls (classifier, resolver)."""

EXPLAIN_VALIDATION: bool = os.getenv("NLQ_EXPLAIN_VALIDATION", "false").lower() in (
    "1",
    "true",
    "yes",
)
"""When True, validate_and_repair() runs EXPLAIN on the final SQL via the caller-supplied
connector. Off by default (set NLQ_EXPLAIN_VALIDATION=true to enable)."""

# ---------------------------------------------------------------------------
# Embedding daemon
# ---------------------------------------------------------------------------
EMBED_SERVER_PORT: int = int(os.getenv("EMBED_SERVER_PORT", "8765"))
"""Port for the persistent embedding daemon (``nlqueries embed-server start``)."""

EMBED_SERVER_REQUIRED: bool = os.getenv("EMBED_SERVER_REQUIRED", "false").lower() in (
    "1",
    "true",
    "yes",
)
"""When True, a daemon failure raises instead of embedding in the calling process.

The in-process fallback is right for the CLI, where there may be no daemon at
all and loading the model is simply how the command works. It is wrong for a
server: a roughly 1 GB torch model loads into the uvicorn worker, taking several
seconds, in the middle of a request — and it stays loaded. Under load, which is
exactly when the daemon is most likely to time out, that converts a transient
queue into a permanent memory increase on every worker.

Servers should set this true. Then a daemon problem surfaces as a clean failure
naming the daemon, instead of a deployment that quietly gets slower and fatter.
"""

EMBED_CLIENT_TIMEOUT_SECONDS: float = float(os.getenv("EMBED_CLIENT_TIMEOUT_SECONDS", "2.0"))
"""How long to wait for the embedding daemon before giving up on a request.

Two seconds suited a single-threaded daemon where a queue meant a long wait for
nothing. With a concurrent daemon (EMBED_SERVER_MAX_CONCURRENCY) waiting is
usually better than the alternative, so a larger value is often the right call —
particularly with EMBED_SERVER_REQUIRED, where the alternative is a failed turn.
"""

EMBED_SERVER_MAX_CONCURRENCY: int = int(
    os.getenv("EMBED_SERVER_MAX_CONCURRENCY", str(min(4, os.cpu_count() or 1)))
)
"""How many embeddings the daemon will compute at once.

The daemon serves every process on the host and sits on the hot path of every
chat turn, so serialising it makes it the bottleneck for the whole machine. It
still needs a ceiling: each concurrent encode holds its own activations, and an
unbounded queue of them is an out-of-memory kill rather than a slowdown.

Set to 1 to serialise, which reproduces the single-threaded behaviour exactly.
"""

EMBED_SERVER_TORCH_THREADS: int = int(os.getenv("EMBED_SERVER_TORCH_THREADS", "1"))
"""Intra-op thread count for the torch backend inside the daemon.

Defaults to 1 because the daemon's workload is many small independent encodes,
not one large one. Left at torch's default, every concurrent request would spawn
threads for every core, and N requests would oversubscribe the machine N-fold —
the threads then spend their time contending rather than working. N concurrent
single-threaded encodes beat one N-threaded encode here.

Set to 0 to leave torch's own default alone.
"""

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KB_PATH: Path = Path(os.getenv("KB_PATH", str(Path.home() / ".nlqueries" / "knowledge_base")))
"""Directory where generated YAML knowledge-base files are stored."""

KB_REFRESH_INTERVAL: int = int(os.getenv("KB_REFRESH_INTERVAL", "3600"))
"""How often (seconds) the knowledge base is refreshed from the source DB."""

GLOSSARY_QUESTION_SCOPED: bool = os.getenv("NLQ_GLOSSARY_QUESTION_SCOPED", "false").lower() in (
    "1",
    "true",
    "yes",
)
"""When True (CG-2.2), glossary terms are injected **per question** — only terms
the question mentions, plus their hierarchy ancestors (context) and descendants
(candidates, depth 3) — instead of the whole glossary in the cached static block.
Off by default: the full glossary ships in the static prompt exactly as before.
Business rules are always injected in full regardless of this flag."""

# ---------------------------------------------------------------------------
# Connectors registry
# ---------------------------------------------------------------------------
CONNECTORS_FILE: Path = Path(
    os.getenv("CONNECTORS_FILE", str(Path.home() / ".nlqueries" / "connectors.yaml"))
)
"""YAML file that stores registered connector configurations."""

# ---------------------------------------------------------------------------
# Connector execution
# ---------------------------------------------------------------------------
CONNECTOR_STATEMENT_TIMEOUT_SECONDS: float = float(
    os.getenv("CONNECTOR_STATEMENT_TIMEOUT_SECONDS", "120")
)
"""Default server-side statement timeout (seconds) applied to connector queries
that don't pass an explicit ``timeout_seconds`` — so a slow query fails fast with
an error instead of hanging a request/chat turn indefinitely. Set to 0 to disable.
Currently enforced by the Postgres connector via ``SET LOCAL statement_timeout``."""

# ---------------------------------------------------------------------------
# Query Capsules
# ---------------------------------------------------------------------------
CAPSULES_DIR: Path = Path(os.getenv("CAPSULES_DIR", str(Path.home() / ".nlqueries" / "capsules")))

# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------
QUERY_HISTORY_LIMIT: int = int(os.getenv("QUERY_HISTORY_LIMIT", "500"))
"""Maximum number of queries to fetch from the database's query history view."""
"""Directory where serialised QueryCapsule JSON files are stored."""

# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
FEEDBACK_DIR: Path = Path(os.getenv("FEEDBACK_DIR", str(Path.home() / ".nlqueries" / "feedback")))
"""Directory where per-agent feedback JSONL files are stored."""

# ---------------------------------------------------------------------------
# Cache thresholds
# ---------------------------------------------------------------------------
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "torch").lower()
"""Embedding backend used by the embed-server daemon. Values: torch | onnx.
  torch — sentence-transformers / PyTorch (default, no extra deps)
  onnx  — ONNX Runtime via optimum[onnxruntime] (faster cold start, no PyTorch)
"""

SCHEMA_FORMAT: str = os.getenv("NLQ_SCHEMA_FORMAT", "compact").lower()
"""Schema format used in the static system prompt. Values: compact | verbose.
  compact  — M-Schema format (【Table】 ...) — fewer tokens, default
  verbose  — Full markdown format (### Table: ...) — backward compatible
"""

SELF_CONSISTENCY: str = os.getenv("NLQ_SELF_CONSISTENCY", "off").lower()
"""Self-consistency mode for hard SQL queries. Values: off | hard | all.
  off  — disabled (default)
  hard — run N parallel candidates only when _is_hard() returns True
  all  — always run N parallel candidates
"""

CACHE_ANSWER_THRESHOLD: float = float(os.getenv("NLQ_CACHE_ANSWER_THRESHOLD", "0.97"))
"""Cosine similarity threshold for the Tier 1 answer cache (kind=answer)."""

CACHE_TEMPLATE_THRESHOLD: float = float(os.getenv("NLQ_CACHE_TEMPLATE_THRESHOLD", "0.90"))
"""Cosine similarity threshold for the Tier 2 template cache (kind=template)."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
