# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Redirect home-dir paths to a container-friendly location.
# These can all be overridden at runtime via env vars or a mounted volume.
ENV KB_PATH=/data/nlqueries/knowledge_base \
    CONNECTORS_FILE=/data/nlqueries/connectors.yaml \
    CAPSULES_DIR=/data/nlqueries/capsules \
    FEEDBACK_DIR=/data/nlqueries/feedback

# A non-root account that owns only what it needs to write. Root in a
# container is not a host compromise on its own, but it removes a layer for
# free, and every other image in this project already does it.
RUN useradd --system --create-home --uid 10001 nlqueries \
 && mkdir -p /data/nlqueries \
 && chown -R nlqueries:nlqueries /data/nlqueries

# ── dependency layer (re-runs only when pyproject.toml changes) ───────────────
# Stub the package so hatchling resolves all deps without the full source tree.
COPY pyproject.toml LICENSE README.md ./
RUN mkdir -p nlqueries && touch nlqueries/__init__.py \
 && pip install --no-cache-dir -e . \
 && rm -rf nlqueries

# ── application source ────────────────────────────────────────────────────────
COPY nlqueries/ ./nlqueries/

# The image exposes a port, so it serves one. `main()` defaults to stdio, which
# is right for Claude Desktop and useless in a detached container.
#
# Binding 0.0.0.0 here is the container's own interface, not the host's — what
# makes it safe is the host publishing it on loopback, which is a decision only
# the operator can make. So the image does NOT set NLQ_ALLOW_INSECURE_BIND: it
# refuses to start with a message explaining the exposure, and compose sets the
# variable deliberately. An image that carried its own bypass would mean every
# `docker run` silently opted in.
ENV MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080

EXPOSE 8080

# A TCP connect, not an HTTP GET. `health` is an MCP tool, not a route — the
# old probe curled /health and would have 404'd on a healthy server. There is no
# cheap HTTP endpoint to hit here, and asking the SSE path would open a stream
# that never closes, so "is anything accepting connections on the port" is the
# honest question. Python, because it is already in the image.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket,os,sys; s=socket.create_connection(('127.0.0.1', int(os.getenv('MCP_PORT','8080'))), 3); s.close()" || exit 1

USER nlqueries

CMD ["python", "-m", "nlqueries.mcp_server"]
