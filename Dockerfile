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

RUN mkdir -p /data/nlqueries

# ── dependency layer (re-runs only when pyproject.toml changes) ───────────────
# Stub the package so hatchling resolves all deps without the full source tree.
COPY pyproject.toml .
RUN mkdir -p nlqueries && touch nlqueries/__init__.py \
 && pip install --no-cache-dir -e . \
 && rm -rf nlqueries

# ── application source ────────────────────────────────────────────────────────
COPY nlqueries/ ./nlqueries/

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "-m", "nlqueries.mcp_server"]
