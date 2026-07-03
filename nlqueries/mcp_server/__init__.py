# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

from nlqueries.mcp_server.server import mcp

__all__ = ["mcp"]
