# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

from nlqueries.warning_filters import quiet_google_python_version_notice

# Before the server module, whose connectors import Google's libraries.
quiet_google_python_version_notice()

from nlqueries.mcp_server.server import mcp  # noqa: E402

__all__ = ["mcp"]
