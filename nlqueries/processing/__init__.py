# nlqueries-core — OSS (BSL 1.1)

from nlqueries.processing.parameterizer import (
    Placeholder,
    QueryCapsule,
    parameterize_cluster,
    parameterize_clusters,
)
from nlqueries.processing.query_clusterer import (
    QueryCluster,
    cluster_queries,
)
from nlqueries.processing.query_filter import (
    NormalizedQuery,
    filter_and_deduplicate,
)

# Aliases consumed by the `nlqueries process-history` CLI command.
filter_queries = filter_and_deduplicate
parameterize = parameterize_clusters

__all__ = [
    "NormalizedQuery",
    "Placeholder",
    "QueryCapsule",
    "QueryCluster",
    "cluster_queries",
    "filter_and_deduplicate",
    "filter_queries",
    "parameterize",
    "parameterize_cluster",
    "parameterize_clusters",
]
