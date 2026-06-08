"""
nlqueries.embeddings.embedder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentence-transformer embedding module (OSS, local, free).

Uses ``all-MiniLM-L6-v2`` (384 dimensions).  The model is loaded lazily
on first use so that importing this module has no cost when embeddings are
not needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the shared ``SentenceTransformer`` instance, loading it on first call."""
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # deferred — heavy import

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Embed a single string and return a list of 384 floats.

    Args:
        text: The string to embed.

    Returns:
        A ``list[float]`` of length 384 (L2-normalised).
    """
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    result: list[float] = vector.tolist()
    return result


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings and return a list of 384-float vectors.

    Args:
        texts: Strings to embed.

    Returns:
        ``list[list[float]]`` — one 384-float vector per input string,
        in the same order, L2-normalised.
    """
    model = _get_model()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
    result: list[list[float]] = [v.tolist() for v in vectors]
    return result
