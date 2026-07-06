"""
Tests for nlqueries.document_connectors.chunker — the dependency-free
recursive-character text splitter used by the document connectors.
"""

from __future__ import annotations

import pytest
from nlqueries.document_connectors.chunker import (
    RecursiveCharacterTextSplitter,
    split_text,
)


def test_shorter_than_chunk_size_returns_single_chunk() -> None:
    assert split_text("short text", chunk_size=800) == ["short text"]


def test_exact_chunk_size_length_returns_single_chunk() -> None:
    text = "a" * 800
    chunks = split_text(text, chunk_size=800)
    assert chunks == [text]


def test_empty_string_returns_empty_list() -> None:
    assert split_text("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert split_text("   \n\n   ") == []


def test_paragraph_boundary_split() -> None:
    paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_long_single_word_hard_splits_at_chunk_size() -> None:
    text = "a" * 5000
    chunks = split_text(text, chunk_size=800, chunk_overlap=0)
    assert all(len(c) <= 800 for c in chunks)
    assert sum(len(c) for c in chunks) >= len(text)


def test_sentence_boundary_fallback() -> None:
    sentences = [f"This is sentence number {i}" for i in range(50)]
    text = ". ".join(sentences) + "."
    chunks = split_text(text, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_word_boundary_fallback() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # No word should be split in half.
    for chunk in chunks:
        for token in chunk.split():
            assert token.startswith("word") or token == ""


def test_overlap_correctness() -> None:
    paragraphs = [f"Paragraph {i}: " + ("x" * 50) for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    for prev_chunk, next_chunk in zip(chunks, chunks[1:], strict=False):
        overlap_len = min(20, len(prev_chunk))
        tail = prev_chunk[-overlap_len:]
        assert next_chunk.startswith(tail[-len(next_chunk) :]) or tail in next_chunk


@pytest.mark.parametrize(
    "text,chunk_size",
    [
        ("short", 800),
        ("a" * 5000, 800),
        ("Para one.\n\nPara two.\n\nPara three." * 50, 300),
        ("word " * 1000, 150),
    ],
)
def test_all_chunks_within_chunk_size_invariant(text: str, chunk_size: int) -> None:
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=50)
    assert all(len(c) <= chunk_size for c in chunks)


def test_never_returns_empty_strings() -> None:
    text = "a\n\n\n\n\nb"
    chunks = split_text(text, chunk_size=800)
    assert all(c for c in chunks)


def test_chunk_overlap_gte_chunk_size_does_not_crash() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 20
    chunks_equal = split_text(text, chunk_size=100, chunk_overlap=100)
    chunks_greater = split_text(text, chunk_size=100, chunk_overlap=500)
    assert all(len(c) <= 100 for c in chunks_equal)
    assert all(len(c) <= 100 for c in chunks_greater)


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("text", chunk_size=-1)


def test_negative_chunk_overlap_raises() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_overlap=-1)


def test_deterministic() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 20
    first = split_text(text, chunk_size=150, chunk_overlap=30)
    second = split_text(text, chunk_size=150, chunk_overlap=30)
    assert first == second


def test_recursive_character_text_splitter_class_delegates() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 20
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    assert splitter.split_text(text) == split_text(text, chunk_size=150, chunk_overlap=30)
