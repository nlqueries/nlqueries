"""MarkdownConnector and TextConnector: .md/.markdown and .txt documents."""

from __future__ import annotations

from pathlib import Path

import pytest
from nlqueries.document_connectors import (
    DOCUMENT_CONNECTOR_REGISTRY,
    MarkdownConnector,
    TextConnector,
)
from nlqueries.document_connectors.text import markdown_sections


def _write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# ---------------------------------------------------------------------------
# Markdown sectioning
# ---------------------------------------------------------------------------


def test_markdown_splits_at_level_one_and_two_headings() -> None:
    text = "# Refunds\nWithin 30 days.\n\n## Exceptions\nSale items.\n"
    assert markdown_sections(text) == [
        ("Refunds", "Within 30 days."),
        ("Exceptions", "Sale items."),
    ]


def test_deeper_headings_stay_in_their_section() -> None:
    """Splitting on every level leaves chunks too small to hold an answer."""
    text = "## Shipping\nIntro.\n### Domestic\n3 days.\n"
    assert markdown_sections(text) == [("Shipping", "Intro.\n### Domestic\n3 days.")]


def test_a_hash_line_inside_a_code_fence_is_not_a_heading() -> None:
    """A shell or Python comment in a sample is not a section break."""
    text = "# Setup\nRun:\n```bash\n# install first\npip install x\n```\nDone.\n"
    sections = markdown_sections(text)
    assert [h for h, _ in sections] == ["Setup"]
    assert "# install first" in sections[0][1]


def test_a_fence_closes_only_with_at_least_as_many_markers() -> None:
    """```` is not closed by ``` -- a fence shown inside a fence stays code."""
    text = "# Doc\n````\n```\n# still code\n```\n````\n## After\nText.\n"
    assert [h for h, _ in markdown_sections(text)] == ["Doc", "After"]


def test_tilde_fences_are_fences_too() -> None:
    text = "# Doc\n~~~\n# code\n~~~\n"
    assert [h for h, _ in markdown_sections(text)] == ["Doc"]


def test_text_before_the_first_heading_is_kept() -> None:
    text = "Preamble.\n# First\nBody.\n"
    assert markdown_sections(text) == [("untitled", "Preamble."), ("First", "Body.")]


def test_an_empty_section_is_dropped() -> None:
    text = "# Empty\n# Full\nBody.\n"
    assert markdown_sections(text) == [("Full", "Body.")]


def test_four_space_indent_is_code_not_a_heading() -> None:
    text = "# Real\nBody.\n    # indented code\n"
    assert [h for h, _ in markdown_sections(text)] == ["Real"]


def test_closing_hashes_are_not_part_of_the_heading() -> None:
    assert markdown_sections("## Returns ##\nBody.\n") == [("Returns", "Body.")]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_markdown_ingest_carries_the_heading_in_metadata(tmp_path: Path) -> None:
    path = _write(tmp_path, "policy.md", "# Refunds\nWithin 30 days.\n")
    chunks = MarkdownConnector().ingest(path, "doc-1")

    assert len(chunks) == 1
    assert chunks[0].text == "Within 30 days."
    assert chunks[0].source_name == "policy.md"
    assert chunks[0].metadata["connector"] == "markdown"
    assert chunks[0].metadata["section_heading"] == "Refunds"


def test_a_long_section_is_split(tmp_path: Path) -> None:
    body = " ".join(f"word{i}" for i in range(600))
    chunks = MarkdownConnector().ingest(_write(tmp_path, "long.md", f"# Long\n{body}\n"), "doc")
    assert len(chunks) > 1
    assert all(c.metadata["section_heading"] == "Long" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.md", "# A\nOne.\n## B\nTwo.\n")
    first = [c.chunk_id for c in MarkdownConnector().ingest(path, "doc")]
    second = [c.chunk_id for c in MarkdownConnector().ingest(path, "doc")]
    assert first == second
    assert len(set(first)) == 2


def test_plain_text_is_one_section(tmp_path: Path) -> None:
    path = _write(tmp_path, "notes.txt", "# not a heading in a .txt\nLine two.\n")
    chunks = TextConnector().ingest(path, "doc")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("# not a heading")
    assert chunks[0].metadata["connector"] == "text"


def test_an_empty_file_produces_no_chunks(tmp_path: Path) -> None:
    assert TextConnector().ingest(_write(tmp_path, "empty.txt", "  \n\n"), "doc") == []
    assert MarkdownConnector().ingest(_write(tmp_path, "empty.md", ""), "doc") == []


def test_a_utf8_byte_order_mark_is_dropped(tmp_path: Path) -> None:
    path = _write(tmp_path, "bom.md", "# Title\nBody.\n", encoding="utf-8-sig")
    chunks = MarkdownConnector().ingest(path, "doc")
    assert chunks[0].metadata["section_heading"] == "Title", "the BOM hid the heading"


def test_text_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    """A guessed encoding ingests mojibake that retrieves as if it were content."""
    path = tmp_path / "latin1.txt"
    path.write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        TextConnector().ingest(path, "doc")


# ---------------------------------------------------------------------------
# Registry and suffixes
# ---------------------------------------------------------------------------


def test_the_registry_holds_both() -> None:
    assert DOCUMENT_CONNECTOR_REGISTRY["markdown"] is MarkdownConnector
    assert DOCUMENT_CONNECTOR_REGISTRY["text"] is TextConnector


def test_supports_matches_only_its_own_suffixes() -> None:
    md, txt = MarkdownConnector(), TextConnector()
    assert md.supports("a.md") and md.supports("a.MARKDOWN")
    assert not md.supports("a.txt")
    assert txt.supports("a.TXT")
    assert not txt.supports("a.md")
    assert not md.supports("a.pdf") and not txt.supports("a.docx")


@pytest.mark.parametrize(
    ("name", "connector"),
    [("notes.md", "markdown"), ("guide.markdown", "markdown"), ("readme.txt", "text")],
)
def test_doc_ingest_reaches_the_new_connectors(tmp_path: Path, name: str, connector: str) -> None:
    """Drive the CLI itself, not the registry.

    `doc-ingest` takes the first registered connector whose `supports()` is
    true, and Notion's and Confluence's accept any path. Registered after them,
    these connectors were unreachable: `notes.md` resolved to Notion, whose
    constructor needs a token, and the command died with a traceback. A test
    that checked the registry with those two filtered out passed throughout.
    """
    from click.testing import CliRunner
    from nlqueries.cli.main import cli

    path = _write(tmp_path, name, "# Title\nSome body text.\n")
    result = CliRunner().invoke(cli, ["doc-ingest", "doc-1", str(path)])

    assert result.exit_code == 0, result.output
    assert f"connector={connector}" in result.output.replace(" ", "")
    assert "Chunks produced" in result.output


# ---------------------------------------------------------------------------
# Setext headings
# ---------------------------------------------------------------------------


def test_setext_underlines_are_headings() -> None:
    text = "Refunds\n=======\nWithin 30 days.\n\nExceptions\n----------\nSale items.\n"
    assert markdown_sections(text) == [
        ("Refunds", "Within 30 days."),
        ("Exceptions", "Sale items."),
    ]


def test_a_multi_line_paragraph_becomes_one_setext_heading() -> None:
    """In CommonMark the whole paragraph above the underline is the heading."""
    assert markdown_sections("Long\ntitle\n===\nBody.\n") == [("Long title", "Body.")]


def test_dashes_after_a_blank_line_are_a_thematic_break() -> None:
    text = "# Doc\nIntro.\n\n---\n\nMore.\n"
    assert markdown_sections(text) == [("Doc", "Intro.\n\n---\n\nMore.")]


def test_dashes_under_a_list_item_are_not_a_heading() -> None:
    text = "# Doc\n- item\n---\nAfter.\n"
    assert [h for h, _ in markdown_sections(text)] == ["Doc"]


def test_an_underline_inside_a_fence_is_code() -> None:
    text = "# Doc\n```\nTitle\n=====\n```\n"
    assert [h for h, _ in markdown_sections(text)] == ["Doc"]


def test_text_before_a_setext_heading_is_kept_in_the_previous_section() -> None:
    text = "# Intro\nFirst.\n\nNext part\n---------\nSecond.\n"
    assert markdown_sections(text) == [("Intro", "First."), ("Next part", "Second.")]


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_a_file_over_the_document_limit_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can interrupt the read once it starts, so the size is checked
    first and the refusal takes the shape the other connectors' limits do."""
    from nlqueries.document_connectors import DocumentTooComplexError
    from nlqueries.document_connectors import text as text_module

    monkeypatch.setattr(text_module, "MAX_EXPANDED_BYTES", 10)
    path = _write(tmp_path, "big.txt", "x" * 11)

    read = []
    monkeypatch.setattr(text_module, "_read_utf8", lambda p: read.append(p) or "")
    with pytest.raises(DocumentTooComplexError, match="big.txt"):
        TextConnector().ingest(path, "doc")
    assert read == [], "the file was read before its size was checked"


def test_a_file_at_the_limit_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nlqueries.document_connectors import text as text_module

    monkeypatch.setattr(text_module, "MAX_EXPANDED_BYTES", 10)
    assert len(TextConnector().ingest(_write(tmp_path, "ok.txt", "x" * 10), "doc")) == 1
