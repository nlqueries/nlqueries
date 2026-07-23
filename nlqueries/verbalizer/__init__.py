# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.verbalizer — deterministic SQL-to-English paraphrase (ACE-1.1).

``verbalize(sql, dialect, vocab)`` renders validated SQL into one sentence of
controlled English, template-driven over the sqlglot AST — no LLM, so it cannot
hallucinate: it describes what will actually run. Coverage v1 is a single SELECT
with joins, WHERE, GROUP BY, HAVING, ORDER BY/LIMIT, and the common aggregates;
anything else degrades gracefully (the unhandled fragment is quoted verbatim and
``Paraphrase.complete`` is ``False``). It never raises.
"""

from __future__ import annotations

from nlqueries.verbalizer.ast_walk import analyze
from nlqueries.verbalizer.templates import Paraphrase, render
from nlqueries.verbalizer.vocab import Vocab, build_vocab, humanize

__all__ = ["Paraphrase", "Vocab", "build_vocab", "humanize", "verbalize"]


def verbalize(sql: str, dialect: str = "postgres", vocab: Vocab | None = None) -> Paraphrase:
    """Render *sql* into a :class:`Paraphrase`. Never raises.

    ``vocab`` supplies short labels for tables/columns; when omitted, identifiers
    are humanized (``customer_id`` → "customer id"), so a paraphrase is always
    readable even without a knowledge base.
    """
    try:
        return render(analyze(sql, dialect), vocab)
    except Exception:  # noqa: BLE001 — verbalization is best-effort, never fatal
        return Paraphrase(text="", complete=False, unhandled=[sql])
