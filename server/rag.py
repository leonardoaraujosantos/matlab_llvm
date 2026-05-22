"""Dependency-free retrieval over the compiler's own docs (plan Phase 6).

A small TF-IDF + cosine index over ``<source_context_root>/docs/**/*.md``
(the roadmaps). No native deps, deterministic, and good enough for the
"which roadmap chunk answers this question" task — swap in a real embedding
backend later behind the same ``query`` interface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from config import settings

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Chunk:
    doc_id: str   # stable id, e.g. "docs/optim_roadmap.md#3"
    source: str   # path relative to the corpus root
    title: str    # nearest heading (or filename)
    text: str


class RagIndex:
    """BM25 ranking — saturates term frequency and rewards rare terms, so a
    rare high-signal word (``fmincon``) dominates common ones in a verbose
    natural-language query."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: list[Chunk] = []
        self._k1 = k1
        self._b = b
        self._df: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk, freq)]
        self._len: list[int] = []
        self._n = 0
        self._avgdl = 1.0

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._df = {}
        self._postings = {}
        self._len = []
        for i, ch in enumerate(chunks):
            counts: dict[str, int] = {}
            toks = _tokenize(f"{ch.title} {ch.text}")
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self._len.append(len(toks) or 1)
            for tok, c in counts.items():
                self._df[tok] = self._df.get(tok, 0) + 1
                self._postings.setdefault(tok, []).append((i, c))
        self._n = len(chunks)
        self._avgdl = (sum(self._len) / self._n) if self._n else 1.0

    def query(self, text: str, top_k: int = 4) -> list[tuple[float, Chunk]]:
        if not self._n:
            return []
        scores: dict[int, float] = {}
        for term in set(_tokenize(text)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for i, freq in postings:
                dl = self._len[i]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                scores[i] = scores.get(i, 0.0) + idf * (freq * (self._k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(score, self.chunks[i]) for i, score in ranked]


def chunk_markdown(path: Path, rel: str, max_chars: int) -> list[Chunk]:
    """Split a markdown file into per-heading chunks."""
    chunks: list[Chunk] = []
    title = path.stem
    buf: list[str] = []
    idx = 0

    def flush() -> None:
        nonlocal idx, buf
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(f"{rel}#{idx}", rel, title, body[:max_chars]))
            idx += 1
        buf = []

    for line in path.read_text(errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            flush()
            title = line.lstrip("# ").strip() or title
        buf.append(line)
    flush()
    return chunks


def build_corpus(root: Path, max_chars: int) -> list[Chunk]:
    docs = root / "docs"
    if not docs.is_dir():
        return []
    out: list[Chunk] = []
    for md in sorted(docs.rglob("*.md")):
        rel = str(md.relative_to(root))
        out.extend(chunk_markdown(md, rel, max_chars))
    return out


# Process-wide index, populated at app startup (see main.lifespan).
INDEX = RagIndex()


def build_default_index() -> int:
    chunks = build_corpus(Path(settings.source_context_root), settings.rag_max_chunk_chars)
    INDEX.build(chunks)
    return len(chunks)
