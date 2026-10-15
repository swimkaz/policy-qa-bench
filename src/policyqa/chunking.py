"""Split markdown policy documents into section-level chunks.

Sections are the unit of meaning in a policy document, so the chunker splits on
"## " headings first and only falls back to word windows when a section is long.
Every chunk carries its document title and section heading, so a chunk that
only says "31 days" still tells the retriever which product it is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # section_id, plus "@n" when a long section was split
    section_id: str  # "<doc_id>#<heading-slug>", the unit the golden set cites
    doc_id: str
    doc_title: str
    heading: str
    body: str

    @property
    def text(self) -> str:
        """What gets indexed and shown to the model."""
        return f"{self.doc_title} | {self.heading}\n{self.body}"


def slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    return [words[i : i + max_words] for i in range(0, len(words) - overlap, step)]


def chunk_markdown(doc_id: str, text: str, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else doc_id

    chunks: list[Chunk] = []
    # Text before the first "## " heading is front matter and is not indexed.
    for part in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = part.partition("\n")
        heading, body = heading.strip(), body.strip()
        if not body:
            continue
        section_id = f"{doc_id}#{slugify(heading)}"
        pieces = _windows(body.split(), max_words, overlap)
        for n, words in enumerate(pieces):
            chunk_id = section_id if len(pieces) == 1 else f"{section_id}@{n}"
            chunks.append(Chunk(chunk_id, section_id, doc_id, doc_title, heading, " ".join(words)))
    return chunks


def load_corpus(corpus_dir: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    paths = sorted(Path(corpus_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md files in {corpus_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks += chunk_markdown(path.stem, path.read_text(encoding="utf-8"), max_words, overlap)
    return chunks
