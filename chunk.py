from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterator


MIN_WORDS = 15
MAX_WORDS = 250

MAX_CHARS = 2000

MIN_PROSE_FRACTION = 0.5

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _enforce_ceiling(piece: str) -> list[str]:
    words = piece.split()
    if len(words) <= MAX_WORDS:
        return [piece]

    out = [words[i:i + MAX_WORDS] for i in range(0, len(words), MAX_WORDS)]
    if len(out) > 1 and len(out[-1]) < MIN_WORDS:
        out[-2].extend(out.pop())
    return [" ".join(w) for w in out]


@dataclass(slots=True)
class Chunk:
    document_id: str 
    source_id: str
    ordinal: int
    text: str
    published_at: str
    content_hash: str = ""
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)
        if not self.token_estimate:
            self.token_estimate = int(len(self.text.split()) * 1.33)


def split_text(text: str) -> list[str]:
    """Paragraphs, with over-long ones broken at sentence boundaries."""
    pieces: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para.split()) <= MAX_WORDS:
            pieces.append(para)
            continue

        # Long paragraph: accumulate sentences until the budget is spent.
        current: list[str] = []
        for sentence in _SENTENCE.split(para):
            if current and len(" ".join(current + [sentence]).split()) > MAX_WORDS:
                pieces.extend(_enforce_ceiling(" ".join(current)))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            # A single sentence can exceed the budget on its own, so the
            # ceiling has to be applied here too, not only on flush.
            pieces.extend(_enforce_ceiling(" ".join(current)))

    return [p for p in pieces if len(p.split()) >= MIN_WORDS]


def chunk_document(doc: dict) -> Iterator[Chunk]:
    """Turn one stored document record into its chunks."""
    for ordinal, piece in enumerate(split_text(doc["text"])):
        yield Chunk(
            document_id=doc["external_id"],
            source_id=doc["source_id"],
            ordinal=ordinal,
            text=piece,
            published_at=doc["published_at"],
        )
