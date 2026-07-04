"""Small dependency-free helpers for semantic search over text chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Protocol


Vector = Sequence[float]


@dataclass(frozen=True)
class SearchResult:
    """One vector search hit."""

    index: int
    score: float
    sequence: str


class TextEmbedder(Protocol):
    """Minimal text embedding interface used by the wiki tools."""

    def embed(self, text: str) -> Vector: ...

    def embed_batch(self, texts: Sequence[str]) -> Sequence[Vector]: ...


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into fixed-width, optionally overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    step = chunk_size - chunk_overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def _cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"dimension mismatch: query dim {len(left)} != vectors dim {len(right)}"
        )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right)
    ) / (left_norm * right_norm)


def vector_build_and_search(
    query: str,
    text: str,
    embedder: TextEmbedder,
    *,
    top_k: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> list[SearchResult]:
    """Chunk and embed text, then return its closest chunks to a query."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks or top_k == 0:
        return []

    embeddings = embedder.embed_batch(chunks)
    if len(embeddings) != len(chunks):
        raise ValueError("embedding vectors must align with chunks")
    query_embedding = embedder.embed(query)
    ranked = sorted(
        (
            SearchResult(
                index=index,
                score=_cosine_similarity(query_embedding, embedding),
                sequence=chunk,
            )
            for index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=True)
            )
        ),
        key=lambda result: result.score,
        reverse=True,
    )
    return ranked[: min(top_k, len(ranked))]
