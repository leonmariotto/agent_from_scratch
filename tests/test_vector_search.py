import pytest

from agent_from_scratch.vector_search import chunk_text, vector_build_and_search


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [
            float("target" in text),
            float("other" in text),
        ]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_chunk_text_applies_overlap() -> None:
    assert chunk_text("abcdefgh", chunk_size=4, chunk_overlap=1) == [
        "abcd",
        "defg",
        "gh",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (4, -1), (4, 4)],
)
def test_chunk_text_rejects_invalid_sizes(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size, chunk_overlap)


def test_vector_build_and_search_ranks_matching_chunks() -> None:
    results = vector_build_and_search(
        "target",
        "other.....target",
        FakeEmbedder(),
        top_k=1,
        chunk_size=10,
    )

    assert len(results) == 1
    assert results[0].sequence == "target"
    assert results[0].score == pytest.approx(1.0)


def test_vector_build_and_search_validates_inputs() -> None:
    with pytest.raises(ValueError, match="top_k"):
        vector_build_and_search("query", "text", FakeEmbedder(), top_k=-1)
