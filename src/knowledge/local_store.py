from src.knowledge.store import VectorStore
from src.knowledge.schemas import Chunk, RetrievedChunk


class LocalVectorStore(VectorStore):
    def __init__(self):
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for chunk, vector in zip(chunks, vectors):
            self._items[chunk.chunk_id] = (chunk, vector)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=self._cosine_similarity(query_vector, vector),
            )
            for chunk, vector in self._items.values()
        ]
        return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")
        return sum(a * b for a, b in zip(left, right))
