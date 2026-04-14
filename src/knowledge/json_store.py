from pathlib import Path
import json

from src.knowledge.store import VectorStore
from src.knowledge.schemas import Chunk, RetrievedChunk


class JsonVectorStore(VectorStore):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path).expanduser()
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

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "content": chunk.content,
                "index": chunk.index,
                "metadata": chunk.metadata,
                "vector": vector,
            }
            for chunk, vector in self._items.values()
        ]

        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.file_path.exists():
            return

        with self.file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            self._items = {
                item["chunk_id"]: (
                    Chunk(
                        chunk_id=item["chunk_id"],
                        source=item["source"],
                        content=item["content"],
                        index=item["index"],
                        metadata=item["metadata"],
                    ),
                    item["vector"],
                )
                for item in payload
            }

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")
        return sum(a * b for a, b in zip(left, right))
