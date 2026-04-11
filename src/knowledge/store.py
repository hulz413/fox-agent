from abc import ABC, abstractmethod

from src.knowledge.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
