from abc import ABC, abstractmethod
import math


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError


class SimpleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            index = hash(token) % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum([x**2 for x in vector]))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]
