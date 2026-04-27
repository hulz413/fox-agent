import hashlib
import math
import re

from src.llm.embedding_provider import EmbeddingProvider


class SimpleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            index = self._stable_index(token)
            vector[index] += 1.0

        norm = math.sqrt(sum([x**2 for x in vector]))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    def _tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        normalized = normalized.replace("_", " ")
        return re.findall(r"[a-z0-9]+", normalized.lower())

    def _stable_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension
