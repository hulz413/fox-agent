from openai import OpenAI

from src.llm.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = None,
        timeout: float = 60.0,
        batch_size: int = 100,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[query])
        return response.data[0].embedding
