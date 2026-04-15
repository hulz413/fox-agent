from openai import OpenAI

from src.knowledge.embeddings import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self, api_key: str, base_url: str, model: str = None, timeout: float = 60.0
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[query])
        return response.data[0].embedding
