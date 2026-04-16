from src.llm.embedding_provider import EmbeddingProvider
from src.knowledge.schemas import Chunk, RetrievedChunk
from src.knowledge.store import VectorStore


class KnowledgeRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        vectors = self.embedding_provider.embed_texts(
            [chunk.content for chunk in chunks]
        )
        self.vector_store.upsert(chunks, vectors)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(query)
        return self.vector_store.search(query_vector, k)

    def render(self, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return ""

        lines = ["Relevant knowledge base context:"]
        lines.extend(
            [
                (
                    f"Source: {item.chunk.source} (chunk {item.chunk.index})\n"
                    f"{item.chunk.content}\n\n"
                )
                for item in retrieved
            ]
        )
        return "\n".join(lines).strip()
