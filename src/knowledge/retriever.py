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

    def retrieve(
        self, query: str, k: int = 3, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(query)
        retrieved = self.vector_store.search(query_vector, k)
        if min_score is None:
            return retrieved
        return [item for item in retrieved if item.score >= min_score]

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

    def render_debug(self, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "No knowledge chunks matched."

        lines: list[str] = []
        for position, item in enumerate(retrieved, start=1):
            preview = " ".join(item.chunk.content.split())
            if len(preview) > 180:
                preview = f"{preview[:180]}..."

            lines.append(
                f"[{position}] score={item.score:.4f} "
                f"source={item.chunk.source} "
                f"chunk={item.chunk.index} "
                f"strategy={item.chunk.metadata.get('chunk_strategy', 'unknown')}"
            )
            lines.append(f"    {preview}")

        return "\n".join(lines)
