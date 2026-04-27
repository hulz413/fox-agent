import re

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
            [self._build_embedding_text(chunk) for chunk in chunks]
        )
        self.vector_store.upsert(chunks, vectors)

    def retrieve(
        self, query: str, k: int = 3, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        candidate_k = k * 4
        query_vector = self.embedding_provider.embed_query(query)
        candidates = self.vector_store.search(query_vector, candidate_k)
        retrieved = self._rerank(query, candidates)[:k]
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

    def _build_embedding_text(self, chunk: Chunk) -> str:
        metadata_lines = [
            f"{key}: {value}" for key, value in sorted(chunk.metadata.items()) if value
        ]

        return "\n".join(
            [
                f"source: {chunk.source}",
                *metadata_lines,
                "",
                chunk.content,
            ]
        )

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return candidates

        reranked: list[RetrievedChunk] = []
        query_symbols = self._extract_query_symbols(query)
        for item in candidates:
            embedding_text = self._build_embedding_text(item.chunk)
            chunk_tokens = set(self._tokenize(embedding_text))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            symbol_boost = self._symbol_boost(query_symbols, embedding_text)
            boosted_score = item.score + 0.25 * overlap + symbol_boost
            reranked.append(RetrievedChunk(chunk=item.chunk, score=boosted_score))

        return sorted(reranked, key=lambda x: x.score, reverse=True)

    def _tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        normalized = normalized.replace("_", " ")
        return re.findall(r"[a-z0-9]+", normalized.lower())
        stop_words = {
            "a",
            "an",
            "are",
            "does",
            "for",
            "how",
            "in",
            "into",
            "is",
            "the",
            "to",
            "where",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", normalized.lower())
            if token not in stop_words
        ]

    def _extract_query_symbols(self, query: str) -> list[str]:
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        return [token for token in candidates if self._looks_like_code_symbol(token)]

    def _looks_like_code_symbol(self, token: str) -> bool:
        if "_" in token:
            return True
        if token.isupper() and len(token) > 1:
            return True
        if any(char.isupper() for char in token[1:]):
            return True
        return False

    def _symbol_boost(self, query_symbols: list[str], embedding_text: str) -> float:
        if not query_symbols:
            return 0.0
        normalized_text = embedding_text.lower()
        boost = 0.0
        for symbol in query_symbols:
            if len(symbol) < 3:
                continue
            normalized_symbol = symbol.lower()
            if normalized_symbol in normalized_text:
                boost += 0.35
        return min(boost, 0.7)
