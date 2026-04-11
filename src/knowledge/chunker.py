from src.knowledge.schemas import Chunk, Document


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than or equal to 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than 0")
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        content = document.content.strip()
        if not content:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0
        step = self.chunk_size - self.chunk_overlap
        content_len = len(content)

        while start < content_len:
            end = min(start + self.chunk_size, content_len)
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.source}::chunk::{index}",
                        source=document.source,
                        content=chunk_content,
                        index=index,
                        metadata=document.metadata,
                    )
                )
            if end >= content_len:
                break
            start += step
            index += 1

        return chunks
