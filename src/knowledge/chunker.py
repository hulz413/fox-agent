from __future__ import annotations

import ast
import re
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
        suffix = document.metadata.get("suffix", "").lower()
        match suffix:
            case ".md":
                return self._chunk_markdown(document)
            case ".py":
                return self._chunk_python(document)
            case ".txt":
                return self._chunk_paragraphs(document)
            case _:
                return self._chunk_fixed_window(document)

    def _chunk_markdown(self, document: Document) -> list[Chunk]:
        content = document.content.strip()
        if not content:
            return []

        matches = list(re.finditer(r"(?m)^#{1,6}\s+.+$", content))
        if not matches:
            return self._chunk_paragraphs(document)

        chunks: list[Chunk] = []
        index = 0

        if matches[0].start() > 0:
            preface = content[: matches[0].start()].strip()
            if preface:
                index = self._append_chunk_slices(
                    chunks,
                    document,
                    index,
                    preface,
                    {**document.metadata, "chunk_strategy": "markdown_preface"},
                )

        for position, match in enumerate(matches):
            start = match.start()
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(content)
            )
            section = content[start:end].strip()
            if not section:
                continue

            index = self._append_chunk_slices(
                chunks,
                document,
                index,
                section,
                {
                    **document.metadata,
                    "chunk_strategy": "markdown_section",
                    "header": match.group(0).strip(),
                    "start_line": str(content[:start].count("\n") + 1),
                    "end_line": str(content[:end].count("\n") + 1),
                },
            )

        return chunks

    def _chunk_python(self, document: Document) -> list[Chunk]:
        try:
            tree = ast.parse(document.content)
        except SyntaxError:
            return self._chunk_fixed_window(document)

        lines = document.content.splitlines()
        chunks: list[Chunk] = []
        index = 0

        for node in tree.body:
            if not isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)
            symbol_text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not symbol_text:
                continue

            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
            index = self._append_chunk_slices(
                chunks,
                document,
                index,
                symbol_text,
                {
                    **document.metadata,
                    "chunk_strategy": "python_symbol",
                    "symbol_name": node.name,
                    "symbol_type": symbol_type,
                    "start_line": str(start_line),
                    "end_line": str(end_line),
                },
            )

        return chunks or self._chunk_fixed_window(document)

    def _chunk_paragraphs(self, document: Document) -> list[Chunk]:
        content = document.content.strip()
        if not content:
            return []

        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n+", content) if part.strip()
        ]
        if not paragraphs:
            return self._chunk_fixed_window(document)

        chunks: list[Chunk] = []
        index = 0
        buffer: list[str] = []

        for paragraph in paragraphs:
            candidate = "\n\n".join([*buffer, paragraph])
            if buffer and len(candidate) > self.chunk_size:
                index = self._append_chunk_slices(
                    chunks,
                    document,
                    index,
                    "\n\n".join(buffer),
                    {**document.metadata, "chunk_strategy": "paragraph"},
                )
                buffer = [paragraph]
                continue

            buffer.append(paragraph)

        if buffer:
            self._append_chunk_slices(
                chunks,
                document,
                index,
                "\n\n".join(buffer),
                {**document.metadata, "chunk_strategy": "paragraph"},
            )

        return chunks

    def _chunk_fixed_window(self, document: Document) -> list[Chunk]:
        content = document.content.strip()
        if not content:
            return []

        chunks: list[Chunk] = []
        self._append_chunk_slices(
            chunks,
            document,
            0,
            content,
            {**document.metadata, "chunk_strategy": "fixed_window"},
        )
        return chunks

    def _append_chunk_slices(
        self,
        chunks: list[Chunk],
        document: Document,
        index: int,
        text: str,
        metadata: dict[str, str],
    ) -> int:
        slices = self._slice_text(text)
        for slice_index, chunk_content in enumerate(slices):
            chunk_metadata = dict(metadata)
            if len(slices) > 1:
                chunk_metadata["slice"] = str(slice_index)

            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}::chunk::{index}",
                    source=document.source,
                    content=chunk_content,
                    index=index,
                    metadata=chunk_metadata,
                )
            )
            index += 1

        return index

    def _slice_text(self, text: str) -> list[str]:
        content = text.strip()
        if not content:
            return []

        if len(content) <= self.chunk_size:
            return [content]

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)
            if end >= len(content):
                break
            start += step

        return chunks
