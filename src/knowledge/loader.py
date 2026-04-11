from pathlib import Path


from src.knowledge.schemas import Document


class DocumentLoader:
    SUPPORTED_SUFFIXES = {".md", ".txt", ".py"}

    def load_path(self, path: str) -> list[Document]:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            raise ValueError(f"Knowledge base path does not exist: {target}")

        if target.is_file():
            return [self._load_file(target)] if self._is_supported(target) else []

        documents: list[Document] = []
        for file_path in sorted(target.rglob("*")):
            if file_path.is_file() and self._is_supported(file_path):
                documents.append(self._load_file(file_path))
        return documents

    def _load_file(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding="utf-8")
        return Document(
            source=str(file_path),
            content=content,
            metadata={"suffix": file_path.suffix},
        )

    def _is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_SUFFIXES
