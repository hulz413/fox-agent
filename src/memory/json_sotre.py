import json
from pathlib import Path

from src.memory.store import MemoryStore
from src.memory.schemas import MemoryRecord


class JsonMemoryStore(MemoryStore):
    def __init__(self, file_path: str = "~/.fox-agent/memory_store.json") -> None:
        self.file_path = Path(file_path).expanduser()

    def set(self, key: str, value: str) -> None:
        memory = self._load()
        memory[key] = value
        self._save(memory)

    def get(self, key: str) -> str:
        memory = self._load()
        if key not in memory:
            raise ValueError(f"Memory not found for key {key}")
        return memory[key]

    def delete(self, key: str) -> None:
        memory = self._load()
        if key in memory:
            del memory[key]
        self._save(memory)

    def list(self) -> list[MemoryRecord]:
        memory = self._load()
        return [MemoryRecord(key=key, value=value) for key, value in memory.items()]

    def _load(self) -> dict[str, str]:
        if not self.file_path.exists():
            return {}
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Memory store must be a JSON object: {self.file_path}")

        memory: dict[str, str] = {}
        for key, value in data.items():
            memory[str(key)] = str(value)
        return memory

    def _save(self, memory: dict[str, str]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
