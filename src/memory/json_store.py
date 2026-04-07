import json
from pathlib import Path

from src.memory.store import MemoryStore
from src.memory.schemas import MemoryRecord


class JsonMemoryStore(MemoryStore):
    def __init__(self, file_path: str = "~/.fox-agent/memory_store.json") -> None:
        self.file_path = Path(file_path).expanduser()

    def set(self, key: str, value: str, namespace: str = "default") -> None:
        memory = self._load()
        if namespace not in memory:
            memory[namespace] = {}
        memory[namespace][key] = value
        self._save(memory)

    def get(self, key: str, namespace: str = "default") -> str:
        memory = self._load()
        bucket = memory.get(namespace, {})
        if key not in bucket:
            raise ValueError(f"Memory not found for namespace={namespace}, key={key}")
        return bucket[key]

    def delete(self, key: str, namespace: str = "default") -> None:
        memory = self._load()
        bucket = memory.get(namespace, {})
        if key not in bucket:
            raise ValueError(f"Memory not found for namespace={namespace}, key={key}")
        del bucket[key]
        if not bucket:
            del memory[namespace]
        self._save(memory)

    def list(self, namespace: str | None = None) -> list[MemoryRecord]:
        memory = self._load()
        records: list[MemoryRecord] = []

        namespaces = [namespace] if namespace else sorted(memory.keys())
        for namespace in namespaces:
            bucket = memory.get(namespace, {})
            records.extend(
                MemoryRecord(namespace=namespace, key=key, value=value)
                for key, value in sorted(bucket.items(), key=lambda item: item[0])
            )

        return records

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.file_path.exists():
            return {}
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Memory store must be a JSON object: {self.file_path}")

        memory: dict[str, dict[str, str]] = {}
        for namespace, bucket in data.items():
            if not isinstance(bucket, dict):
                raise ValueError(
                    f"Memory store must be a JSON object: {self.file_path}"
                )
            memory[str(namespace)] = {
                str(key): str(value) for key, value in bucket.items()
            }
        return memory

    def _save(self, memory: dict[str, dict[str, str]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
