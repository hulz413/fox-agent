from src.memory.schemas import MemoryRecord
from src.memory.store import MemoryStore


class MemoryRetriever:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def retrieve(
        self, namespaces: list[str] | None = None, max_records: int = 10
    ) -> list[MemoryRecord]:
        if not namespaces:
            namespaces = ["user", "project"]

        return self.memory_store.list(namespaces)[:max_records]

    def render_context(self, records: list[MemoryRecord]) -> str:
        if not records:
            return ""

        lines = ["Relevant memory:"]
        for record in records:
            lines.append(f"[{record.namespace}] {record.key} = {record.value}")
        return "\n".join(lines)

    def build_context(
        self, namespaces: list[str] | None = None, max_records: int = 10
    ) -> str:
        records = self.retrieve(namespaces=namespaces, max_records=max_records)
        return self.render_context(records)
