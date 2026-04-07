from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryRecord:
    namespace: str
    key: str
    value: str
