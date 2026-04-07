from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
