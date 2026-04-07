from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryRecord:
    namespace: Literal["default", "user", "project"]
    key: str
    value: str
