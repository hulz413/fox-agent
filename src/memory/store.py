from abc import ABC, abstractmethod
from src.memory.schemas import MemoryRecord


class MemoryStore(ABC):
    @abstractmethod
    def get(self, key: str, namespace: str = "default") -> str:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: str, namespace: str = "default") -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str, namespace: str = "default") -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self, namespace: str | None = None) -> list[MemoryRecord]:
        raise NotImplementedError
