from abc import ABC, abstractmethod
from src.memory.schemas import MemoryRecord


class MemoryStore(ABC):
    @abstractmethod
    def get(self, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[MemoryRecord]:
        raise NotImplementedError
