from abc import ABC, abstractmethod


class ChatProvider(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        raise NotImplementedError
