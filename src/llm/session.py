from src.llm.client import LLMClient
from src.llm.schemas import Message, LLMResponse


class ChatSession:
    def __init__(self, client: LLMClient, system_prompt: str | None = None) -> None:
        self.client = client
        self.messages: list[Message] = []

        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content: str) -> None:
        self.messages.append(Message(role="system", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def chat(self, user_input: str) -> LLMResponse:
        self.add_user_message(user_input)
        response = self.client.chat(self.messages)
        self.add_assistant_message(response.content)
        return response

    def clear(self) -> None:
        self.messages = [
            message for message in self.messages if message.role == "system"
        ]
