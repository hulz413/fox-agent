from src.llm.schemas import LLMResponse, Message
from src.llm.chat_provider import ChatProvider
from src.tools.schemas import ToolDefinition


class LLMClient:
    def __init__(self, provider: ChatProvider):
        self.provider = provider

    def chat(
        self, messages: list[Message], tools: list[ToolDefinition] | None = None
    ) -> LLMResponse:
        return self.provider.chat(messages=messages, tools=tools)
