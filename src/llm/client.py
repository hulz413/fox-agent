from src.core.config import Config
from src.llm.schemas import LLMResponse, Message
from src.llm.openai_chat_provider import OpenAIChatProvider
from src.tools.schemas import ToolDefinition


class LLMClient:
    def __init__(self, config: Config):
        self.provider = OpenAIChatProvider(config)

    def chat(
        self, messages: list[Message], tools: list[ToolDefinition] | None = None
    ) -> LLMResponse:
        return self.provider.chat(messages=messages, tools=tools)
