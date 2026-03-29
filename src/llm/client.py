from src.core.config import Config
from src.llm.schemas import LLMResponse, Message
from src.llm.deepseek_provider import DeepSeekProvider


class LLMClient:
    def __init__(self, config: Config):
        self.provider = DeepSeekProvider(config)

    def chat(self, messages: list[Message]) -> LLMResponse:
        return self.provider.chat(messages)
