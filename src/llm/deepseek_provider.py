from openai import OpenAI
from src.core.config import Config
from src.llm.schemas import LLMResponse, Message


class DeepSeekProvider:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(self, messages: list[Message]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=response.usage,
            raw=response,
        )
