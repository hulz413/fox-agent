import json
from openai import OpenAI

from src.core.config import Config
from src.llm.schemas import LLMResponse, Message, ToolCall, Usage
from src.tools.schemas import ToolDefinition
from src.llm.chat_provider import ChatProvider


class OpenAIChatProvider(ChatProvider):
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.chat_api_key,
            base_url=config.chat_base_url,
            timeout=config.chat_timeout,
        )

    def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.config.chat_model,
            "temperature": self.config.chat_temperature,
            "messages": [self._message_to_dict(message) for message in messages],
        }
        if tools:
            payload["tools"] = [self._tool_to_dict(tool) for tool in tools]

        response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )

    def _message_to_dict(self, message: Message) -> dict[str, object]:
        payload: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }

        match message.role:
            case "assistant":
                if message.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    tool_call.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
            case "tool":
                if message.tool_call_id:
                    payload["tool_call_id"] = message.tool_call_id

        return payload

    def _tool_to_dict(self, tool: ToolDefinition) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
