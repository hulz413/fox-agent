from src.llm.client import LLMClient
from src.llm.schemas import Message, LLMResponse, ToolCall
from src.tools.registry import ToolRegistry


class ChatSession:
    def __init__(
        self,
        client: LLMClient,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry or ToolRegistry()
        self.messages: list[Message] = []

        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content: str) -> None:
        self.messages.append(Message(role="system", content=content))

    def add_assistant_message(
        self, content: str, tool_calls: list[ToolCall] | None = None
    ) -> None:
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.messages.append(
            Message(role="tool", content=content, tool_call_id=tool_call_id)
        )

    def get_history(self) -> list[Message]:
        return list(self.messages)

    def chat(self, user_input: str) -> LLMResponse:
        self.add_user_message(user_input)
        response = self.client.chat(
            messages=self.messages, tools=self.tool_registry.list()
        )
        self.add_assistant_message(
            content=response.content, tool_calls=response.tool_calls
        )

        if not response.tool_calls:
            return response

        for tool_call in response.tool_calls:
            tool = self.tool_registry.get(tool_call.name)
            if tool is None:
                raise ValueError(f"Tool registry is missing tool: {tool_call.name}")

            result = tool.handler(**tool_call.arguments)
            self.add_tool_message(result, tool_call.id)

        final_response = self.client.chat(
            messages=self.messages, tools=self.tool_registry.list()
        )
        self.add_assistant_message(final_response.content)

        return final_response

    def clear(self) -> None:
        self.messages = [
            message for message in self.messages if message.role == "system"
        ]
