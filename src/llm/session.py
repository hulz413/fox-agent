import logging
from src.llm.client import LLMClient
from src.llm.schemas import Message, LLMResponse, ToolCall
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(
        self,
        client: LLMClient,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry or ToolRegistry()
        self.messages: list[Message] = []
        self.max_steps = max_steps

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
        logger.info(f"User input: {user_input}")
        self.add_user_message(user_input)
        return self._run_loop()

    def _run_loop(self) -> LLMResponse:
        for step in range(1, self.max_steps + 1):
            logger.info(f"Step {step}/{self.max_steps} started")
            response = self.client.chat(
                messages=self.messages, tools=self.tool_registry.list()
            )

            if response.usage:
                logger.info(
                    "Model usage: "
                    f"prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

            self.add_assistant_message(
                content=response.content, tool_calls=response.tool_calls
            )

            if not response.tool_calls:
                logger.info(f"Assistant final response: {response.content}")
                return response

            if response.content:
                logger.info(f"Assistant response: {response.content}")

            for tool_call in response.tool_calls:
                logger.info(
                    f"Calling tool: {tool_call.name} with arguments={tool_call.arguments}"
                )
                tool = self.tool_registry.get(tool_call.name)

                try:
                    result = tool.handler(**tool_call.arguments)
                    logger.info(f"Tool calling {tool_call.name} result: {result}")
                except Exception as e:
                    logger.exception(f"Tool calling {tool_call.name} failed")
                    result = f"Tool call {tool_call.name} failed: {str(e)}"
                self.add_tool_message(result, tool_call.id)

        raise ValueError("Max steps reached without a response :(")

    def clear(self) -> None:
        self.messages = [
            message for message in self.messages if message.role == "system"
        ]
        logger.info("Session cleared, system message preserved")
