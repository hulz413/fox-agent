from src.core.logging import setup_logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolDefinition


def decorate_text(text: str) -> str:
    return f"**{text}**"


def main() -> None:
    setup_logging()

    config = Config.from_env()
    client = LLMClient(config)
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools():
        tool_registry.register(tool)

    decorate_text_tool = ToolDefinition(
        name="decorate_text",
        description="Decorate the text.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
        handler=decorate_text,
    )
    tool_registry.register(decorate_text_tool)

    session = ChatSession(
        client,
        tool_registry,
        system_prompt=("You are a helpful assistant. Use tools when needed."),
        max_steps=5,
    )

    user_input = "Get current time, and then decorate it."
    response = session.chat(user_input)

    print("Assistant: " + response.content)
    print()


if __name__ == "__main__":
    main()
