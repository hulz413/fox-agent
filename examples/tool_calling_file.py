import logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry


def decorate_text(text: str) -> str:
    return f"**{text}**"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
    )

    config = Config.from_env()
    client = LLMClient(config)
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools():
        tool_registry.register(tool)

    session = ChatSession(
        client,
        tool_registry,
        system_prompt=("You are a helpful assistant. Use tools when needed."),
        max_steps=10,
    )

    user_input = (
        "List files in the current directory, "
        "then write 'Hello, Fox Agent!' to /tmp/hello.txt', "
    )
    response = session.chat(user_input)
    print("Assistant: " + response.content)
    print()

    "then read src/tools/builtins.py and summarize what tools are available."


if __name__ == "__main__":
    main()
