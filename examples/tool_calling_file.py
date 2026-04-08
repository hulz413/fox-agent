from src.core.logging import setup_logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.tools.policy import ToolPolicy


def decorate_text(text: str) -> str:
    return f"**{text}**"


def main() -> None:
    setup_logging()

    config = Config.from_env()
    client = LLMClient(config)
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools():
        tool_registry.register(tool)

    tool_policy = ToolPolicy(
        allowed_roots=[".", "/tmp"],
        allow_file_write=True,
    )

    session = ChatSession(
        client,
        tool_registry,
        tool_policy=tool_policy,
        system_prompt=("You are a helpful assistant. Use tools when needed."),
    )

    user_input = (
        "List files in the current directory, "
        "then write 'Hello, Fox Agent!' to .tmp/hello.txt, "
        "then read src/tools/executor.py and summarize what it does."
    )
    response = session.chat(user_input)
    print("Assistant: " + response.content)
    print()


if __name__ == "__main__":
    main()
