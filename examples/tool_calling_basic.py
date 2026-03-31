from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry


def main() -> None:
    config = Config.from_env()
    client = LLMClient(config)
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools():
        tool_registry.register(tool)

    session = ChatSession(
        client,
        tool_registry,
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks for the current time, use the available tool."
        ),
    )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye~")
            break

        response = session.chat(user_input)

        print("Assistant: " + response.content)
        print()


if __name__ == "__main__":
    main()
