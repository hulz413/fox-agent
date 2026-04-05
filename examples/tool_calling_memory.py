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
        system_prompt=(
            "You are a helpful assistant. Use tools when needed. "
            "When the user explicitly asks you to remember, save or store information, "
            "use the save_memory tool. "
            "When the user explicitly asks what you remember or asks for previously stored information, "
            "use the load_memory tool. "
            "Choose short and stable memory keys, such as project_goal, preferred_language, etc. "
            "Do not use memory tools when the request does not involve remembering or recalling information."
        ),
        max_steps=10,
    )

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        match user_input.lower():
            case "exit" | "quit":
                print("Assistant: Bye~")
                break
            case "history":
                print("=== History ===")
                for message in session.get_history():
                    print(f"[{message.role}] {message.content}")
                print()
                continue
            case "clear":
                session.clear()
                print("History cleared!")
                continue
            case _:
                response = session.chat(user_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
