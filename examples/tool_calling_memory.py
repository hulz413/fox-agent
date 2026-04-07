from src.core.logging import setup_logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.memory.json_sotre import JsonMemoryStore


def decorate_text(text: str) -> str:
    return f"**{text}**"


# Example input cases:
# Remember that this project is for learning AI Agent development. Save it under key project_goal.
# Remember that I prefer concise answers. Save it under key response_style.
# Load the memory stored under key project_goal.
# What do you remember under key response_style?
def main() -> None:
    setup_logging()

    config = Config.from_env()
    client = LLMClient(config)
    memory_store = JsonMemoryStore()
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools(memory_store):
        tool_registry.register(tool)

    session = ChatSession(
        client,
        tool_registry,
        memory_store=memory_store,
        system_prompt=(
            "You are a helpful assistant. "
            "Use save_memory to remember important user preferences and project facts. "
            "Store user preferences in the 'user' namespace and project facts in the 'project' namespace. "
            "Use load_memory with the correct namespace when the user asks what you remember."
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
            case _:
                response = session.chat(user_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
