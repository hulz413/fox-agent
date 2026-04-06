import logging

from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.schemas import Message
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry


def create_plan(client: LLMClient, user_input: str) -> str:
    messages = [
        Message(
            role="system",
            content=(
                "You are a planning assistant for an AI agent. "
                "Create a short execution plan for the user's request. "
                "Keep the plan 3-5 concise and numbered steps. "
                "Do not solve the task, do not call any tools."
            ),
        ),
        Message(
            role="user",
            content=user_input,
        ),
    ]

    response = client.chat(messages)
    return response.content


# Example input cases:
# List files in the current directory, then read requirements.txt and summarize the dependencies.
# Read src/llm/session.py and explain how the agent loop works.
# List files in src/tools, then read src/tools/builtins.py and summarize the available tools.
# Remember that this project goal is learning AI Agent development, then tell me what you stored.
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

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        match user_input.lower():
            case "exit" | "quit":
                print("Assistant: Bye~")
                break
            case _:
                plan = create_plan(client, user_input)
                print("=== Plan ===")
                print(plan)
                print()

                session = ChatSession(
                    client,
                    tool_registry,
                    system_prompt=(
                        "You are a helpful assistant. "
                        "First follow the provided plan, then complete the user's request. "
                        "Use tools when needed."
                    ),
                    max_steps=20,
                )
                execution_input = (
                    f"User request: {user_input}\n\n"
                    f"Plan:\n{plan}\n\n"
                    "Execute the plan step by step and give the final answer."
                )

                response = session.chat(execution_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
