import logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession


# Example input cases:
# What is an AI agent?
# Explain tool calling with a simple example.
# Compare memory and session history in one short paragraph.
# What is the difference between planning and direct execution?
# Give me a simple roadmap for learning AI Agent development.
# clear
# history
# exit
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
    )

    config = Config.from_env()
    client = LLMClient(config)

    session = ChatSession(
        client=client,
        system_prompt="You are a helpful assistant.",
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
