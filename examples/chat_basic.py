import logging
from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.schemas import Message


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
    )

    config = Config.from_env()
    client = LLMClient(config)
    messages = [
        Message(role="system", content="You are a helpful AI assistant."),
        Message(role="user", content="Please introduce AI Agent in one sentence."),
    ]

    response = client.chat(messages)

    print("=== Assistant ===")
    print(response.content)
    print()

    print("=== Finish Reason ===")
    print(response.finish_reason)
    print()

    print("=== Usage ===")
    print(response.usage)
    print()

    print("=== Raw Response ===")
    print(response.raw)
    print()


if __name__ == "__main__":
    main()
