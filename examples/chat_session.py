from dataclasses import replace
from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


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
    setup_logging()

    config = AgentConfig.from_env()
    config = replace(config, system_prompt="You are a helpful assistant.")
    agent = Agent(config)

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
                for message in agent.get_history():
                    print(f"[{message.role}] {message.content}")
                print()
                continue
            case "clear":
                agent.clear()
                print("History cleared!")
                continue
            case _:
                response = agent.run(user_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
