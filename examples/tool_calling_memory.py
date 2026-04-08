from dataclasses import replace
from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


def decorate_text(text: str) -> str:
    return f"**{text}**"


# Example input cases:
# Remember that this project is for learning AI Agent development. Save it in the project namespace under key project_goal.
# Remember that I prefer concise answers. Save it in the user namespace under key response_style.
# Load the memory stored in the project namespace under key project_goal.
# What do you remember in the user namespace, under key response_style?
def main() -> None:
    setup_logging()

    config = AgentConfig.from_env()
    config = replace(
        config,
        system_prompt=(
            "You are a helpful assistant. "
            "Use save_memory to remember important user preferences and project facts. "
            "Store user preferences in the 'user' namespace and project facts in the 'project' namespace. "
            "Use load_memory with the correct namespace when the user asks what you remember."
        ),
        memory_mode="auto",
    )
    agent = Agent(config)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        match user_input.lower():
            case "exit" | "quit":
                print("Assistant: Bye~")
                break
            case _:
                response = agent.run(user_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
