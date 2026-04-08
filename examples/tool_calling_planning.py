from dataclasses import replace
from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


# Example input cases:
# List files in src/tools, then read builtins.py and summarize the available tools.
# Read src/llm/session.py, explain how the agent loop works, then summarize it in 3 bullet points.
# Remember that this project is for learning AI Agent development, then tell me what was stored.
# Get the current time, save it to a file at /tmp/current.txt, then read the file back and summarize the result.
# Read requirements.txt and explain what dependencies are used in this project.
def main() -> None:
    setup_logging()

    config = AgentConfig.from_env()
    config = replace(
        config,
        system_prompt=(
            "You are a helpful assistant executing a structured plan. "
            "Complete only the current step. "
            "Use tools when needed. "
            "Do not jump ahead to future steps. "
            "Be concise."
        ),
        plan_mode="auto",
        max_steps=20,
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
