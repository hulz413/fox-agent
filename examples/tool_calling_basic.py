from dataclasses import replace
from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


def main() -> None:
    setup_logging()

    config = AgentConfig.from_env()
    config = replace(
        config, system_prompt="You are a helpful assistant. Use tools when needed."
    )
    agent = Agent(config)
    response = agent.run("What time is it?")
    print("Assistant: " + response.content)
    print()


if __name__ == "__main__":
    main()
