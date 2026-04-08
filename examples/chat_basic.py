from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


def main():
    setup_logging()

    config = AgentConfig.from_env()
    agent = Agent(config)
    response = agent.run("Please introduce AI Agent in one sentence.")

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
