from dataclasses import replace
from src.core.logging import setup_logging
from src.agent import AgentConfig, Agent


def decorate_text(text: str) -> str:
    return f"**{text}**"


def main() -> None:
    setup_logging()

    config = AgentConfig.from_env()
    config = replace(
        config,
        system_prompt="You are a helpful assistant. Use tools when needed.",
        allow_file_write=True,
        allowed_roots=[".", "/tmp"],
    )
    agent = Agent(config)

    user_input = (
        "List files in the current directory, "
        "then write 'Hello, Fox Agent!' to .tmp/hello.txt, "
        "then read src/tools/executor.py and summarize what it does."
    )
    response = agent.run(user_input)
    print("Assistant: " + response.content)
    print()


if __name__ == "__main__":
    main()
