from dataclasses import replace
from src.core.logging import setup_logging
from src.tools.schemas import ToolDefinition
from src.agent import AgentConfig, Agent


def decorate_text(text: str) -> str:
    return f"**{text}**"


def main() -> None:
    setup_logging()

    decorate_text_tool = ToolDefinition(
        name="decorate_text",
        description="Decorate the text.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
        handler=decorate_text,
    )
    config = AgentConfig.from_env()
    config = replace(
        config, system_prompt="You are a helpful assistant. Use tools when needed."
    )
    agent = Agent(config)
    agent.register_tool(decorate_text_tool)
    response = agent.run("Get current time, and then decorate it.")
    print("Assistant: " + response.content)
    print()


if __name__ == "__main__":
    main()
