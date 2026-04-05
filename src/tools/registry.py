import logging
from src.tools.schemas import ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        logger.info(f"Registering tool: {tool.name}")
        if tool.name in self.tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self.tools[name]
        except KeyError as e:
            raise ValueError(f"Unknown tool: {name}") from e

    def list(self) -> list[ToolDefinition]:
        return list(self.tools.values())
