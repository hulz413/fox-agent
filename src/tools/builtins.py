from datetime import datetime
from src.tools.schemas import ToolDefinition


def get_current_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_builtin_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_current_time",
            description="Get the current local time.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=get_current_time,
        ),
    ]
