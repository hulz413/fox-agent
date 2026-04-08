from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecutionResult:
    success: bool
    content: str = ""
    error: str | None = None

    def to_message(self, tool_name: str) -> str:
        if self.success:
            return self.content
        else:
            return f"Tool call {tool_name} failed.\nError: {self.error}"
