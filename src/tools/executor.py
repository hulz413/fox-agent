from pathlib import Path

from src.tools.policy import ToolPolicy
from src.tools.result import ToolExecutionResult
from src.tools.schemas import ToolDefinition


class ToolExecutor:
    FILE_TOOLS = {"list_files", "read_file", "write_file"}

    def __init__(self, policy: ToolPolicy | None = None) -> None:
        self.policy = policy or ToolPolicy()

    def execute(
        self, tool: ToolDefinition, args: dict[str, object]
    ) -> ToolExecutionResult:
        try:
            self._validate(tool.name, args)
            result = tool.handler(**args)
            return ToolExecutionResult(success=True, content=str(result))
        except Exception as e:
            return ToolExecutionResult(success=False, error=str(e))

    def _validate(self, tool_name: str, args: dict[str, object]) -> None:
        if tool_name not in self.FILE_TOOLS:
            return

        path = args["path"]
        if not path:
            return

        target = Path(str(path)).expanduser().resolve()
        allowed_roots = self.policy.resolve_allowed_roots()
        if not any(self._is_within_root(target, root) for root in allowed_roots):
            raise ValueError(
                f"Path {target} is not within allowed roots: {allowed_roots}"
            )

        if tool_name == "write_file" and not self.policy.allow_file_write:
            raise ValueError(f"Write file {target} is disabled by tool policy")

    def _is_within_root(self, target: Path, root: Path) -> bool:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            return False
