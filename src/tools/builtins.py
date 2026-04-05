from datetime import datetime
from pathlib import Path
from src.tools.schemas import ToolDefinition


def get_current_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


def list_files(path: str = ".") -> str:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise ValueError(f"Path {target} does not exist.")
    if not target.is_dir():
        raise ValueError(f"Path {target} is not a directory.")

    entries = sorted(target.iterdir(), key=lambda item: item.name.lower())
    if not entries:
        return f"Directory {target} is empty."

    lines: list[str] = []
    for entry in entries:
        lines.append(f"[{'dir' if entry.is_dir() else 'file'}]\t{entry.name}")

    return "\n".join(lines)


def read_file(path: str, max_chars: int = 1024) -> str:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise ValueError(f"Path {target} does not exist.")
    if not target.is_file():
        raise ValueError(f"Path {target} is not a file.")

    content = target.read_text(encoding="utf-8")
    if len(content) > max_chars:
        content = (
            content[:max_chars]
            + "\n\n"
            + f"[File is too long, truncated to {max_chars} characters]"
        )

    return content


def write_file(path: str, content: str) -> str:
    target = Path(path).expanduser().resolve()
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"File {target} written successfully."


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
        ToolDefinition(
            name="list_files",
            description="List files and directories in the given directory path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Use '.' for the current directory.",
                    },
                },
                "required": [],
            },
            handler=list_files,
        ),
        ToolDefinition(
            name="read_file",
            description="Read the full content of a file at the given path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file at the given path. Overwrites if it already exists.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=write_file,
        ),
    ]
