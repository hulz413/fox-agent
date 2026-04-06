import json
from datetime import datetime
from pathlib import Path
from src.tools.schemas import ToolDefinition

MEMORY_STORE_FILE = Path("~/.fox-agent/memory_store.json").expanduser()


def get_current_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


def list_files(path: str = ".") -> str:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise ValueError(f"Path does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {target}")

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
        raise ValueError(f"Path does not exist: {target}")
    if not target.is_file():
        raise ValueError(f"Path is not a file: {target}")

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

    return f"File written successfully: {target}"


def _load_memory_store() -> dict[str, str]:
    if not MEMORY_STORE_FILE.exists():
        return {}
    if not MEMORY_STORE_FILE.is_file():
        raise ValueError(f"Path is not a file: {MEMORY_STORE_FILE}")
    with MEMORY_STORE_FILE.open("r", encoding="utf-8") as f:
        memory: dict[str, str] = {}
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Memory store must be a JSON object: {MEMORY_STORE_FILE}")
        for key, value in data.items():
            memory[str(key)] = str(value)
        return memory


def _save_memory_store(memory: dict[str, str]) -> None:
    MEMORY_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_STORE_FILE.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def save_memory(key: str, value: str) -> str:
    memory = _load_memory_store()
    memory[key] = value
    _save_memory_store(memory)
    return f"Memory saved successfully: {key}"


def load_memory(key: str) -> str:
    memory = _load_memory_store()
    if key not in memory:
        raise ValueError(f"Memory not found for key {key}")
    return memory[key]


def list_memory_keys() -> str:
    memory = _load_memory_store()
    return "\n".join(memory.keys())


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
        ToolDefinition(
            name="save_memory",
            description="Save a memory value under a given key in persistent local storage.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to store.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Memory value to store.",
                    },
                },
                "required": ["key", "value"],
            },
            handler=save_memory,
        ),
        ToolDefinition(
            name="load_memory",
            description="Load a memory value from a given key in persistent local storage. Use this tool to recall information stored earlier.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to load.",
                    },
                },
                "required": ["key"],
            },
            handler=load_memory,
        ),
        ToolDefinition(
            name="list_memory_keys",
            description="List all memory keys stored in persistent local storage.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=list_memory_keys,
        ),
    ]
