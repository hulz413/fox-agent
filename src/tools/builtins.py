from datetime import datetime
from pathlib import Path
from src.tools.schemas import ToolDefinition
from src.memory.json_sotre import JsonMemoryStore
from src.memory.store import MemoryStore


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


def build_memroy_tools(
    memory_store: JsonMemoryStore | None = None,
) -> list[ToolDefinition]:
    store = memory_store or JsonMemoryStore()

    def save_memory(key: str, value: str, namespace: str = "default") -> str:
        store.set(key, value, namespace)
        return f"Memory saved successfully: {key}"

    def load_memory(key: str, namespace: str = "default") -> str:
        return store.get(key, namespace)

    def list_memory_keys() -> str:
        return "\n".join(store.list().keys())

    return [
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
                    "namespace": {
                        "type": "string",
                        "description": "Memory namespace to store under. Defaults to 'default'.",
                        "default": "default",
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
                    "namespace": {
                        "type": "string",
                        "description": "Memory namespace to load from. Defaults to 'default'.",
                        "default": "default",
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


def build_builtin_tools(
    memory_store: MemoryStore | None = None,
) -> list[ToolDefinition]:
    core_tools = [
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

    return core_tools + build_memroy_tools(memory_store)
