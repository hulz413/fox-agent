from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_calls: list[ToolCall] | None = None
    raw: object | None = None
