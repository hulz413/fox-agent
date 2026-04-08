from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolPolicy:
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    allow_file_write: bool = False

    def resolve_allowed_roots(self) -> list[Path]:
        return [Path(root).expanduser().resolve() for root in self.allowed_roots]
