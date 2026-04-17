import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionConfig:
    plan_mode: Literal["auto", "enable", "disable"] = "disable"
    memory_mode: Literal["auto", "disable"] = "disable"
    retrieval_mode: Literal["auto", "disable"] = "disable"
    max_steps: int = 10
    max_memory_records: int = 5
    retrieval_top_k: int = 5
    retrieval_min_score: float | None = None
    direct_memory_namespaces: list[str] = field(default_factory=lambda: ["user"])
    planned_memory_namespaces: list[str] = field(
        default_factory=lambda: ["user", "project"]
    )

    def resolve_memory_namespaces(self) -> list[str]:
        match self.plan_mode:
            case "auto" | "enable":
                return self.planned_memory_namespaces
            case "disable":
                return self.direct_memory_namespaces
            case _:
                raise ValueError(f"Unknown plan mode: {self.plan_mode}")
