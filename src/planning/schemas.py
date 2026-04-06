from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanStep:
    step_id: int
    description: str
    requires_tools: bool = True


@dataclass(frozen=True)
class Plan:
    original_request: str
    steps: list[PlanStep] = field(default_factory=list)
