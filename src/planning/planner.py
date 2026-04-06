import json
from src.llm.client import LLMClient
from src.llm.schemas import Message
from src.planning.schemas import Plan, PlanStep


class Planner:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def create_plan(self, user_input: str) -> Plan:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a planning assistant for an AI agent. "
                    "Break the user's request into 1-5 concise executable steps. "
                    "Return valid JSON only. "
                    'Use this exact schema: {"steps":[{"step_id":1,"description":"...","requires_tools":true}]}. '
                    "Do not solve the task, do not call any tools. "
                    "Each description must be one sentence. "
                    "Set requires_tools=true only when the step needs external actions such as file access, time lookup, or memory access."
                ),
            ),
            Message(
                role="user",
                content=user_input,
            ),
        ]

        response = self.client.chat(messages=messages)
        steps = self._parse_steps(response.content)
        if not steps:
            raise ValueError(
                f"Planner response must contain at least one step: {steps}"
            )

        return Plan(original_request=user_input, steps=steps)

    def _parse_steps(self, content: str) -> list[PlanStep]:
        normalized = content.strip()
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            normalized = "\n".join(lines[1:-1]).strip()

        data = json.loads(normalized)
        if not isinstance(data, dict):
            raise ValueError(f"Planner response must be a JSON object: {normalized}")

        payload = data.get("steps")
        if not isinstance(payload, list):
            raise ValueError(f"Planner must contain a steps list: {normalized}")

        steps: list[PlanStep] = []
        for index, step in enumerate(payload, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Each step must be a JSON object: [{index}] {step}")

            description = str(step.get("description", "")).strip()
            if not description:
                raise ValueError(
                    f"Each step must contain a description: [{index}] {step}"
                )

            step_id = int(step.get("step_id", index))
            requires_tools = bool(step.get("requires_tools", True))
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    requires_tools=requires_tools,
                )
            )

        return steps
