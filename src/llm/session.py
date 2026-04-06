import logging
from src.llm.client import LLMClient
from src.llm.schemas import Message, LLMResponse, ToolCall
from src.tools.registry import ToolRegistry
from src.planning.schemas import Plan, PlanStep

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(
        self,
        client: LLMClient,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry or ToolRegistry()
        self.messages: list[Message] = []
        self.max_steps = max_steps

        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content: str) -> None:
        self.messages.append(Message(role="system", content=content))

    def add_assistant_message(
        self, content: str, tool_calls: list[ToolCall] | None = None
    ) -> None:
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.messages.append(
            Message(role="tool", content=content, tool_call_id=tool_call_id)
        )

    def get_history(self) -> list[Message]:
        return list(self.messages)

    def chat(self, user_input: str) -> LLMResponse:
        logger.info(f"User input: {user_input}")
        self.add_user_message(user_input)
        return self._run_loop()

    def chat_with_plan(self, plan: Plan) -> LLMResponse:
        logger.info(f"Plan execution started with {len(plan.steps)} steps")
        step_results: list[str] = []

        for step in plan.steps:
            logger.info(f"Plan step started: [{step.step_id}] {step.description}")
            response = self._execute_plan_step(plan, step, step_results)
            step_results.append(f"Step {step.step_id}: {response.content}")
            logger.info(f"Plan step finished: [{step.step_id}] {response.content}")

        final_input = self._build_final_input(plan, step_results)
        logger.info("Generating final response from plan execution results")
        self.add_user_message(final_input)
        return self._run_loop()

    def _run_loop(self) -> LLMResponse:
        for step in range(1, self.max_steps + 1):
            logger.info(f"Step {step}/{self.max_steps} started")
            response = self.client.chat(
                messages=self.messages, tools=self.tool_registry.list()
            )

            if response.usage:
                logger.info(
                    "Model usage: "
                    f"prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

            self.add_assistant_message(
                content=response.content, tool_calls=response.tool_calls
            )

            if not response.tool_calls:
                logger.info(f"Assistant final response: {response.content}")
                return response

            if response.content:
                logger.info(f"Assistant response: {response.content}")

            for tool_call in response.tool_calls:
                logger.info(
                    f"Calling tool: {tool_call.name} with arguments={tool_call.arguments}"
                )
                tool = self.tool_registry.get(tool_call.name)

                try:
                    result = tool.handler(**tool_call.arguments)
                    logger.info(f"Tool calling {tool_call.name} result: {result}")
                except Exception as e:
                    logger.exception(f"Tool calling {tool_call.name} failed")
                    result = f"Tool call {tool_call.name} failed: {str(e)}"
                self.add_tool_message(result, tool_call.id)

        raise ValueError("Max steps reached without a response :(")

    def _execute_plan_step(
        self, plan: Plan, step: PlanStep, step_results: list[str]
    ) -> LLMResponse:
        step_input = self._build_step_input(plan, step, step_results)
        self.add_user_message(step_input)
        return self._run_loop()

    def _build_step_input(
        self, plan: Plan, step: PlanStep, step_results: list[str]
    ) -> str:
        completed_results = "\n".join(step_results) if step_results else "None"
        return (
            f"Original user request: {plan.original_request}\n\n"
            f"Current step ({step.step_id}/{len(plan.steps)}): {step.description}\n"
            f"Requires tools: {step.requires_tools}\n\n"
            f"Completed step results:\n{completed_results}\n\n"
            "Complete only the current step. "
            "Use tools only if the current step requires them. "
            "Do not work on other steps. "
            "Return a concise result for this step."
        )

    def _build_final_input(self, plan: Plan, step_results: list[str]) -> str:
        plan_lines = [
            f"{step.step_id}. [{('tool' if step.requires_tools else 'no-tool')}] {step.description}"
            for step in plan.steps
        ]
        rendered_plan = "\n".join(plan_lines)
        rendered_results = "\n".join(step_results)

        return (
            f"Original user request: {plan.original_request}\n\n"
            f"Plan: {rendered_plan}\n\n"
            f"Plan step results:\n{rendered_results}\n\n"
            "Base on the executed step results, give the final answer to the user."
        )

    def clear(self) -> None:
        self.messages = [
            message for message in self.messages if message.role == "system"
        ]
        logger.info("Session cleared, system message preserved")
