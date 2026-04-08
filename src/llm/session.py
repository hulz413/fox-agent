import logging
from typing import Literal
from src.llm.client import LLMClient
from src.llm.schemas import Message, LLMResponse, ToolCall
from src.llm.session_config import SessionConfig
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.policy import ToolPolicy
from src.planning.planner import Planner
from src.planning.schemas import Plan, PlanStep
from src.memory.store import MemoryStore
from src.memory.json_store import JsonMemoryStore
from src.memory.retrieval import MemoryRetriever

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(
        self,
        client: LLMClient,
        tool_registry: ToolRegistry | None = None,
        tool_policy: ToolPolicy | None = None,
        planner: Planner | None = None,
        memory_store: MemoryStore | None = None,
        config: SessionConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry or ToolRegistry()
        self.planner = planner or Planner(client)
        self.memory_store = memory_store or JsonMemoryStore()
        self.memory_retriever = MemoryRetriever(self.memory_store)
        self.config = config or SessionConfig()
        self.tool_policy = tool_policy or ToolPolicy()
        self.tool_executor = ToolExecutor(self.tool_policy)
        self.messages: list[Message] = []
        self.max_steps = self.config.max_steps
        self._last_memory_context: str | None = None

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

    def chat(
        self,
        user_input: str,
        plan_mode: Literal["auto", "enable", "disable"] | None = None,
        memory_mode: Literal["auto", "disable"] | None = None,
    ) -> LLMResponse:
        plan_mode = plan_mode or self.config.plan_mode
        memory_mode = memory_mode or self.config.memory_mode

        logger.info(f"User input: {user_input}")
        logger.info(f"Plan mode: {plan_mode}")
        logger.info(f"Memory mode: {memory_mode}")

        if memory_mode == "auto":
            self._inject_memory_context(
                namespaces=self.config.resolve_memory_namespaces(),
                max_records=self.config.max_memory_records,
            )

        match plan_mode:
            case "auto":
                should_plan = self._should_plan(user_input)
                logger.info(f"Auto plan decision: {should_plan}")
                if should_plan:
                    plan = self.planner.create_plan(user_input)
                    return self.chat_with_plan(plan)
                else:
                    self.add_user_message(user_input)
                    return self._run_loop()
            case "enable":
                plan = self.planner.create_plan(user_input)
                return self.chat_with_plan(plan)
            case "disable":
                self.add_user_message(user_input)
                return self._run_loop()
            case _:
                raise ValueError(f"Unknown plan mode: {plan_mode}")

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

    def _inject_memory_context(
        self, namespaces: list[str] | None = None, max_records: int = 10
    ) -> None:
        context = self.memory_retriever.build_context(
            namespaces=namespaces, max_records=max_records
        )
        if not context:
            logger.info("No relevant memory context retrieved")
            return

        if self._last_memory_context == context:
            logger.info("Memory context unchanged, skipping injection")
            return

        logger.info("Injecting retrieved memory into session context")
        self.add_system_message(context)
        self._last_memory_context = context

    def _should_plan(self, user_input: str) -> bool:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a routing assistant for an AI agent. "
                    "Decide whether the user's request needs planning before execution. "
                    "Planning is useful for multi-step tasks, tasks with dependencies, "
                    "tasks requiring file/tool access across several steps, or tasks that "
                    "must be broken down before answering. "
                    "Reply with exactly one word: yes or no."
                ),
            ),
            Message(role="user", content=user_input),
        ]
        response = self.client.chat(messages)
        decision = response.content.strip().lower()
        return decision == "yes"

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

                result = self.tool_executor.execute(tool, tool_call.arguments)
                if result.success:
                    logger.info(
                        f"Tool calling {tool_call.name} result: {result.content}"
                    )
                else:
                    logger.error(
                        f"Tool calling {tool_call.name} failed: {result.error}"
                    )
                self.add_tool_message(result.to_message(tool_call.name), tool_call.id)

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
            f"Plan:\n{rendered_plan}\n\n"
            f"Plan step results:\n{rendered_results}\n\n"
            "Based on the executed step results, give the final answer to the user."
        )

    def clear(self) -> None:
        self.messages = [
            message for message in self.messages if message.role == "system"
        ]
        self._last_memory_context = None
        logger.info("Session cleared, system message preserved")
