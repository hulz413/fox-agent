import logging

from src.core.config import Config
from src.llm.client import LLMClient
from src.llm.session import ChatSession
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.planning.planner import Planner
from src.planning.schemas import Plan


def format_plan(plan: Plan) -> str:
    lines = ["=== Plan ==="]
    for step in plan.steps:
        tool_hint = "tool" if step.requires_tools else "no-tool"
        lines.append(f"{step.step_id}. [{tool_hint}] {step.description}")
    return "\n".join(lines)


def execute_plan(session: ChatSession, plan: Plan) -> list[str]:
    step_results: list[str] = []
    total_steps = len(plan.steps)

    for step in plan.steps:
        completed_results = "\n".join(step_results) if step_results else "None"
        step_input = (
            f"Original user request: {plan.original_request}\n\n"
            f"Current step ({step.step_id}/{total_steps}): {step.description}\n"
            f"Requires tools: {step.requires_tools}\n\n"
            f"Completed step results:\n{completed_results}\n\n"
            "Complete only the current step. "
            "Use tools only if the current step requires them. "
            "Return a concise result for this step."
        )
        response = session.chat(step_input)
        step_results.append(f"Step {step.step_id}: {response.content}")

    return step_results


# Example input cases:
# List files in src/tools, then read builtins.py and summarize the available tools.
# Read src/llm/session.py, explain how the agent loop works, then summarize it in 3 bullet points.
# Remember that this project is for learning AI Agent development, then tell me what was stored.
# Get the current time, save it to a file, then read the file back and summarize the result.
# Read requirements.txt and explain what dependencies are used in this project.
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
    )

    config = Config.from_env()
    client = LLMClient(config)
    planner = Planner(client)
    tool_registry = ToolRegistry()
    for tool in build_builtin_tools():
        tool_registry.register(tool)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        match user_input.lower():
            case "exit" | "quit":
                print("Assistant: Bye~")
                break
            case _:
                plan = planner.create_plan(user_input)
                print(format_plan(plan))
                print()

                session = ChatSession(
                    client,
                    tool_registry,
                    system_prompt=(
                        "You are a helpful assistant executing a structured plan. "
                        "Complete only the current step. "
                        "Use tools when needed. "
                        "Do not jump ahead to future steps. "
                        "Be concise."
                    ),
                    max_steps=20,
                )

                step_results = execute_plan(session, plan)
                print("=== Step Results ===")
                for result in step_results:
                    print(result)
                print()

                final_input = (
                    f"User request: {user_input}\n\n"
                    f"Plan:\n{format_plan(plan)}\n\n"
                    f"Step results:\n{'\n'.join(step_results)}\n\n"
                    "Base on the executed steps, give the final answer to the user."
                )

                response = session.chat(final_input)
                print("Assistant:", response.content)
                print()


if __name__ == "__main__":
    main()
