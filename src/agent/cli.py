from __future__ import annotations

import argparse
import sys

from src.agent import AgentConfig, Agent
from src.core.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fox-agent",
        description="A helpful assistant.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Run a single prompt, print the response, then exit.",
    )
    parser.add_argument(
        "--plan-mode",
        choices=["auto", "enable", "disable"],
        help="Enable plan mode.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["auto", "disable"],
        help="Enable memory mode.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["auto", "disable"],
        help="Enable retrieval mode.",
    )
    return parser


def run_interactive(
    agent: Agent,
    plan_mode: str,
    memory_mode: str,
    retrieval_mode: str,
) -> None:
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break

        if not user_input:
            continue

        match user_input.lower():
            case "exit" | "quit":
                print("Assistant: Bye~")
                break
            case "history":
                print("=== History ===")
                for message in agent.get_history():
                    print(f"[{message.role}] {message.content}")
                print()
                continue
            case "clear":
                agent.clear()
                print("History cleared!")
                continue
            case _:
                response = agent.run(
                    user_input,
                    plan_mode,
                    memory_mode,
                    retrieval_mode,
                )
                print("Assistant:", response.content)
                print()


def run_once(
    agent: Agent,
    prompt: str,
    plan_mode: str,
    memory_mode: str,
    retrieval_mode: str,
) -> None:
    response = agent.run(
        prompt,
        plan_mode,
        memory_mode,
        retrieval_mode,
    )
    print(response.content)


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    config = AgentConfig.from_env()
    agent = Agent(config)
    piped_input = sys.stdin.read().strip() if not sys.stdin.isatty() else ""

    if args.prompt is not None:
        prompt = args.prompt
        if piped_input:
            prompt = f"{prompt}\n\nInput from stdin:\n{piped_input}"
        run_once(agent, prompt, args.plan_mode, args.memory_mode, args.retrieval_mode)
    elif piped_input:
        run_once(
            agent, piped_input, args.plan_mode, args.memory_mode, args.retrieval_mode
        )
    else:
        run_interactive(agent, args.plan_mode, args.memory_mode, args.retrieval_mode)
