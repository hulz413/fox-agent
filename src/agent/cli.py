from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

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
        "--ingest",
        help="Build and persist the knowledge index from the given path, then exit.",
    )
    parser.add_argument(
        "--eval",
        help="Run retrieval evaluation from a JSONL file, then exit.",
        type=Path,
    )
    parser.add_argument(
        "--eval-top-k",
        help="Top-k used for retrieval evaluation.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--search",
        help="Search the knowledge index and print retrieved chunks, then exit.",
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


def run_ingest(config: AgentConfig, ingest_path: str) -> None:
    config = replace(config, knowledge_base_path=ingest_path)
    Agent(config)
    print(f"Knowledge index built and saved to: {config.knowledge_index_path}")


def _load_eval_cases(file_path: str) -> list[dict]:
    path = Path(file_path).expanduser()
    cases: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)
            if "query" not in item or "expected_sources" not in item:
                raise ValueError(
                    f"Invalid eval case at line {line_number}: "
                    "query and expected_sources are required"
                )
            cases.append(item)

    if not cases:
        raise ValueError(f"No eval cases found in: {path}")

    return cases


def _first_hit_rank(retrieved, expected_sources: list[str]) -> int | None:
    for rank, item in enumerate(retrieved, start=1):
        if any(expected in item.chunk.source for expected in expected_sources):
            return rank
    return None


def run_eval(agent: Agent, eval_path: str, top_k: int) -> None:
    cases = _load_eval_cases(eval_path)
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0.0

    for index, case in enumerate(cases, start=1):
        query = case["query"]
        expected_sources = case["expected_sources"]
        retrieved = agent.search_knowledge(query, top_k)
        rank = _first_hit_rank(retrieved, expected_sources)

        if rank == 1:
            hit_at_1 += 1
        if rank is not None:
            hit_at_k += 1
            reciprocal_rank_sum += 1.0 / rank

        status = f"hit@{rank}" if rank is not None else "miss"

        print(f"[{index}] {status} query={query}")
        print(f"    expected={expected_sources}")
        if retrieved:
            print("    retrieved:")
            for rank_index, item in enumerate(retrieved, start=1):
                print(
                    f"      {rank_index}. "
                    f"{item.chunk.source} "
                    f"score={item.score:.4f} "
                    f"chunk={item.chunk.index} "
                    f"strategy={item.chunk.metadata.get('chunk_strategy', 'unknown')}"
                )
        else:
            print("    retrieved=[]")

    total = len(cases)
    print()
    print(f"cases={total}")
    print(f"hit@1={hit_at_1 / total:.2%}")
    print(f"hit@{top_k}={hit_at_k / total:.2%}")
    print(f"mrr={reciprocal_rank_sum / total:.4f}")


def run_search(agent: Agent, search_query: str) -> None:
    print(agent.render_knowledge_search(search_query))


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    config = AgentConfig.from_env()
    if args.ingest:
        run_ingest(config, args.ingest)
        return

    agent = Agent(config)
    if args.eval:
        run_eval(agent, args.eval, args.eval_top_k)
        return

    if args.search is not None:
        run_search(agent, args.search)
        return

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
