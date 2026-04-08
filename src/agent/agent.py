from __future__ import annotations

from typing import Literal
from src.agent.config import AgentConfig
from src.llm.client import LLMClient
from src.llm.schemas import LLMResponse, Message
from src.llm.session import ChatSession
from src.memory.json_store import JsonMemoryStore
from src.planning.planner import Planner
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolDefinition


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = LLMClient(config.to_llm_config())
        self.memory_store = JsonMemoryStore(config.memory_store_path)
        self.tool_registry = ToolRegistry()

        for tool in build_builtin_tools(self.memory_store):
            self.tool_registry.register(tool)

        self.planner = Planner(self.client)
        self.session = ChatSession(
            client=self.client,
            tool_registry=self.tool_registry,
            tool_policy=config.to_tool_policy(),
            planner=self.planner,
            memory_store=self.memory_store,
            config=config.to_session_config(),
            system_prompt=config.system_prompt,
        )

    def run(
        self,
        user_input: str,
        plan_mode: Literal["auto", "enable", "disable"] | None = None,
        memory_mode: Literal["auto", "disable"] | None = None,
    ) -> LLMResponse:
        return self.session.chat(
            user_input=user_input,
            plan_mode=plan_mode,
            memory_mode=memory_mode,
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tool_registry.register(tool)

    def clear(self) -> None:
        self.session.clear()

    def get_history(self) -> list[Message]:
        return self.session.get_history()
