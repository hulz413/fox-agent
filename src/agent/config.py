from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from src.core.config import Config as LLMConfig
from src.llm.session_config import SessionConfig
from src.tools.policy import ToolPolicy


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    temperature: float = 0.7

    plan_mode: Literal["auto", "enable", "disable"] = "disable"
    memory_mode: Literal["auto", "disable"] = "disable"
    max_steps: int = 10
    max_memory_records: int = 8
    direct_memory_namespaces: list[str] = field(default_factory=lambda: ["user"])
    planned_memory_namespaces: list[str] = field(
        default_factory=lambda: ["user", "project"]
    )
    memory_store_path: str = "~/.fox-agent/memory_store.json"
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    allow_file_write: bool = False
    system_prompt: str | None = None

    @classmethod
    def from_env(cls) -> AgentConfig:
        llm_config = LLMConfig.from_env()

        return cls(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
            timeout=llm_config.timeout,
            temperature=llm_config.temperature,
            plan_mode=os.getenv("FOX_AGENT_PLAN_MODE", "disable").strip(),
            memory_mode=os.getenv("FOX_AGENT_MEMORY_MODE", "disable").strip(),
            max_steps=int(os.getenv("FOX_AGENT_MAX_STEPS", "10").strip()),
            max_memory_records=int(
                os.getenv("FOX_AGENT_MAX_MEMORY_RECORDS", "8").strip()
            ),
            memory_store_path=os.getenv(
                "FOX_AGENT_MEMORY_STORE_PATH", "~/.fox-agent/memory_store.json"
            ).strip(),
            allowed_roots=[
                item.strip()
                for item in os.getenv("FOX_AGENT_ALLOWED_ROOTS", ".").split(",")
                if item.strip()
            ],
            allow_file_write=(
                os.getenv("FOX_AGENT_ALLOW_FILE_WRITE", "false").strip().lower()
                == "true"
            ),
            system_prompt=os.getenv("FOX_AGENT_SYSTEM_PROMPT", "").strip() or None,
        )

    def to_llm_config(self) -> LLMConfig:
        return LLMConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            temperature=self.temperature,
        )

    def to_session_config(self) -> SessionConfig:
        return SessionConfig(
            plan_mode=self.plan_mode,
            memory_mode=self.memory_mode,
            max_steps=self.max_steps,
            max_memory_records=self.max_memory_records,
            direct_memory_namespaces=list(self.direct_memory_namespaces),
            planned_memory_namespaces=list(self.planned_memory_namespaces),
        )

    def to_tool_policy(self) -> ToolPolicy:
        return ToolPolicy(
            allowed_roots=list(self.allowed_roots),
            allow_file_write=self.allow_file_write,
        )
