from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from src.core.config import Config as LLMConfig
from src.runtime.session_config import SessionConfig
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
    retrieval_mode: Literal["auto", "disable"] = "disable"
    max_steps: int = 10
    max_memory_records: int = 8
    retrieval_top_k: int = 5
    retrieval_min_score: float | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    direct_memory_namespaces: list[str] = field(default_factory=lambda: ["user"])
    planned_memory_namespaces: list[str] = field(
        default_factory=lambda: ["user", "project"]
    )
    memory_store_path: str = "~/.fox-agent/memory_store.json"
    knowledge_base_path: str | None = None
    knowledge_index_path: str = "~/.fox-agent/knowledge_index.json"
    embedding_provider: Literal["simple", "openai"] = "simple"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_timeout: float = 60.0
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    allow_file_write: bool = False
    system_prompt: str | None = None

    @classmethod
    def from_env(cls) -> AgentConfig:
        llm_config = LLMConfig.from_env()
        retrieval_min_score_raw = os.getenv("FOX_AGENT_RETRIEVAL_MIN_SCORE", "").strip()

        return cls(
            api_key=llm_config.chat_api_key,
            base_url=llm_config.chat_base_url,
            model=llm_config.chat_model,
            timeout=llm_config.chat_timeout,
            temperature=llm_config.chat_temperature,
            plan_mode=os.getenv("FOX_AGENT_PLAN_MODE", "disable").strip(),
            memory_mode=os.getenv("FOX_AGENT_MEMORY_MODE", "disable").strip(),
            retrieval_mode=os.getenv("FOX_AGENT_RETRIEVAL_MODE", "disable").strip(),
            max_steps=int(os.getenv("FOX_AGENT_MAX_STEPS", "10").strip()),
            max_memory_records=int(
                os.getenv("FOX_AGENT_MAX_MEMORY_RECORDS", "8").strip()
            ),
            retrieval_top_k=int(os.getenv("FOX_AGENT_RETRIEVAL_TOP_K", "5").strip()),
            retrieval_min_score=(
                float(retrieval_min_score_raw) if retrieval_min_score_raw else None
            ),
            chunk_size=int(os.getenv("FOX_AGENT_CHUNK_SIZE", "1000").strip()),
            chunk_overlap=int(os.getenv("FOX_AGENT_CHUNK_OVERLAP", "200").strip()),
            memory_store_path=os.getenv(
                "FOX_AGENT_MEMORY_STORE_PATH", "~/.fox-agent/memory_store.json"
            ).strip(),
            knowledge_base_path=os.getenv("FOX_AGENT_KNOWLEDGE_BASE_PATH", "").strip()
            or None,
            knowledge_index_path=os.getenv(
                "FOX_AGENT_KNOWLEDGE_INDEX_PATH", "~/.fox-agent/knowledge_index.json"
            ).strip(),
            embedding_provider=os.getenv(
                "FOX_AGENT_EMBEDDING_PROVIDER", "simple"
            ).strip(),
            embedding_api_key=os.getenv("FOX_AGENT_EMBEDDING_API_KEY", "").strip()
            or None,
            embedding_base_url=os.getenv("FOX_AGENT_EMBEDDING_BASE_URL", "").strip()
            or None,
            embedding_model=os.getenv("FOX_AGENT_EMBEDDING_MODEL", "").strip() or None,
            embedding_timeout=float(
                os.getenv("FOX_AGENT_EMBEDDING_TIMEOUT", "60.0").strip()
            ),
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
            chat_api_key=self.api_key,
            chat_base_url=self.base_url,
            chat_model=self.model,
            chat_timeout=self.timeout,
            chat_temperature=self.temperature,
            embedding_provider=self.embedding_provider,
            embedding_api_key=self.embedding_api_key,
            embedding_base_url=self.embedding_base_url,
            embedding_model=self.embedding_model,
            embedding_timeout=self.embedding_timeout,
        )

    def to_session_config(self) -> SessionConfig:
        return SessionConfig(
            plan_mode=self.plan_mode,
            memory_mode=self.memory_mode,
            retrieval_mode=self.retrieval_mode,
            max_steps=self.max_steps,
            max_memory_records=self.max_memory_records,
            retrieval_top_k=self.retrieval_top_k,
            retrieval_min_score=self.retrieval_min_score,
            direct_memory_namespaces=list(self.direct_memory_namespaces),
            planned_memory_namespaces=list(self.planned_memory_namespaces),
        )

    def to_tool_policy(self) -> ToolPolicy:
        return ToolPolicy(
            allowed_roots=list(self.allowed_roots),
            allow_file_write=self.allow_file_write,
        )
